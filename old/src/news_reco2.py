import re
import json
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from dateutil.parser import isoparse
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# ----------------------------
# Utils
# ----------------------------


def now_utc() -> datetime:
	return datetime.now(timezone.utc)


def parse_date(d: Any) -> Optional[datetime]:
	if not d:
		return None
	if isinstance(d, datetime):
		dt = d
	else:
		try:
			dt = isoparse(str(d))
		except Exception:
			return None

	# If naive, assume UTC (avoid local-time shifting)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


def norm_text(s: str) -> str:
	s = (s or "").strip().lower()
	s = re.sub(r"\s+", " ", s)
	return s



def article_fingerprint(title: str, text: str, url: str = "", n_chars: int = 400) -> str:
	u = norm_text(url)
	t = norm_text(title)
	lead = norm_text((text or "")[:n_chars])
	raw = (u + "||" + t + "||" + lead).encode("utf-8", errors="ignore")
	return hashlib.md5(raw).hexdigest()


def article_id_from_url(url: str) -> str:
	hex_id = hashlib.md5((url or "").encode("utf-8", errors="ignore")).hexdigest()
	return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"


# ----------------------------
# Data model
# ----------------------------


@dataclass
class Article:
	id: str
	title: str
	text: str
	url: str
	domain: str
	date: Optional[datetime]
	canonical_text: str = ""
	fingerprint: str = ""
	embedding: Optional[np.ndarray] = None

	@staticmethod
	def from_dict(d: Dict[str, Any]) -> "Article":
		title = d.get("title") or ""
		desc = d.get("description") or d.get("summary") or ""
		text = d.get("text") or d.get("content") or d.get("article") or ""
		url = d.get("url") or ""

		domain = d.get("domain") or ""
		if not domain and url:
			domain = urlparse(url).netloc
		if not domain:
			domain = d.get("source") or ""

		dt = parse_date(
			d.get("date")
			or d.get("published_at")
			or d.get("published_date")
			or d.get("datetime")
		)

		if d.get("id"):
			aid = d.get("id")
		else:
			aid = article_id_from_url(url)
			if not aid:
				hex_id = hashlib.md5((title + desc).encode()).hexdigest()
				aid = f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"

		lead = (text or "")[:1200]
		canonical = f"{title}\n{desc}\n{lead}".strip()
		fp = d.get("fingerprint") or article_fingerprint(title, text, url=url)

		return Article(
			id=str(aid),
			title=title,
			text=text,
			url=url,
			domain=domain,
			date=dt,
			canonical_text=canonical,
			fingerprint=fp,
		)


# ----------------------------
# Embedding (Qwen3-Embedding-8B)
# ----------------------------


class QwenEmbedder:
	def __init__(
		self,
		model_name: str = "Qwen/Qwen3-Embedding-8B",
		max_length: int = 512,
		task: str = "Given a web search query, retrieve relevant passages that answer the query",
	):
		self.model_name = model_name
		self.max_length = int(max_length)
		self.task = task

		from transformers import AutoTokenizer, AutoModel

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name,
			trust_remote_code=True,
			padding_side="left",
		)
		if getattr(self.tokenizer, "pad_token", None) is None:
			# Many decoder-only models don't define pad_token.
			self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or self.tokenizer.pad_token

		# Qwen3-Embedding-8B is very large; on 16GB GPUs it often OOMs if fully loaded.
		# If accelerate is available, use device_map='auto' with GPU headroom and CPU offload.
		model_kwargs = {
			"trust_remote_code": True,
			"torch_dtype": torch_dtype,
		}

		if self.device == "cuda":
			try:
				import accelerate  # noqa: F401

				total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
				total_gib = max(1, int(total_bytes // (1024**3)))
				# Keep ~2 GiB free for activations/caches
				max_gpu_gib = max(1, total_gib - 2)
				model_kwargs.update(
					{
						"device_map": "auto",
						"max_memory": {0: f"{max_gpu_gib}GiB", "cpu": "64GiB"},
						"low_cpu_mem_usage": True,
						"attn_implementation": "sdpa",
					}
				)
			except Exception:
				# Fallback: if we can't offload, load on CPU to avoid OOM.
				self.device = "cpu"
				model_kwargs["torch_dtype"] = torch.float32

		self.model = AutoModel.from_pretrained(model_name, **model_kwargs)

		# When device_map is used, model is already placed across devices.
		if self.device == "cpu" and not hasattr(self.model, "hf_device_map"):
			self.model.to("cpu")
		self.model.eval()

		self.input_device = self._infer_input_device()

	def _infer_input_device(self) -> str:
		# For sharded models, prefer a CUDA device if present.
		device_map = getattr(self.model, "hf_device_map", None)
		if isinstance(device_map, dict) and device_map:
			for dev in device_map.values():
				if isinstance(dev, str) and dev.startswith("cuda"):
					return dev
		return "cuda" if self.device == "cuda" else "cpu"

	@staticmethod
	def _last_token_pool(last_hidden_states, attention_mask):
		# Recommended pooling for Qwen3-Embedding models.
		left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
		if left_padding:
			return last_hidden_states[:, -1]
		sequence_lengths = attention_mask.sum(dim=1) - 1
		batch_size = last_hidden_states.shape[0]
		return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

	def _wrap_query(self, query: str) -> str:
		q = (query or "").strip()
		return f"Instruct: {self.task}\nQuery: {q}"

	def encode_queries(self, queries: List[str], batch_size: int = 8) -> np.ndarray:
		return self._encode([self._wrap_query(q) for q in queries], batch_size=batch_size)

	def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
		return self._encode(documents, batch_size=batch_size)

	def _encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:

		all_vecs: List[np.ndarray] = []
		bs = int(batch_size)
		for i in range(0, len(texts), bs):
			chunk = texts[i : i + bs]
			enc = self.tokenizer(
				chunk,
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			enc = {k: v.to(self.input_device) for k, v in enc.items()}

			with torch.inference_mode():
				out = self.model(**enc)
				last_hidden = getattr(out, "last_hidden_state", None)
				if last_hidden is None:
					raise RuntimeError("Model output has no last_hidden_state; cannot pool embeddings")
				pooled = self._last_token_pool(last_hidden, enc.get("attention_mask"))

			vecs = pooled.detach().float().cpu().numpy().astype(np.float32)
			vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
			all_vecs.append(vecs)

		if not all_vecs:
			return np.zeros((0, 0), dtype=np.float32)
		return np.vstack(all_vecs)


# ----------------------------
# Qdrant index (dense)
# ----------------------------


class DenseIndexQdrant:
	def __init__(self, url: str = "http://localhost:6333", collection: str = "news_qwen3_emb8b"):
		self.client = QdrantClient(url=url)
		self.collection = collection

	def collection_exists(self) -> bool:
		try:
			return bool(self.client.collection_exists(self.collection))
		except Exception:
			return False

	def get_points_count(self) -> int:
		try:
			info = self.client.get_collection(self.collection)
			return int(getattr(info, "points_count", 0) or 0)
		except Exception:
			return 0

	def _build_date_filter(self, days: int) -> Optional[qm.Filter]:
		if not days or days <= 0:
			return None
		gte = (now_utc() - timedelta(days=days)).isoformat()

		# Include undated items (common for synthetic or poorly-parsed articles).
		null_date = qm.IsNullCondition(is_null=qm.PayloadField(key="date"))
		empty_date = qm.IsEmptyCondition(is_empty=qm.PayloadField(key="date"))

		if hasattr(qm, "DatetimeRange"):
			in_range = qm.FieldCondition(key="date", range=qm.DatetimeRange(gte=gte))
			return qm.Filter(min_should=qm.MinShould(conditions=[in_range, null_date, empty_date], min_count=1))

		try:
			in_range = qm.FieldCondition(key="date", range=qm.Range(gte=gte))
			return qm.Filter(min_should=qm.MinShould(conditions=[in_range, null_date, empty_date], min_count=1))
		except Exception:
			return None

	def ensure_collection(self, vector_dim: int):
		if self.collection_exists():
			return
		self.client.recreate_collection(
			collection_name=self.collection,
			vectors_config=qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
		)

	def recreate(self, vector_dim: int):
		self.client.recreate_collection(
			collection_name=self.collection,
			vectors_config=qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
		)

	def upsert_articles(self, articles: List[Article], batch_size: int = 128):
		for i in range(0, len(articles), batch_size):
			batch = articles[i : i + batch_size]
			points = []
			for a in batch:
				assert a.embedding is not None
				payload = {
					"article_id": a.id,
					"title": a.title,
					"canonical_text": a.canonical_text,
					"url": a.url,
					"domain": a.domain,
					"date": a.date.isoformat() if a.date else None,
					"fingerprint": a.fingerprint,
				}
				points.append(qm.PointStruct(id=a.id, vector=a.embedding.tolist(), payload=payload))
			self.client.upsert(collection_name=self.collection, points=points)

	def search(self, query_vec: np.ndarray, limit: int = 20, days: int = 100) -> List[qm.ScoredPoint]:
		flt = self._build_date_filter(days)
		if hasattr(self.client, "search"):
			return self.client.search(
				collection_name=self.collection,
				query_vector=query_vec.tolist(),
				limit=limit,
				query_filter=flt,
				with_payload=True,
				with_vectors=False,
			)
		if hasattr(self.client, "search_points"):
			return self.client.search_points(
				collection_name=self.collection,
				query_vector=query_vec.tolist(),
				limit=limit,
				query_filter=flt,
				with_payload=True,
				with_vectors=False,
			)

		resp = self.client.query_points(
			collection_name=self.collection,
			query=query_vec.tolist(),
			limit=limit,
			query_filter=flt,
			with_payload=True,
			with_vectors=False,
		)
		return list(getattr(resp, "points", []))


# ----------------------------
# Dataset
# ----------------------------


def load_articles(path: str, limit: Optional[int] = None) -> List[Article]:
	with open(path, "r", encoding="utf-8") as f:
		first = f.read(1)
		f.seek(0)
		if first == "[":
			data = json.load(f)
			out: List[Article] = []
			for d in data:
				if not isinstance(d, dict):
					continue
				out.append(Article.from_dict(d))
				if limit and len(out) >= limit:
					break
			return out

		out: List[Article] = []
		for line in f:
			line = line.strip()
			if not line:
				continue
			d = json.loads(line)
			if not isinstance(d, dict):
				continue
			out.append(Article.from_dict(d))
			if limit and len(out) >= limit:
				break
		return out


def dedup_articles(articles: List[Article]) -> List[Article]:
	seen = set()
	out: List[Article] = []
	for a in articles:
		if a.fingerprint in seen:
			continue
		seen.add(a.fingerprint)
		out.append(a)
	return out


# ----------------------------
# Main
# ----------------------------


def main(
	data_path: str,
	qdrant_url: str = "http://localhost:6333",
	collection: str = "news_qwen3_emb8b",
	model: str = "Qwen/Qwen3-Embedding-8B",
	max_length: int = 512,
	batch_size: int = 1,
	max_articles: Optional[int] = None,
	reindex: bool = False,
	days: int = 100,
	topk: int = 20,
	interests: Optional[List[str]] = None,
	out_path: Optional[str] = None,
):
	raw = load_articles(data_path, limit=max_articles)
	articles = dedup_articles(raw)
	print(f"Loaded: {len(raw)} | After dedup: {len(articles)}")

	idx = DenseIndexQdrant(url=qdrant_url, collection=collection)
	points_count = idx.get_points_count() if idx.collection_exists() else 0
	need_index = reindex or (points_count == 0)

	embedder: Optional[QwenEmbedder] = None

	if need_index:
		embedder = QwenEmbedder(model_name=model, max_length=max_length)
		texts = [a.canonical_text for a in articles]

		all_vecs: List[np.ndarray] = []
		bs = max(1, int(batch_size))
		for i in tqdm(range(0, len(texts), bs), desc="Embedding (Qwen3)"):
			chunk = texts[i : i + bs]
			vecs = embedder.encode_documents(chunk, batch_size=bs)
			all_vecs.append(vecs)
		all_vecs = np.vstack(all_vecs)

		for a, v in zip(articles, all_vecs):
			a.embedding = v

		dim = int(all_vecs.shape[1])
		print("Embedding dim:", dim)

		if reindex:
			idx.recreate(vector_dim=dim)
		else:
			idx.ensure_collection(vector_dim=dim)
		idx.upsert_articles(articles, batch_size=128)
		print("Qdrant upsert done.")
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	else:
		print(f"Qdrant already has {points_count} points. Skipping re-embedding.")

	if embedder is None:
		embedder = QwenEmbedder(model_name=model, max_length=max_length)

	use_interests = [i for i in (interests or []) if i and i.strip()]
	if not use_interests:
		use_interests = [
			"Guerres et conflits internationaux",
			"IA et les LLMs",
			"Politique Française",
			"SpaceX",
			"Apple",
		]

	print("\n--- FEED (PER INTEREST, Qwen3-Embedding-8B) ---")
	total = 0
	results_export: List[Dict[str, Any]] = []
	for j, interest in enumerate(use_interests, 1):
		q = embedder.encode_queries([interest], batch_size=1)[0]
		hits = idx.search(q, limit=topk, days=days)
		block: Dict[str, Any] = {
			"interest": interest,
			"n": len(hits),
			"hits": [],
		}

		print(f"\n### Interest {j}/{len(use_interests)}: {interest} (n={len(hits)})")
		for i, h in enumerate(hits, 1):
			total += 1
			payload = getattr(h, "payload", None) or {}
			block["hits"].append(
				{
					"rank": i,
					"id": str(getattr(h, "id", "")),
					"score": float(getattr(h, "score", 0.0) or 0.0),
					"payload": payload,
				}
			)
			title = payload.get("title") or ""
			domain = payload.get("domain") or ""
			url = payload.get("url") or ""
			date = payload.get("date") or "no-date"
			print(f"{i:02d}. [{date}] ({domain}) {title}\n    {url}\n")
		results_export.append(block)

	print(f"Total articles printed: {total}")

	if out_path:
		out_dir = os.path.dirname(out_path)
		if out_dir:
			os.makedirs(out_dir, exist_ok=True)
		payload = {
			"generated_at": now_utc().isoformat(),
			"data_path": data_path,
			"qdrant_url": qdrant_url,
			"collection": collection,
			"model": model,
			"max_length": max_length,
			"days": days,
			"topk": topk,
			"results": results_export,
		}
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		print(f"Saved retrieval results to: {out_path}")


if __name__ == "__main__":
	import argparse

	default_data_path = str((Path(__file__).resolve().parent / "merged_dataset.json"))

	parser = argparse.ArgumentParser(description="News retrieval (Qwen3-Embedding-8B + Qdrant only)")
	parser.add_argument(
		"data_path",
		nargs="?",
		default=default_data_path,
		help=f"Path to JSON or JSONL dataset (default: {default_data_path})",
	)
	parser.add_argument("--qdrant-url", default="http://localhost:6333")
	parser.add_argument("--collection", default="news_qwen3_emb8b")
	parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
	parser.add_argument("--max-length", type=int, default=512)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1,
		help="Embedding batch size for indexing (Qwen3-Embedding-8B is memory heavy)",
	)
	parser.add_argument("--max-articles", type=int, default=None)
	parser.add_argument("--reindex", action="store_true")
	parser.add_argument(
		"--days",
		type=int,
		default=100,
		help="Dense retrieval date window in days (Qdrant filter). Use 0 to disable.",
	)
	parser.add_argument("--topk", type=int, default=20)
	parser.add_argument(
		"--interest",
		action="append",
		default=None,
		help="Interest phrase (repeat for multiple). Default: 5 preset interests.",
	)
	parser.add_argument(
		"--out",
		default=None,
		help="Write retrieval candidates to JSON (grouped by interest).",
	)

	args = parser.parse_args()

	main(
		data_path=args.data_path,
		qdrant_url=args.qdrant_url,
		collection=args.collection,
		model=args.model,
		max_length=args.max_length,
		batch_size=args.batch_size,
		max_articles=args.max_articles,
		reindex=args.reindex,
		days=args.days,
		topk=args.topk,
		interests=args.interest,
		out_path=args.out,
	)