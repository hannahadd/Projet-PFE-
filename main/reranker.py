from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np

try:
	from qdrant_client import QdrantClient
	_HAS_QDRANT = True
except Exception:
	QdrantClient = None
	_HAS_QDRANT = False

from db import PostgresStore


ROOT = Path(__file__).resolve().parents[1]
SRC_RERANKER_PATH = Path(__file__).resolve().parent / "reranker_core.py"

try:
	from rapidfuzz import fuzz
	_RAPIDFUZZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
	fuzz = None
	_RAPIDFUZZ_IMPORT_ERROR = exc


def _require_dependencies() -> None:
	if _RAPIDFUZZ_IMPORT_ERROR is not None:
		raise RuntimeError(
			"Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz"
		) from _RAPIDFUZZ_IMPORT_ERROR
	if not _HAS_QDRANT:
		raise RuntimeError(
			"Missing dependency 'qdrant-client'. Install with: pip install qdrant-client"
		)


def _load_src_reranker_module():
	spec = importlib.util.spec_from_file_location("src_reranker_core", SRC_RERANKER_PATH)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to load module from {SRC_RERANKER_PATH}")
	mod = importlib.util.module_from_spec(spec)
	sys.modules[spec.name] = mod
	spec.loader.exec_module(mod)
	return mod


core = _load_src_reranker_module()
DEFAULT_INSTRUCTION = core.DEFAULT_INSTRUCTION


_STOP_TOKENS = {
	"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with",
	"from", "into", "about", "after", "before", "over", "under", "amid", "during",
	"this", "that", "these", "those", "new", "more", "most", "make", "makes", "made",
	"will", "would", "could", "should", "can", "may", "might", "than", "then", "its",
	"their", "his", "her", "our", "your", "is", "are", "was", "were", "be", "been",
	"being", "as", "it", "they", "he", "she", "we", "you",
	"unveils", "unveiled", "announces", "announced", "launches", "launched",
	"architecture", "framework", "system", "platform", "model", "models",
	"company", "companies", "startup", "lab", "labs", "news", "report", "reports",
	"toward", "across", "ahead", "behind", "there", "here",
}


def normalize_text(text: str) -> str:
	s = unicodedata.normalize("NFKC", (text or "")).strip().casefold()
	s = re.sub(r"\s+", " ", s)
	return s


def normalize_url(url: str) -> str:
	raw = (url or "").strip()
	if not raw:
		return ""
	parts = urlsplit(raw)
	cleaned = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
	return cleaned.rstrip("/")


def _clean_token(tok: str) -> str:
	t = tok.strip().casefold().strip("'_-/")
	if t.endswith("'s"):
		t = t[:-2]
	return t


def tokenize(text: str) -> List[str]:
	s = normalize_text(text)
	tokens = re.findall(r"\b[\w'-]+\b", s, flags=re.UNICODE)
	out: List[str] = []
	for tok in tokens:
		clean = _clean_token(tok)
		if len(clean) <= 1:
			continue
		out.append(clean)
	return out


def jaccard_set(a: Set[str], b: Set[str]) -> float:
	if not a or not b:
		return 0.0
	return float(len(a & b) / max(1, len(a | b)))


def ratio(a: str, b: str, score_cutoff: int = 0) -> float:
	if not a or not b:
		return 0.0
	score = fuzz.ratio(a, b, score_cutoff=score_cutoff)
	if score is None:
		return 0.0
	return float(score) / 100.0


def token_set_ratio(a: str, b: str, score_cutoff: int = 0) -> float:
	if not a or not b:
		return 0.0
	score = fuzz.token_set_ratio(a, b, score_cutoff=score_cutoff)
	if score is None:
		return 0.0
	return float(score) / 100.0


def payload_text(payload: Dict[str, Any]) -> str:
	return str(
		payload.get("canonical_text")
		or payload.get("text")
		or payload.get("content")
		or payload.get("body")
		or payload.get("description")
		or ""
	)


def _pick_field(hit: Dict[str, Any], *candidates: str) -> str:
	payload = hit.get("payload") or {}
	full = hit.get("full_article") or {}
	for key in candidates:
		value = payload.get(key)
		if value:
			return str(value)
	for key in candidates:
		value = full.get(key)
		if value:
			return str(value)
	return ""


def _important_overlap(a: Set[str], b: Set[str]) -> Set[str]:
	out: Set[str] = set()
	for tok in (a & b):
		if tok in _STOP_TOKENS:
			continue
		if len(tok) >= 3 or any(ch.isdigit() for ch in tok):
			out.add(tok)
	return out


@dataclass
class CollapsePreparedHit:
	raw_hit: Dict[str, Any]
	sort_order: int
	rerank_score: float
	dense_rank: int
	article_id: str
	point_id: str
	title_norm: str
	body_norm: str
	body_1200: str
	title_token_set: Set[str]
	body_token_set: Set[str]
	url_norm: str
	fingerprint: str
	lang: str


def prepare_collapse_hit(hit: Dict[str, Any], sort_order: int) -> CollapsePreparedHit:
	payload = hit.get("payload") or {}
	full = hit.get("full_article") or {}

	article_id = str(hit.get("id") or payload.get("article_id") or full.get("article_id") or "").strip()
	point_id = article_id
	title = _pick_field(hit, "title")
	summary = _pick_field(hit, "summary", "description", "excerpt")
	body = payload_text(payload) or payload_text(full)

	title_norm = normalize_text(title)
	body_source = "\n\n".join(part for part in [summary, body] if part).strip()
	body_norm = normalize_text(body_source)[:4000]
	title_tokens = tokenize(title)
	body_tokens = tokenize(body_source)[:260]

	return CollapsePreparedHit(
		raw_hit=dict(hit),
		sort_order=int(sort_order),
		rerank_score=float(hit.get("rerank_score") or 0.0),
		dense_rank=int(hit.get("dense_rank") or hit.get("rank") or (sort_order + 1)),
		article_id=article_id,
		point_id=point_id,
		title_norm=title_norm,
		body_norm=body_norm,
		body_1200=body_norm[:1200],
		title_token_set=set(title_tokens),
		body_token_set=set(body_tokens),
		url_norm=normalize_url(str(payload.get("url") or full.get("url") or "")),
		fingerprint=str(payload.get("fingerprint") or full.get("fingerprint") or "").strip(),
		lang=str(payload.get("lang") or full.get("lang") or "").strip().casefold(),
	)


def _exact_same_story_reason(a: CollapsePreparedHit, b: CollapsePreparedHit) -> Optional[str]:
	if a.article_id and b.article_id and a.article_id == b.article_id:
		return "same_article_id"
	if a.url_norm and b.url_norm and a.url_norm == b.url_norm:
		return "same_url"
	if a.fingerprint and b.fingerprint and a.fingerprint == b.fingerprint:
		return "same_fingerprint"
	return None


def _lexical_fallback_same_story_reason(a: CollapsePreparedHit, b: CollapsePreparedHit) -> Optional[str]:
	title_set_ratio = token_set_ratio(a.title_norm, b.title_norm, score_cutoff=88)
	body_ratio = ratio(a.body_1200, b.body_1200, score_cutoff=55)
	title_j = jaccard_set(a.title_token_set, b.title_token_set)
	body_j = jaccard_set(a.body_token_set, b.body_token_set)
	important_title_overlap = _important_overlap(a.title_token_set, b.title_token_set)

	if title_set_ratio >= 0.96:
		return "fallback_title_token_set_ratio>=0.96"
	if title_set_ratio >= 0.90 and body_j >= 0.18:
		return "fallback_title_token_set_ratio>=0.90_body_j>=0.18"
	if body_ratio >= 0.95 and title_j >= 0.12:
		return "fallback_body_ratio>=0.95"
	if len(important_title_overlap) >= 3 and body_j >= 0.12:
		return "fallback_anchor_overlap"
	return None


class UnionFind:
	def __init__(self, n: int):
		self.parent = list(range(n))
		self.rank = [0] * n

	def find(self, x: int) -> int:
		while self.parent[x] != x:
			self.parent[x] = self.parent[self.parent[x]]
			x = self.parent[x]
		return x

	def union(self, a: int, b: int) -> bool:
		ra = self.find(a)
		rb = self.find(b)
		if ra == rb:
			return False
		if self.rank[ra] < self.rank[rb]:
			self.parent[ra] = rb
		elif self.rank[ra] > self.rank[rb]:
			self.parent[rb] = ra
		else:
			self.parent[rb] = ra
			self.rank[ra] += 1
		return True


class QdrantVectorProvider:
	def __init__(self, url: str, collection: str):
		self.url = str(url)
		self.collection = str(collection)
		self.client = QdrantClient(url=self.url)

	@staticmethod
	def _as_vector(raw: Any) -> Optional[np.ndarray]:
		if raw is None:
			return None
		if isinstance(raw, dict):
			if not raw:
				return None
			raw = next(iter(raw.values()), None)
		if raw is None:
			return None
		try:
			vec = np.asarray(raw, dtype=np.float32).reshape(-1)
			if vec.size == 0:
				return None
			norm = float(np.linalg.norm(vec))
			if norm <= 1e-12:
				return None
			vec = vec / norm
			return vec
		except Exception:
			return None

	def fetch_vectors(self, point_ids: List[str], batch_size: int = 128) -> Dict[str, np.ndarray]:
		ids = [str(x).strip() for x in point_ids if str(x).strip()]
		if not ids:
			return {}

		out: Dict[str, np.ndarray] = {}
		for i in range(0, len(ids), max(1, int(batch_size))):
			batch = ids[i : i + max(1, int(batch_size))]
			try:
				records = self.client.retrieve(
					collection_name=self.collection,
					ids=batch,
					with_payload=False,
					with_vectors=True,
				)
			except Exception:
				continue

			for record in records or []:
				rid = str(getattr(record, "id", "") or "")
				if not rid:
					continue
				vec = self._as_vector(getattr(record, "vector", None))
				if vec is None:
					continue
				out[rid] = vec
		return out

	def fetch_vector(self, point_id: str) -> Optional[np.ndarray]:
		point_id = str(point_id or "").strip()
		if not point_id:
			return None
		vectors = self.fetch_vectors([point_id], batch_size=1)
		return vectors.get(point_id)


def compute_cosine_for_article_ids(
	vector_provider: QdrantVectorProvider,
	id1: str,
	id2: str,
) -> Tuple[float, bool, bool]:
	left_id = str(id1 or "").strip()
	right_id = str(id2 or "").strip()
	if not left_id or not right_id:
		raise ValueError("Both id1 and id2 must be provided")

	vectors = vector_provider.fetch_vectors([left_id, right_id], batch_size=2)
	v1 = vectors.get(left_id)
	v2 = vectors.get(right_id)
	if v1 is None or v2 is None:
		return 0.0, (v1 is not None), (v2 is not None)

	cos = float(np.dot(v1, v2))
	return cos, True, True


@dataclass
class ClusterStats:
	pool_size: int
	qdrant_found: int
	qdrant_missing: int
	pair_total: int
	pair_checks_vector: int
	pair_checks_fallback: int
	edges_total: int
	edges_embedding: int
	clusters: int
	representatives: int
	reasons: Dict[str, int]


def _cluster_pool_semantic(
	pool_hits: List[Dict[str, Any]],
	vector_provider: QdrantVectorProvider,
	embedding_sim_threshold: float,
) -> Tuple[List[CollapsePreparedHit], ClusterStats]:
	pool_docs = [prepare_collapse_hit(h, sort_order=i) for i, h in enumerate(pool_hits)]
	n = len(pool_docs)
	if n == 0:
		return [], ClusterStats(
			pool_size=0,
			qdrant_found=0,
			qdrant_missing=0,
			pair_total=0,
			pair_checks_vector=0,
			pair_checks_fallback=0,
			edges_total=0,
			edges_embedding=0,
			clusters=0,
			representatives=0,
			reasons={},
		)

	point_ids = [d.point_id for d in pool_docs if d.point_id]
	vectors = vector_provider.fetch_vectors(point_ids)
	qdrant_found = sum(1 for d in pool_docs if d.point_id and d.point_id in vectors)
	qdrant_missing = n - qdrant_found

	sim_threshold = max(-1.0, min(1.0, float(embedding_sim_threshold)))
	uf = UnionFind(n)
	reason_counts: Counter[str] = Counter()
	edges_total = 0
	edges_embedding = 0
	pair_checks_vector = 0
	pair_checks_fallback = 0

	for i in range(n):
		a = pool_docs[i]
		for j in range(i + 1, n):
			b = pool_docs[j]

			reason_exact = _exact_same_story_reason(a, b)
			if reason_exact:
				if uf.union(i, j):
					edges_total += 1
					reason_counts[reason_exact] += 1
				continue

			va = vectors.get(a.point_id)
			vb = vectors.get(b.point_id)
			if va is not None and vb is not None:
				pair_checks_vector += 1
				sim = float(np.dot(va, vb))
				if sim >= sim_threshold:
					if uf.union(i, j):
						edges_total += 1
						edges_embedding += 1
						reason_counts[f"embedding_sim>={sim_threshold:.2f}"] += 1
				continue

			# Fallback lexical léger UNIQUEMENT si vecteurs absents.
			pair_checks_fallback += 1
			reason_fallback = _lexical_fallback_same_story_reason(a, b)
			if reason_fallback and uf.union(i, j):
				edges_total += 1
				reason_counts[reason_fallback] += 1

	clusters: Dict[int, List[CollapsePreparedHit]] = {}
	for idx, doc in enumerate(pool_docs):
		root = uf.find(idx)
		clusters.setdefault(root, []).append(doc)

	representatives: List[CollapsePreparedHit] = []
	for members in clusters.values():
		best = sorted(
			members,
			key=lambda d: (-d.rerank_score, d.dense_rank, d.sort_order),
		)[0]
		representatives.append(best)

	representatives.sort(key=lambda d: (-d.rerank_score, d.dense_rank, d.sort_order))

	stats = ClusterStats(
		pool_size=n,
		qdrant_found=qdrant_found,
		qdrant_missing=qdrant_missing,
		pair_total=(n * (n - 1)) // 2,
		pair_checks_vector=pair_checks_vector,
		pair_checks_fallback=pair_checks_fallback,
		edges_total=edges_total,
		edges_embedding=edges_embedding,
		clusters=len(clusters),
		representatives=len(representatives),
		reasons=dict(reason_counts),
	)
	return representatives, stats


def collapse_reranked_hits(
	hits: List[Dict[str, Any]],
	interest: str,
	final_topn: int,
	diversity_scan_k: int,
	vector_provider: QdrantVectorProvider,
	embedding_sim_threshold: float,
	pool_chunk_size: int,
) -> List[Dict[str, Any]]:
	if not hits:
		return []

	target = max(1, int(final_topn))
	scan_k = max(target, int(diversity_scan_k))
	chunk_size = max(1, int(pool_chunk_size))
	sorted_hits = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

	pool_end = min(len(sorted_hits), scan_k)
	representatives, stats = _cluster_pool_semantic(
		sorted_hits[:pool_end],
		vector_provider=vector_provider,
		embedding_sim_threshold=embedding_sim_threshold,
	)

	while len(representatives) < target and pool_end < len(sorted_hits):
		pool_end = min(len(sorted_hits), pool_end + chunk_size)
		representatives, stats = _cluster_pool_semantic(
			sorted_hits[:pool_end],
			vector_provider=vector_provider,
			embedding_sim_threshold=embedding_sim_threshold,
		)

	final_docs = representatives[:target]
	final_hits: List[Dict[str, Any]] = []
	for rank, d in enumerate(final_docs, 1):
		h = dict(d.raw_hit)
		h["rank"] = rank
		final_hits.append(h)

	print(
		f"[rerank:{interest}] post-rerank semantic clustering fini | "
		f"input={len(hits)} pool={stats.pool_size} qdrant_found={stats.qdrant_found} "
		f"qdrant_missing={stats.qdrant_missing} pair_checks={stats.pair_checks_vector} "
		f"fallback_checks={stats.pair_checks_fallback} edges={stats.edges_total} "
		f"embedding_edges={stats.edges_embedding} clusters={stats.clusters} "
		f"representatives={stats.representatives} final={len(final_hits)}"
	)
	if stats.reasons:
		details = ", ".join(
			f"{k}:{v}" for k, v in sorted(stats.reasons.items(), key=lambda x: (-x[1], x[0]))
		)
		print(f"[rerank:{interest}] post-rerank semantic clustering reasons | {details}")

	return final_hits


def main() -> int:
	_require_dependencies()

	parser = argparse.ArgumentParser(description="Rerank retrieval candidates from PostgreSQL with semantic post-rerank clustering")
	parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
	parser.add_argument("--table", choices=["retrieval_hits", "dedup_hits"], default="retrieval_hits")
	parser.add_argument("--retrieval-run-id", "--run-id", dest="run_id", type=int, required=False)
	parser.add_argument("--model", default=core.DEFAULT_MODEL)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--topn", type=int, default=20, help="Final top-N kept after rerank + semantic clustering")
	parser.add_argument("--diversity-scan-k", type=int, default=80, help="Initial top reranked pool size used for semantic story clustering")
	parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
	parser.add_argument("--qdrant-collection", default="news_dense")
	parser.add_argument("--embedding-sim-threshold", type=float, default=0.86)
	parser.add_argument("--pool-chunk-size", type=int, default=80)
	parser.add_argument("--id1", default=None, help="Article ID #1 for cosine similarity check using existing Qdrant vectors")
	parser.add_argument("--id2", default=None, help="Article ID #2 for cosine similarity check using existing Qdrant vectors")
	parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
	parser.add_argument("--hydrate", action="store_true", help="Attach full article from PostgreSQL articles table")
	args = parser.parse_args()

	if args.id1 or args.id2:
		if not args.id1 or not args.id2:
			raise RuntimeError("When using similarity tool, provide both --id1 and --id2")
		vector_provider = QdrantVectorProvider(
			url=str(args.qdrant_url),
			collection=str(args.qdrant_collection),
		)
		cos, found1, found2 = compute_cosine_for_article_ids(
			vector_provider=vector_provider,
			id1=str(args.id1),
			id2=str(args.id2),
		)
		if not found1 or not found2:
			print(
				"[rerank:cosine] lookup | "
				f"id1_found={found1} id2_found={found2} collection={args.qdrant_collection}"
			)
			if not found1:
				print(f"[rerank:cosine] missing point for id1={args.id1}")
			if not found2:
				print(f"[rerank:cosine] missing point for id2={args.id2}")
			return 2

		print(
			"[rerank:cosine] similarity | "
			f"id1={args.id1} id2={args.id2} cosine={cos:.6f} "
			f"qdrant_collection={args.qdrant_collection}"
		)
		return 0

	if args.run_id is None:
		raise RuntimeError("--run-id is required unless using --id1/--id2 cosine mode")

	store = PostgresStore(args.db_url)
	store.init_db()

	if args.table == "retrieval_hits":
		blocks = store.fetch_retrieval_blocks(args.run_id)
		representative_retrieval_run_id = int(args.run_id)
		dedup_run_id = None
	else:
		blocks = store.fetch_dedup_blocks(args.run_id)
		dedup_meta = store.fetch_dedup_run(args.run_id)
		if dedup_meta is None:
			raise RuntimeError(f"No dedup run found for dedup_run_id={args.run_id}")
		representative_retrieval_run_id = int(dedup_meta.get("representative_retrieval_run_id") or 0)
		if representative_retrieval_run_id <= 0:
			raise RuntimeError(f"Dedup run {args.run_id} has no representative retrieval run id")
		dedup_run_id = int(args.run_id)

	if not blocks:
		raise RuntimeError(f"No hits found for table={args.table} run_id={args.run_id}")

	vector_provider = QdrantVectorProvider(
		url=str(args.qdrant_url),
		collection=str(args.qdrant_collection),
	)

	reranker = core.QwenReranker(
		model_name=args.model,
		max_length=int(args.max_length),
		instruction=args.instruction,
	)

	out_blocks: List[Dict[str, Any]] = []
	for block in blocks:
		interest = str(block.get("interest") or "").strip()
		hits = block.get("hits") or []
		if not interest or not hits:
			continue

		pairs: List[str] = []
		for h in hits:
			doc = core._extract_doc_text(h)
			pairs.append(core.format_instruction(args.instruction, interest, doc))

		scores = reranker.score_pairs(pairs, batch_size=int(args.batch_size))
		augmented: List[Dict[str, Any]] = []
		for idx, (h, s) in enumerate(zip(hits, scores), 1):
			h2 = dict(h)
			h2["dense_rank"] = int(h2.get("rank") or idx)
			h2["rerank_score"] = float(s)
			if args.hydrate:
				payload = h2.get("payload") or {}
				art = store.find_article(
					article_id=h2.get("id") or payload.get("article_id"),
					url=payload.get("url"),
				)
				if art is not None:
					h2["full_article"] = art
			augmented.append(h2)

		augmented.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
		final_hits = collapse_reranked_hits(
			augmented,
			interest=interest,
			final_topn=int(args.topn),
			diversity_scan_k=max(int(args.topn), int(args.diversity_scan_k)),
			vector_provider=vector_provider,
			embedding_sim_threshold=float(args.embedding_sim_threshold),
			pool_chunk_size=int(args.pool_chunk_size),
		)

		out_blocks.append({"interest": interest, "n": len(final_hits), "hits": final_hits})

	rerank_run_id = store.create_rerank_run(
		{
			"retrieval_run_id": representative_retrieval_run_id,
			"model": args.model,
			"max_length": int(args.max_length),
			"batch_size": int(args.batch_size),
			"topn": int(args.topn),
			"instruction": args.instruction,
			"source_table": args.table,
			"source_run_id": int(args.run_id),
			"dedup_run_id": dedup_run_id,
			"postprocess": {
				"type": "semantic_story_cluster_representative_selection",
				"vector_source": "existing_qdrant_bge_m3",
				"qdrant_url": str(args.qdrant_url),
				"qdrant_collection": str(args.qdrant_collection),
				"embedding_similarity_threshold": float(args.embedding_sim_threshold),
				"diversity_scan_k": max(int(args.topn), int(args.diversity_scan_k)),
				"pool_chunk_size": int(args.pool_chunk_size),
				"cluster_method": "union_find",
				"representative": "best_rerank_score_then_dense_rank_then_order",
			},
		}
	)
	n_rows = store.insert_rerank_hits(rerank_run_id, out_blocks)
	print(f"Saved rerank run in PostgreSQL: rerank_run_id={rerank_run_id} | hits={n_rows}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
