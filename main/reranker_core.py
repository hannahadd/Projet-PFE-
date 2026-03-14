import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_INSTRUCTION = (
	"Given a user interest and a news article, score whether the article is "
	"centrally about the interest. Prefer major new developments, official "
	"announcements, wars, military escalations, sanctions, launches, product "
	"releases, mergers, legal decisions, and clear breaking-news events. "
)

PREFIX = (
	"<|im_start|>system\n"
	"Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
	"Note that the answer can only be \"yes\" or \"no\"."
	"<|im_end|>\n"
	"<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
	inst = (instruction or DEFAULT_INSTRUCTION).strip()
	q = (query or "").strip()
	d = (doc or "").strip()
	return f"<Instruct>: {inst}\n<Query>: {q}\n<Document>: {d}"


def _token_id_for_word(tokenizer, word: str) -> int:
	ids = tokenizer.encode(word, add_special_tokens=False)
	if not ids:
		raise RuntimeError(f"Could not tokenize word={word!r}")
	return int(ids[-1])


@dataclass
class RerankHit:
	interest: str
	hit: Dict[str, Any]
	text: str


class QwenReranker:
	def __init__(
		self,
		model_name: str = DEFAULT_MODEL,
		max_length: int = 1024,
		instruction: str = DEFAULT_INSTRUCTION,
	):
		self.model_name = model_name
		self.max_length = int(max_length)
		self.instruction = instruction

		from transformers import AutoTokenizer, AutoModelForCausalLM

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		dtype = torch.float16 if self.device == "cuda" else torch.float32

		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name,
			padding_side="left",
			trust_remote_code=True,
		)
		if getattr(self.tokenizer, "pad_token", None) is None:
			self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or self.tokenizer.pad_token

		# Same anti-OOM strategy as embedding: device_map='auto' with GPU headroom + CPU offload.
		model_kwargs: Dict[str, Any] = {
			"trust_remote_code": True,
			"dtype": dtype,
		}

		if self.device == "cuda":
			try:
				import accelerate  # noqa: F401

				total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
				total_gib = max(1, int(total_bytes // (1024**3)))
				max_gpu_gib = max(1, total_gib - 2)  # keep headroom
				model_kwargs.update(
					{
						"device_map": "auto",
						"max_memory": {0: f"{max_gpu_gib}GiB", "cpu": "64GiB"},
						"low_cpu_mem_usage": True,
						"attn_implementation": "sdpa",
					}
				)
			except Exception:
				# Fallback: CPU-only to avoid OOM
				self.device = "cpu"
				model_kwargs["dtype"] = torch.float32

		self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()

		# Input device when model is sharded
		self.input_device = self._infer_input_device()

		self.token_false_id = _token_id_for_word(self.tokenizer, "no")
		self.token_true_id = _token_id_for_word(self.tokenizer, "yes")
		self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
		self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

	def _infer_input_device(self) -> str:
		device_map = getattr(self.model, "hf_device_map", None)
		if isinstance(device_map, dict) and device_map:
			for dev in device_map.values():
				if isinstance(dev, str) and dev.startswith("cuda"):
					return dev
		return "cuda" if self.device == "cuda" else "cpu"

	def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
		max_len = self.max_length
		inner_max = max(1, max_len - len(self.prefix_tokens) - len(self.suffix_tokens))

		# Tokenize without padding first, then manually add prefix/suffix, then pad.
		inputs = self.tokenizer(
			pairs,
			padding=False,
			truncation="longest_first",
			return_attention_mask=False,
			max_length=inner_max,
			add_special_tokens=False,
		)
		for i, ids in enumerate(inputs["input_ids"]):
			inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

		batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_len)
		# Ensure padding honors max_length
		# (avoids warnings and keeps memory stable when offloading)
		if isinstance(batch, dict) and "input_ids" in batch and batch["input_ids"].shape[1] != max_len:
			batch = self.tokenizer.pad(inputs, padding="max_length", return_tensors="pt", max_length=max_len)
		batch = {k: v.to(self.input_device) for k, v in batch.items()}
		return batch

	def score_pairs(self, pairs: List[str], batch_size: int = 1) -> List[float]:
		bs = max(1, int(batch_size))
		out_scores: List[float] = []
		for i in range(0, len(pairs), bs):
			chunk = pairs[i : i + bs]
			inputs = self._process_inputs(chunk)
			with torch.inference_mode():
				logits = self.model(**inputs).logits
				last = logits[:, -1, :]
				true_vec = last[:, self.token_true_id]
				false_vec = last[:, self.token_false_id]
				stacked = torch.stack([false_vec, true_vec], dim=1)
				probs_yes = torch.softmax(stacked, dim=1)[:, 1]
				scores = probs_yes.detach().float().cpu().tolist()
			out_scores.extend([float(s) for s in scores])
		return out_scores


def _extract_doc_text(hit: Dict[str, Any]) -> str:
	payload = hit.get("payload") or {}

	title = (payload.get("title") or "").strip()
	summary = (
		payload.get("summary")
		or payload.get("description")
		or payload.get("excerpt")
		or ""
	).strip()
	text = (payload.get("canonical_text") or payload.get("text") or "").strip()

	# On privilégie le titre + chapô + début du corps,
	# pas le full text brut qui dilue le signal.
	lead = text[:1800].strip()

	parts = []
	if title:
		parts.append(f"TITLE: {title}")
	if summary:
		parts.append(f"SUMMARY: {summary}")
	if lead:
		parts.append(f"BODY: {lead}")

	if not parts:
		url = (payload.get("url") or "").strip()
		if url:
			parts.append(f"URL: {url}")

	return "\n\n".join(parts)


def rerank_file(
	in_path: str,
	out_path: str,
	model_name: str,
	max_length: int,
	batch_size: int,
	topn: int,
	instruction: str,
	hydrate: bool,
	dataset_path: Optional[str],
) -> None:
	with open(in_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	results = data.get("results")
	if not isinstance(results, list):
		raise RuntimeError("Input JSON must contain a top-level 'results' list (from news_reco2.py --out)")

	reranker = QwenReranker(model_name=model_name, max_length=max_length, instruction=instruction)

	lookup: Optional["DatasetLookup"] = None
	if hydrate:
		src_path = dataset_path or data.get("data_path")
		if not src_path:
			raise RuntimeError("Hydration enabled but no dataset path provided and input JSON has no 'data_path'")
		lookup = DatasetLookup.build(str(src_path))

	new_results: List[Dict[str, Any]] = []
	for block in results:
		interest = str(block.get("interest") or "").strip()
		hits = block.get("hits") or []
		if not interest or not isinstance(hits, list) or not hits:
			new_results.append(block)
			continue

		pairs: List[str] = []		
		for h in hits:
			doc = _extract_doc_text(h)
			pairs.append(format_instruction(instruction, interest, doc))

		scores = reranker.score_pairs(pairs, batch_size=batch_size)
		# attach and sort
		augmented: List[Dict[str, Any]] = []
		for idx, (h, s) in enumerate(zip(hits, scores), 1):
			h2 = dict(h)
			h2["dense_rank"] = h2.get("rank", idx)
			h2["rerank_score"] = float(s)
			if lookup is not None:
				full = lookup.find(h2)
				if full is not None:
					h2["full_article"] = full
			augmented.append(h2)

		augmented.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
		if topn and topn > 0:
			augmented = augmented[: int(topn)]
		for i, h in enumerate(augmented, 1):
			h["rank"] = i

		new_results.append(
			{
				"interest": interest,
				"n": len(augmented),
				"hits": augmented,
			}
		)

	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	out_payload = dict(data)
	out_payload["reranker_model"] = model_name
	out_payload["reranker_max_length"] = int(max_length)
	out_payload["reranker_batch_size"] = int(batch_size)
	out_payload["reranker_instruction"] = instruction
	out_payload["results"] = new_results

	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(out_payload, f, ensure_ascii=False, indent=2)


def main() -> None:
	parser = argparse.ArgumentParser(description="Rerank candidates with Qwen3-Reranker-8B (offload-friendly)")
	parser.add_argument("--in", dest="in_path", required=True, help="Input candidates JSON from news_reco2.py --out")
	parser.add_argument("--out", dest="out_path", default=None, help="Output reranked JSON")
	parser.add_argument("--model", default=DEFAULT_MODEL)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--topn", type=int, default=20, help="Keep top-N per interest after reranking")
	parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
	parser.add_argument(
		"--hydrate",
		action="store_true",
		help="Attach full article fields from dataset (uses input JSON 'data_path' unless --dataset is provided).",
	)
	parser.add_argument(
		"--dataset",
		default=None,
		help="Dataset path (JSON/JSONL) used for hydration. Overrides input JSON 'data_path'.",
	)
	args = parser.parse_args()

	in_path = args.in_path
	out_path = args.out_path
	if not out_path:
		p = Path(in_path)
		out_path = str(p.with_suffix(".reranked.json"))

	rerank_file(
		in_path=in_path,
		out_path=out_path,
		model_name=args.model,
		max_length=args.max_length,
		batch_size=args.batch_size,
		topn=args.topn,
		instruction=args.instruction,
		hydrate=bool(args.hydrate),
		dataset_path=args.dataset,
	)
	print(f"Saved reranked results to: {out_path}")


def _iter_dataset_records(path: str) -> Iterable[Dict[str, Any]]:
	with open(path, "r", encoding="utf-8") as f:
		first = f.read(1)
		f.seek(0)
		if first == "[":
			data = json.load(f)
			for d in data:
				if isinstance(d, dict):
					yield d
			return
		for line in f:
			line = line.strip()
			if not line:
				continue
			d = json.loads(line)
			if isinstance(d, dict):
				yield d


def _norm(s: Any) -> str:
	return str(s or "").strip().lower()


class DatasetLookup:
	def __init__(self):
		self.by_url: Dict[str, Dict[str, Any]] = {}
		self.by_id: Dict[str, Dict[str, Any]] = {}
		self.by_fp: Dict[str, Dict[str, Any]] = {}
		self.by_title: Dict[str, List[Dict[str, Any]]] = {}

	@staticmethod
	def build(path: str) -> "DatasetLookup":
		lk = DatasetLookup()
		for rec in _iter_dataset_records(path):
			url = rec.get("url") or rec.get("link") or ""
			title = rec.get("title") or ""
			rec_id = rec.get("id")
			fp = rec.get("fingerprint")

			if url:
				lk.by_url[_norm(url)] = rec
			if rec_id:
				lk.by_id[str(rec_id)] = rec
			if fp:
				lk.by_fp[str(fp)] = rec
			if title:
				key = _norm(title)
				lk.by_title.setdefault(key, []).append(rec)
		return lk

	def find(self, hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		payload = hit.get("payload") or {}

		# 1) Prefer id-based lookup (stable)
		rec_id = hit.get("id") or payload.get("id") or payload.get("article_id")
		if rec_id:
			rec = self.by_id.get(str(rec_id))
			if rec is not None:
				return rec

		# 2) URL-based
		url = payload.get("url")
		if url:
			rec = self.by_url.get(_norm(url))
			if rec is not None:
				return rec

		# 3) fingerprint-based
		fp = payload.get("fingerprint")
		if fp:
			rec = self.by_fp.get(str(fp))
			if rec is not None:
				return rec

		# 4) title fallback
		title = payload.get("title")
		if title:
			candidates = self.by_title.get(_norm(title))
			if candidates:
				return candidates[0]
		return None


if __name__ == "__main__":
	main()
