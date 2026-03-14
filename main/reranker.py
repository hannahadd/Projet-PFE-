from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlsplit, urlunsplit

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
	article_id: str
	title_norm: str
	body_norm: str
	body_1200: str
	title_token_set: Set[str]
	body_token_set: Set[str]
	url_norm: str
	fingerprint: str
	lang: str


def prepare_collapse_hit(hit: Dict[str, Any]) -> CollapsePreparedHit:
	payload = hit.get("payload") or {}
	full = hit.get("full_article") or {}

	article_id = str(hit.get("id") or payload.get("article_id") or full.get("article_id") or "").strip()
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
		article_id=article_id,
		title_norm=title_norm,
		body_norm=body_norm,
		body_1200=body_norm[:1200],
		title_token_set=set(title_tokens),
		body_token_set=set(body_tokens),
		url_norm=normalize_url(str(payload.get("url") or full.get("url") or "")),
		fingerprint=str(payload.get("fingerprint") or full.get("fingerprint") or "").strip(),
		lang=str(payload.get("lang") or full.get("lang") or "").strip().casefold(),
	)


def is_post_rerank_same_story(kept: CollapsePreparedHit, cand: CollapsePreparedHit) -> Optional[str]:
	if kept.article_id and cand.article_id and kept.article_id == cand.article_id:
		return "same_article_id"
	if kept.url_norm and cand.url_norm and kept.url_norm == cand.url_norm:
		return "same_url"
	if kept.fingerprint and cand.fingerprint and kept.fingerprint == cand.fingerprint:
		return "same_fingerprint"

	title_ratio = ratio(kept.title_norm, cand.title_norm, score_cutoff=75)
	title_set_ratio = token_set_ratio(kept.title_norm, cand.title_norm, score_cutoff=75)
	body_ratio = ratio(kept.body_1200, cand.body_1200, score_cutoff=30)
	title_j = jaccard_set(kept.title_token_set, cand.title_token_set)
	body_j = jaccard_set(kept.body_token_set, cand.body_token_set)
	important_title_overlap = _important_overlap(kept.title_token_set, cand.title_token_set)
	important_body_overlap = _important_overlap(kept.body_token_set, cand.body_token_set)

	if title_set_ratio >= 0.97:
		return "title_token_set_ratio>=0.97"
	if title_ratio >= 0.93 and body_j >= 0.12:
		return "title_ratio>=0.93_body_j>=0.12"
	if body_ratio >= 0.94 and title_j >= 0.10:
		return "body_ratio>=0.94_title_j>=0.10"
	if title_set_ratio >= 0.88 and body_j >= 0.18:
		return "title_token_set_ratio>=0.88_body_j>=0.18"
	if len(important_title_overlap) >= 3 and body_j >= 0.14:
		return "important_title_overlap>=3_body_j>=0.14"
	if len(important_title_overlap) >= 2 and len(important_body_overlap) >= 5 and body_j >= 0.10:
		return "important_title_overlap>=2_important_body_overlap>=5"

	# Cas cross-language / forte reformulation
	if kept.lang and cand.lang and kept.lang != cand.lang:
		if len(important_title_overlap) >= 2 and len(important_body_overlap) >= 4 and body_j >= 0.08:
			return "cross_lang_anchor_overlap"

	return None


def collapse_reranked_hits(
	hits: List[Dict[str, Any]],
	interest: str,
	final_topn: int,
	diversity_scan_k: int,
) -> List[Dict[str, Any]]:
	if not hits:
		return []

	target = max(1, int(final_topn))
	scan_k = max(target, int(diversity_scan_k))

	sorted_hits = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
	pool = sorted_hits[:scan_k]

	selected_raw: List[Dict[str, Any]] = []
	selected_prepared: List[CollapsePreparedHit] = []
	reason_counts: Dict[str, int] = {}
	collapsed = 0
	comparisons = 0

	def try_add(candidate_hit: Dict[str, Any]) -> bool:
		nonlocal collapsed, comparisons
		cand = prepare_collapse_hit(candidate_hit)
		for prev in selected_prepared:
			comparisons += 1
			reason = is_post_rerank_same_story(prev, cand)
			if reason:
				collapsed += 1
				reason_counts[reason] = reason_counts.get(reason, 0) + 1
				return False
		selected_prepared.append(cand)
		selected_raw.append(candidate_hit)
		return True

	for h in pool:
		try_add(h)
		if len(selected_raw) >= target:
			break

	# Si le pool initial ne suffit pas à remplir topn unique, on continue plus bas.
	if len(selected_raw) < target and len(sorted_hits) > len(pool):
		for h in sorted_hits[len(pool):]:
			try_add(h)
			if len(selected_raw) >= target:
				break

	for i, h in enumerate(selected_raw, 1):
		h["rank"] = i

	print(
		f"[rerank:{interest}] post-rerank diversity fini | "
		f"input={len(hits)} scan={min(len(sorted_hits), scan_k)} "
		f"final={len(selected_raw)} collapsed={collapsed} comparisons={comparisons}"
	)
	if reason_counts:
		details = ", ".join(
			f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))
		)
		print(f"[rerank:{interest}] post-rerank diversity reasons | {details}")

	return selected_raw


def main() -> int:
	_require_dependencies()

	parser = argparse.ArgumentParser(description="Rerank retrieval candidates from PostgreSQL with post-rerank diversity")
	parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
	parser.add_argument("--table", choices=["retrieval_hits", "dedup_hits"], default="retrieval_hits")
	parser.add_argument("--retrieval-run-id", "--run-id", dest="run_id", type=int, required=True)
	parser.add_argument("--model", default=core.DEFAULT_MODEL)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--topn", type=int, default=20, help="Final top-N kept after rerank + diversity collapse")
	parser.add_argument("--diversity-scan-k", type=int, default=80, help="Number of top reranked candidates scanned before story collapse")
	parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
	parser.add_argument("--hydrate", action="store_true", help="Attach full article from PostgreSQL articles table")
	args = parser.parse_args()

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
				"type": "greedy_story_collapse",
				"diversity_scan_k": max(int(args.topn), int(args.diversity_scan_k)),
			},
		}
	)
	n_rows = store.insert_rerank_hits(rerank_run_id, out_blocks)
	print(f"Saved rerank run in PostgreSQL: rerank_run_id={rerank_run_id} | hits={n_rows}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())