from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import PostgresStore


ROOT = Path(__file__).resolve().parents[1]
SRC_RERANKER_PATH = Path(__file__).resolve().parent / "reranker_core.py"


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
DEFAULT_PAIRWISE_INSTRUCTION = (
	"Given a reference news article and a candidate news article, answer yes only if they report "
	"the same specific real-world news event or announcement. Treat different outlet framing, "
	"different wording, translations, and consequence-focused headlines as the same story when "
	"the underlying event is the same. Answer no for merely related topics, background, analysis, "
	"or follow-up commentary about a broader subject."
)
PAIRWISE_BODY_CHARS = 600


@dataclass
class PairwiseDoc:
	raw_hit: Dict[str, Any]
	sort_order: int
	rerank_score: float
	dense_rank: int
	doc_text: str


def _pick_text(hit: Dict[str, Any], *keys: str) -> str:
	payload = hit.get("payload") or {}
	full = hit.get("full_article") or {}
	for key in keys:
		value = payload.get(key)
		if value:
			return str(value)
	for key in keys:
		value = full.get(key)
		if value:
			return str(value)
	return ""


def _extract_pairwise_doc_text(hit: Dict[str, Any]) -> str:
	title = _pick_text(hit, "title").strip()
	summary = _pick_text(hit, "summary", "description", "excerpt").strip()
	body = _pick_text(hit, "canonical_text", "text", "content", "body").strip()
	lead = body[:PAIRWISE_BODY_CHARS].strip()

	parts: List[str] = []
	if title:
		parts.append(f"TITLE: {title}")
	if summary:
		parts.append(f"SUMMARY: {summary}")
	if lead:
		parts.append(f"BODY: {lead}")
	if not parts:
		url = _pick_text(hit, "url").strip()
		if url:
			parts.append(f"URL: {url}")
	return "\n\n".join(parts)


def _build_pairwise_prompt(doc_a: str, doc_b: str) -> str:
	query = (
		"Reference news article:\n"
		f"{doc_a}\n\n"
		"Task: retrieve documents about the same specific news event or announcement. "
		"Different outlet framing is allowed if the underlying event is the same."
	)
	return core.format_instruction(DEFAULT_PAIRWISE_INSTRUCTION, query, doc_b)


def _pairwise_score_bidirectional(
	reranker: Any,
	doc_a: str,
	doc_b: str,
	batch_size: int,
) -> float:
	prompt_ab = _build_pairwise_prompt(doc_a, doc_b)
	prompt_ba = _build_pairwise_prompt(doc_b, doc_a)
	scores = reranker.score_pairs([prompt_ab, prompt_ba], batch_size=max(1, int(batch_size)))
	if not scores:
		return 0.0
	if len(scores) == 1:
		return float(scores[0])
	return float((float(scores[0]) + float(scores[1])) / 2.0)


def _prepare_pairwise_docs(sorted_hits: List[Dict[str, Any]]) -> List[PairwiseDoc]:
	docs: List[PairwiseDoc] = []
	for i, hit in enumerate(sorted_hits):
		docs.append(
			PairwiseDoc(
				raw_hit=dict(hit),
				sort_order=i,
				rerank_score=float(hit.get("rerank_score") or 0.0),
				dense_rank=int(hit.get("dense_rank") or hit.get("rank") or (i + 1)),
				doc_text=_extract_pairwise_doc_text(hit),
			)
		)
	return docs


def _hit_from_article_row(article: Dict[str, Any]) -> Dict[str, Any]:
	article = dict(article or {})
	payload = {
		"article_id": article.get("id"),
		"title": article.get("title") or "",
		"canonical_text": article.get("content") or "",
		"text": article.get("content") or "",
		"url": article.get("url") or "",
		"fingerprint": article.get("fingerprint") or "",
		"lang": article.get("lang") or "",
	}
	return {
		"id": article.get("id"),
		"rank": 1,
		"payload": payload,
		"full_article": article,
	}


def run_manual_pairwise_judge(
	store: PostgresStore,
	reranker: Any,
	id1: str,
	id2: str,
	threshold: float,
	pairwise_batch_size: int,
) -> int:
	left_id = str(id1 or "").strip()
	right_id = str(id2 or "").strip()
	if not left_id or not right_id:
		raise RuntimeError("Provide both --id1 and --id2 for manual judge mode")

	a1 = store.find_article(article_id=left_id, url=None)
	a2 = store.find_article(article_id=right_id, url=None)
	if a1 is None or a2 is None:
		print(
			"[pairwise:manual] lookup | "
			f"id1_found={a1 is not None} id2_found={a2 is not None}"
		)
		if a1 is None:
			print(f"[pairwise:manual] missing article for id1={left_id}")
		if a2 is None:
			print(f"[pairwise:manual] missing article for id2={right_id}")
		return 2

	h1 = _hit_from_article_row(a1)
	h2 = _hit_from_article_row(a2)
	doc1 = _extract_pairwise_doc_text(h1)
	doc2 = _extract_pairwise_doc_text(h2)
	score = _pairwise_score_bidirectional(
		reranker=reranker,
		doc_a=doc1,
		doc_b=doc2,
		batch_size=max(1, int(pairwise_batch_size)),
	)
	verdict = "yes" if score >= float(threshold) else "no"

	print(
		"[pairwise:manual] result | "
		f"id1={left_id} id2={right_id} score={score:.6f} "
		f"threshold={float(threshold):.3f} same_story={verdict}"
	)
	return 0


def _greedy_pairwise_keep(
	docs: List[PairwiseDoc],
	reranker: Any,
	pairwise_threshold: float,
	pairwise_batch_size: int,
	interest: str,
	target: int,
) -> tuple[List[PairwiseDoc], Dict[str, int], int, int]:
	if not docs:
		return [], {}, 0, 0

	threshold = float(pairwise_threshold)
	kept: List[PairwiseDoc] = [docs[0]]
	reason_counts: Counter[str] = Counter()
	comparisons = 0
	rejected = 0

	print(f"[rerank:{interest}] pairwise judge start | mode=greedy")

	for cand in docs[1:]:
		if len(kept) >= target:
			break

		if not kept:
			kept.append(cand)
			continue

		pair_prompts: List[str] = []
		for prev in kept:
			pair_prompts.append(_build_pairwise_prompt(prev.doc_text, cand.doc_text))
			pair_prompts.append(_build_pairwise_prompt(cand.doc_text, prev.doc_text))

		scores = reranker.score_pairs(pair_prompts, batch_size=max(1, int(pairwise_batch_size)))
		if len(scores) < (2 * len(kept)):
			scores = list(scores) + [0.0] * ((2 * len(kept)) - len(scores))

		is_same_story = False
		for idx_prev in range(len(kept)):
			s_ab = float(scores[2 * idx_prev])
			s_ba = float(scores[2 * idx_prev + 1])
			pair_avg = (s_ab + s_ba) / 2.0
			comparisons += 1
			if pair_avg >= threshold:
				is_same_story = True
				break

		if is_same_story:
			rejected += 1
			reason_counts[f"pairwise_yes>={threshold:.2f}"] += 1
			continue

		kept.append(cand)

	return kept, dict(reason_counts), comparisons, rejected


def collapse_reranked_hits_pairwise(
	hits: List[Dict[str, Any]],
	interest: str,
	final_topn: int,
	diversity_scan_k: int,
	reranker: Any,
	pairwise_threshold: float,
	pairwise_batch_size: int,
) -> List[Dict[str, Any]]:
	if not hits:
		return []

	target = max(1, int(final_topn))
	scan_k = max(target, int(diversity_scan_k))
	sorted_hits = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
	pool = sorted_hits[: min(len(sorted_hits), scan_k)]

	print(f"[rerank:{interest}] pool created | input={len(hits)} pool={len(pool)}")
	docs = _prepare_pairwise_docs(pool)
	kept_docs, reasons, pair_checks, rejected = _greedy_pairwise_keep(
		docs=docs,
		reranker=reranker,
		pairwise_threshold=pairwise_threshold,
		pairwise_batch_size=pairwise_batch_size,
		interest=interest,
		target=target,
	)

	final_docs = kept_docs[:target]
	final_hits: List[Dict[str, Any]] = []
	for rank, doc in enumerate(final_docs, 1):
		h = dict(doc.raw_hit)
		h["rank"] = rank
		final_hits.append(h)

	print(
		f"[rerank:{interest}] post-rerank pairwise greedy fini | "
		f"input={len(hits)} pool={len(pool)} comparisons={pair_checks} rejected={rejected} "
		f"kept={len(kept_docs)} final={len(final_hits)}"
	)
	if reasons:
		details = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0])))
		print(f"[rerank:{interest}] post-rerank pairwise reasons | {details}")

	return final_hits


def main() -> int:
	parser = argparse.ArgumentParser(description="Rerank retrieval candidates from PostgreSQL with pairwise post-rerank clustering")
	parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
	parser.add_argument("--table", choices=["retrieval_hits", "dedup_hits"], default="retrieval_hits")
	parser.add_argument("--retrieval-run-id", "--run-id", dest="run_id", type=int, required=False)
	parser.add_argument("--model", default=core.DEFAULT_MODEL)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--pairwise-batch-size", type=int, default=2)
	parser.add_argument("--pairwise-threshold", type=float, default=0.62)
	parser.add_argument("--id1", default=None, help="Manual pairwise judge: article_id #1")
	parser.add_argument("--id2", default=None, help="Manual pairwise judge: article_id #2")
	parser.add_argument("--topn", type=int, default=20, help="Final top-N kept after rerank + pairwise clustering")
	parser.add_argument("--diversity-scan-k", type=int, default=80, help="Top reranked pool size used for pairwise clustering")
	parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
	parser.add_argument("--hydrate", action="store_true", help="Attach full article from PostgreSQL articles table")
	args = parser.parse_args()

	store = PostgresStore(args.db_url)
	store.init_db()

	reranker = core.QwenReranker(
		model_name=args.model,
		max_length=int(args.max_length),
		instruction=args.instruction,
	)

	if args.id1 or args.id2:
		return run_manual_pairwise_judge(
			store=store,
			reranker=reranker,
			id1=str(args.id1 or ""),
			id2=str(args.id2 or ""),
			threshold=float(args.pairwise_threshold),
			pairwise_batch_size=max(1, int(args.pairwise_batch_size)),
		)

	if args.run_id is None:
		raise RuntimeError("--run-id is required unless you use manual mode with --id1 and --id2")

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

		scores = reranker.score_pairs(pairs, batch_size=max(1, int(args.batch_size)))
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
		final_hits = collapse_reranked_hits_pairwise(
			hits=augmented,
			interest=interest,
			final_topn=int(args.topn),
			diversity_scan_k=max(int(args.topn), int(args.diversity_scan_k)),
			reranker=reranker,
			pairwise_threshold=float(args.pairwise_threshold),
			pairwise_batch_size=max(1, int(args.pairwise_batch_size)),
		)

		out_blocks.append({"interest": interest, "n": len(final_hits), "hits": final_hits})

	rerank_run_id = store.create_rerank_run(
		{
			"retrieval_run_id": representative_retrieval_run_id,
			"model": args.model,
			"max_length": int(args.max_length),
			"batch_size": int(args.batch_size),
			"pairwise_batch_size": int(args.pairwise_batch_size),
			"pairwise_threshold": float(args.pairwise_threshold),
			"topn": int(args.topn),
			"instruction": args.instruction,
			"source_table": args.table,
			"source_run_id": int(args.run_id),
			"dedup_run_id": dedup_run_id,
			"postprocess": {
				"type": "pairwise_qwen_greedy_keep_selection",
				"pool_size": max(int(args.topn), int(args.diversity_scan_k)),
				"selection_method": "greedy_compare_with_kept_only",
				"pairwise_model": str(args.model),
				"pairwise_threshold": float(args.pairwise_threshold),
				"pairwise_direction": "bidirectional_mean",
				"pairwise_body_chars": int(PAIRWISE_BODY_CHARS),
				"order": "rerank_desc",
			},
		}
	)
	n_rows = store.insert_rerank_hits(rerank_run_id, out_blocks)
	print(f"Saved rerank run in PostgreSQL: rerank_run_id={rerank_run_id} | hits={n_rows}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
