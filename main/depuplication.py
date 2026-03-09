from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from db import PostgresStore


def normalize_text(text: str) -> str:
	s = (text or "").strip().lower()
	s = re.sub(r"\s+", " ", s)
	return s


def normalize_url(url: str) -> str:
	raw = (url or "").strip()
	if not raw:
		return ""
	parts = urlsplit(raw)
	cleaned = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
	return cleaned.rstrip("/")


def tokenize(text: str) -> List[str]:
	return [tok for tok in re.findall(r"[a-z0-9][a-z0-9'_-]*", normalize_text(text)) if len(tok) > 1]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
	sa = set(a)
	sb = set(b)
	if not sa or not sb:
		return 0.0
	return float(len(sa & sb) / max(1, len(sa | sb)))


def ratio(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	return float(SequenceMatcher(None, a, b).ratio())


def payload_text(payload: Dict[str, Any]) -> str:
	return str(
		payload.get("canonical_text")
		or payload.get("text")
		or payload.get("content")
		or payload.get("description")
		or ""
	)


@dataclass
class PreparedHit:
	raw_hit: Dict[str, Any]
	interest: str
	source_run_id: int
	article_id: str
	score: float
	payload: Dict[str, Any]
	title_norm: str
	body_norm: str
	title_tokens: List[str]
	body_tokens: List[str]
	url_norm: str
	fingerprint: str


Predicate = Callable[[PreparedHit, PreparedHit], Optional[str]]


def prepare_hit(hit: Dict[str, Any], interest: str, source_run_id: int) -> PreparedHit:
	payload = dict(hit.get("payload") or {})
	article_id = str(hit.get("id") or payload.get("article_id") or "")
	title = str(payload.get("title") or "")
	body = payload_text(payload)
	return PreparedHit(
		raw_hit=dict(hit),
		interest=interest,
		source_run_id=int(source_run_id),
		article_id=article_id,
		score=float(hit.get("score") or 0.0),
		payload=payload,
		title_norm=normalize_text(title),
		body_norm=normalize_text(body)[:4000],
		title_tokens=tokenize(title),
		body_tokens=tokenize(body)[:180],
		url_norm=normalize_url(str(payload.get("url") or "")),
		fingerprint=str(payload.get("fingerprint") or "").strip(),
	)


def is_simple_duplicate(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	if kept.article_id and cand.article_id and kept.article_id == cand.article_id:
		return "same_article_id"
	if kept.url_norm and cand.url_norm and kept.url_norm == cand.url_norm:
		return "same_url"
	if kept.fingerprint and cand.fingerprint and kept.fingerprint == cand.fingerprint:
		return "same_fingerprint"
	if kept.title_norm and cand.title_norm and kept.title_norm == cand.title_norm:
		body_a = kept.body_norm[:1200]
		body_b = cand.body_norm[:1200]
		if body_a and body_b and (body_a == body_b or ratio(body_a, body_b) >= 0.98):
			return "same_title_same_body"
	return None


def is_near_text_duplicate(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	title_ratio = ratio(kept.title_norm, cand.title_norm)
	body_ratio = ratio(kept.body_norm[:1600], cand.body_norm[:1600])
	title_j = jaccard(kept.title_tokens, cand.title_tokens)
	body_j = jaccard(kept.body_tokens, cand.body_tokens)

	if title_ratio >= 0.96:
		return "title_ratio>=0.96"
	if body_ratio >= 0.97:
		return "body_ratio>=0.97"
	if title_ratio >= 0.90 and body_j >= 0.62:
		return "title_ratio>=0.90_body_j>=0.62"
	if title_j >= 0.85 and body_j >= 0.55:
		return "title_j>=0.85_body_j>=0.55"
	return None


def is_same_story_duplicate(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	title_ratio = ratio(kept.title_norm, cand.title_norm)
	body_ratio = ratio(kept.body_norm[:1400], cand.body_norm[:1400])
	title_j = jaccard(kept.title_tokens, cand.title_tokens)
	body_j = jaccard(kept.body_tokens, cand.body_tokens)

	if title_ratio >= 0.88 and body_ratio >= 0.45:
		return "title_ratio>=0.88_body_ratio>=0.45"
	if title_j >= 0.70 and body_j >= 0.45 and title_ratio >= 0.72:
		return "title_j>=0.70_body_j>=0.45"
	if title_j >= 0.80 and body_ratio >= 0.40:
		return "title_j>=0.80_body_ratio>=0.40"
	return None


def run_dedup_stage(hits: List[PreparedHit], predicate: Predicate, stage_name: str, interest: str) -> List[PreparedHit]:
	kept: List[PreparedHit] = []
	removed = 0
	reason_counts: Dict[str, int] = {}

	for cand in hits:
		matched_reason: Optional[str] = None
		for prev in kept:
			matched_reason = predicate(prev, cand)
			if matched_reason:
				break
		if matched_reason:
			removed += 1
			reason_counts[matched_reason] = reason_counts.get(matched_reason, 0) + 1
			continue
		kept.append(cand)

	print(
		f"[dedup:{interest}] {stage_name} fini | in={len(hits)} out={len(kept)} removed={removed}"
	)
	if reason_counts:
		details = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])))
		print(f"[dedup:{interest}] {stage_name} reasons | {details}")
	return kept


def build_output_hits(hits: List[PreparedHit]) -> List[Dict[str, Any]]:
	out: List[Dict[str, Any]] = []
	for rank, hit in enumerate(hits, 1):
		payload = dict(hit.payload)
		payload["article_id"] = hit.article_id or payload.get("article_id")
		payload["dedup_meta"] = {
			"source_table": "retrieval_hits",
			"source_run_id": hit.source_run_id,
			"source_interest": hit.interest,
			"source_rank": int(hit.raw_hit.get("rank") or rank),
		}
		out.append(
			{
				"rank": rank,
				"id": hit.article_id or payload.get("article_id"),
				"score": float(hit.score),
				"payload": payload,
			}
		)
	return out


def deduplicate_interest_hits(raw_hits: List[Dict[str, Any]], interest: str, source_run_id: int) -> List[Dict[str, Any]]:
	prepared = [prepare_hit(hit, interest=interest, source_run_id=source_run_id) for hit in raw_hits]
	prepared.sort(key=lambda h: (int(h.raw_hit.get("rank") or 10**9), -h.score))

	print(f"[dedup:{interest}] start | source_run_id={source_run_id} hits={len(prepared)}")
	stage1 = run_dedup_stage(prepared, is_simple_duplicate, "deduplication simple", interest)
	stage2 = run_dedup_stage(stage1, is_near_text_duplicate, "deduplication texte similaire", interest)
	stage3 = run_dedup_stage(stage2, is_same_story_duplicate, "deduplication meme sujet", interest)
	print(f"[dedup:{interest}] done | final_hits={len(stage3)}")
	return build_output_hits(stage3)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Deduplicate latest retrieval hits per interest and store them in PostgreSQL")
	parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/pfe_news")
	parser.add_argument("--interest", "--interet", dest="interests", action="append", default=None)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	store = PostgresStore(args.db_url)
	store.init_db()

	interests = [str(x).strip() for x in (args.interests or []) if str(x).strip()]
	latest_by_interest = store.fetch_latest_retrieval_run_ids_by_interest(interests=interests or None)
	if not latest_by_interest:
		raise RuntimeError("No retrieval interests found to deduplicate")

	print(f"[dedup] interests_to_process={len(latest_by_interest)}")
	for interest, run_id in latest_by_interest.items():
		print(f"[dedup] selected latest retrieval run | interest={interest} retrieval_run_id={run_id}")

	representative_retrieval_run_id = max(latest_by_interest.values())
	dedup_run_id = store.create_dedup_run(
		{
			"source_table": "retrieval_hits",
			"representative_retrieval_run_id": representative_retrieval_run_id,
			"interests": list(latest_by_interest.keys()),
			"source_run_ids": latest_by_interest,
			"params": {
				"stages": [
					"simple_text_duplicate",
					"near_text_duplicate",
					"same_story_duplicate",
				]
			},
		}
	)
	print(f"[dedup] created dedup_run_id={dedup_run_id}")

	out_blocks: List[Dict[str, Any]] = []
	for interest, run_id in latest_by_interest.items():
		blocks = store.fetch_retrieval_blocks(run_id)
		block = next((b for b in blocks if str(b.get("interest") or "") == interest), None)
		if not block:
			print(f"[dedup:{interest}] skipped | no block found in retrieval_run_id={run_id}")
			continue
		dedup_hits = deduplicate_interest_hits(block.get("hits") or [], interest=interest, source_run_id=run_id)
		out_blocks.append({"interest": interest, "n": len(dedup_hits), "hits": dedup_hits})

	n_rows = store.insert_dedup_hits(dedup_run_id, out_blocks)
	print(f"[dedup] saved dedup run | dedup_run_id={dedup_run_id} hits={n_rows}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
