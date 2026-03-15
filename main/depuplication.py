from __future__ import annotations

import argparse
import importlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit, urlunsplit

from db import PostgresStore

try:
	fuzz = importlib.import_module("rapidfuzz.fuzz")
	_RAPIDFUZZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
	fuzz = None
	_RAPIDFUZZ_IMPORT_ERROR = exc

try:
	QdrantClient = importlib.import_module("qdrant_client").QdrantClient
	qm = importlib.import_module("qdrant_client.http.models")
	_QDRANT_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
	QdrantClient = None
	qm = None
	_QDRANT_IMPORT_ERROR = exc


def _require_dependencies(require_qdrant: bool) -> None:
	if _RAPIDFUZZ_IMPORT_ERROR is not None:
		raise RuntimeError(
			"Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz"
		) from _RAPIDFUZZ_IMPORT_ERROR
	if require_qdrant and _QDRANT_IMPORT_ERROR is not None:
		raise RuntimeError(
			"Missing dependency 'qdrant-client'. Install with: pip install qdrant-client"
		) from _QDRANT_IMPORT_ERROR


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


def _token_set(tokens: Iterable[str]) -> Set[str]:
	return set(tokens)


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


def payload_text(payload: Dict[str, Any]) -> str:
	return str(
		payload.get("canonical_text")
		or payload.get("text")
		or payload.get("content")
		or payload.get("description")
		or ""
	)


def _safe_datetime_from_iso(value: str) -> Optional[datetime]:
	raw = (value or "").strip()
	if not raw:
		return None
	try:
		return datetime.fromisoformat(raw.replace("Z", "+00:00"))
	except Exception:
		return None


def _title_char_ngrams(text: str, n: int = 3, max_items: int = 4) -> List[str]:
	s = re.sub(r"[^a-z0-9]", "", text)
	if len(s) < n:
		return [f"g:{s}"] if s else []
	seen: Set[str] = set()
	out: List[str] = []
	for i in range(0, len(s) - n + 1):
		g = s[i : i + n]
		if g in seen:
			continue
		seen.add(g)
		out.append(f"g:{g}")
		if len(out) >= max_items:
			break
	return out


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
	title_token_set: Set[str]
	body_token_set: Set[str]
	body_1200: str
	body_1400: str
	body_1600: str
	title_len: int
	body_len: int
	url_norm: str
	fingerprint: str
	source_domain: str
	lang: str
	published_at: str


Predicate = Callable[[PreparedHit, PreparedHit], Optional[str]]


def prepare_hit(hit: Dict[str, Any], interest: str, source_run_id: int) -> PreparedHit:
	payload = dict(hit.get("payload") or {})
	article_id = str(hit.get("id") or payload.get("article_id") or "")
	title = str(payload.get("title") or "")
	body = payload_text(payload)
	url_norm = normalize_url(str(payload.get("url") or ""))
	title_tokens = tokenize(title)
	body_tokens = tokenize(body)[:180]
	title_norm = normalize_text(title)
	body_norm = normalize_text(body)[:4000]
	source_domain = str(payload.get("domain") or "").strip().lower()
	if not source_domain and url_norm:
		source_domain = urlsplit(url_norm).netloc.lower()
	lang = str(payload.get("lang") or "").strip().lower()
	published_at = str(payload.get("date") or payload.get("published_date") or "").strip()

	return PreparedHit(
		raw_hit=dict(hit),
		interest=interest,
		source_run_id=int(source_run_id),
		article_id=article_id,
		score=float(hit.get("score") or 0.0),
		payload=payload,
		title_norm=title_norm,
		body_norm=body_norm,
		title_tokens=title_tokens,
		body_tokens=body_tokens,
		title_token_set=_token_set(title_tokens),
		body_token_set=_token_set(body_tokens),
		body_1200=body_norm[:1200],
		body_1400=body_norm[:1400],
		body_1600=body_norm[:1600],
		title_len=len(title_norm),
		body_len=len(body_norm),
		url_norm=url_norm,
		fingerprint=str(payload.get("fingerprint") or "").strip(),
		source_domain=source_domain,
		lang=lang,
		published_at=published_at,
	)


def is_simple_duplicate(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	if kept.article_id and cand.article_id and kept.article_id == cand.article_id:
		return "same_article_id"
	if kept.url_norm and cand.url_norm and kept.url_norm == cand.url_norm:
		return "same_url"
	if kept.fingerprint and cand.fingerprint and kept.fingerprint == cand.fingerprint:
		return "same_fingerprint"
	if kept.title_norm and cand.title_norm and kept.title_norm == cand.title_norm:
		body_a = kept.body_1200
		body_b = cand.body_1200
		if body_a and body_b and (body_a == body_b or ratio(body_a, body_b, score_cutoff=98) >= 0.98):
			return "same_title_same_body"
	return None


def is_near_text_duplicate(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	title_ratio = ratio(kept.title_norm, cand.title_norm, score_cutoff=72)
	body_ratio = ratio(kept.body_1600, cand.body_1600, score_cutoff=80)
	title_j = jaccard_set(kept.title_token_set, cand.title_token_set)
	body_j = jaccard_set(kept.body_token_set, cand.body_token_set)

	if title_ratio >= 0.96:
		return "title_ratio>=0.96"
	if body_ratio >= 0.97:
		return "body_ratio>=0.97"
	if title_ratio >= 0.90 and body_j >= 0.62:
		return "title_ratio>=0.90_body_j>=0.62"
	if title_j >= 0.85 and body_j >= 0.55:
		return "title_j>=0.85_body_j>=0.55"
	return None


def is_same_story_duplicate_light(kept: PreparedHit, cand: PreparedHit) -> Optional[str]:
	title_ratio = ratio(kept.title_norm, cand.title_norm, score_cutoff=68)
	body_ratio = ratio(kept.body_1400, cand.body_1400, score_cutoff=35)
	title_j = jaccard_set(kept.title_token_set, cand.title_token_set)
	body_j = jaccard_set(kept.body_token_set, cand.body_token_set)

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


class BlockingIndex:
	def __init__(self) -> None:
		self.postings: Dict[str, List[int]] = defaultdict(list)
		self.domain_postings: Dict[str, List[int]] = defaultdict(list)

	@staticmethod
	def _keys(hit: PreparedHit) -> List[str]:
		keys: List[str] = []
		for tok in list(dict.fromkeys(hit.title_tokens))[:6]:
			keys.append(f"t:{tok}")
			if hit.source_domain:
				keys.append(f"d:{hit.source_domain}|{tok}")
		keys.extend(_title_char_ngrams(hit.title_norm, n=3, max_items=4))
		if not keys and hit.source_domain:
			keys.append(f"donly:{hit.source_domain}")
		return keys

	def add(self, hit: PreparedHit, kept_idx: int) -> None:
		for key in self._keys(hit):
			self.postings[key].append(kept_idx)
		if hit.source_domain:
			self.domain_postings[hit.source_domain].append(kept_idx)

	def shortlist(self, cand: PreparedHit, max_candidates: int) -> List[int]:
		votes: Dict[int, int] = defaultdict(int)
		for key in self._keys(cand):
			bucket = self.postings.get(key) or []
			for idx in bucket[-256:]:
				votes[idx] += 1
		if not votes and cand.source_domain:
			for idx in self.domain_postings.get(cand.source_domain, [])[-max_candidates:]:
				votes[idx] += 1
		if not votes:
			return []
		ordered = sorted(votes.items(), key=lambda x: (-x[1], -x[0]))
		return [idx for idx, _ in ordered[:max_candidates]]


def _cheap_near_precheck(kept: PreparedHit, cand: PreparedHit) -> bool:
	if kept.title_len and cand.title_len:
		max_len = max(kept.title_len, cand.title_len)
		if abs(kept.title_len - cand.title_len) > max(20, int(0.65 * max_len)):
			return False
	if kept.body_len and cand.body_len:
		max_body = max(kept.body_len, cand.body_len)
		if abs(kept.body_len - cand.body_len) > max(240, int(0.8 * max_body)):
			return False
	if kept.title_token_set and cand.title_token_set and not (kept.title_token_set & cand.title_token_set):
		return False
	return True


def run_dedup_stage2_blocking(
	hits: List[PreparedHit],
	interest: str,
	max_candidates: int = 120,
) -> List[PreparedHit]:
	kept: List[PreparedHit] = []
	index = BlockingIndex()
	reason_counts: Dict[str, int] = {}
	removed = 0
	total_shortlist = 0
	total_checks = 0

	for cand in hits:
		shortlist = index.shortlist(cand, max_candidates=max_candidates)
		total_shortlist += len(shortlist)
		matched_reason: Optional[str] = None
		for kept_idx in shortlist:
			prev = kept[kept_idx]
			if not _cheap_near_precheck(prev, cand):
				continue
			total_checks += 1
			matched_reason = is_near_text_duplicate(prev, cand)
			if matched_reason:
				break

		if matched_reason:
			removed += 1
			reason_counts[matched_reason] = reason_counts.get(matched_reason, 0) + 1
			continue

		kept_idx = len(kept)
		kept.append(cand)
		index.add(cand, kept_idx)

	avg_shortlist = (total_shortlist / len(hits)) if hits else 0.0
	avg_checks = (total_checks / len(hits)) if hits else 0.0
	print(
		f"[dedup:{interest}] deduplication texte similaire fini | in={len(hits)} out={len(kept)} removed={removed} avg_shortlist={avg_shortlist:.2f} avg_checks={avg_checks:.2f}"
	)
	if reason_counts:
		details = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])))
		print(f"[dedup:{interest}] deduplication texte similaire reasons | {details}")
	return kept


@dataclass
class Stage3Config:
	qdrant_url: str = "http://localhost:6333"
	qdrant_collection: str = "news_dense"
	top_k: int = 40
	max_intersection_candidates: int = 32
	fallback_candidates: int = 24
	date_window_days: int = 3
	use_lang_filter: bool = True


class QdrantNeighborFinder:
	def __init__(self, cfg: Stage3Config) -> None:
		self.cfg = cfg
		self.client = QdrantClient(url=cfg.qdrant_url)
		self._cache: Dict[Tuple[str, str, str], List[str]] = {}

	def _build_filter(self, hit: PreparedHit) -> Optional[Any]:
		must: List[Any] = []
		if self.cfg.use_lang_filter and hit.lang:
			must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=hit.lang)))

		if self.cfg.date_window_days > 0 and hit.published_at:
			dt = _safe_datetime_from_iso(hit.published_at)
			if dt is not None:
				gte = (dt - timedelta(days=self.cfg.date_window_days)).isoformat()
				lte = (dt + timedelta(days=self.cfg.date_window_days)).isoformat()
				if hasattr(qm, "DatetimeRange"):
					rng = qm.DatetimeRange(gte=gte, lte=lte)
				else:
					rng = qm.Range(gte=gte, lte=lte)
				must.append(qm.FieldCondition(key="date", range=rng))

		if not must:
			return None
		return qm.Filter(must=must)

	def _recommend_neighbors(self, article_id: str, qfilter: Optional[Any]) -> List[str]:
		if not hasattr(self.client, "recommend"):
			return []
		try:
			recs = self.client.recommend(
				collection_name=self.cfg.qdrant_collection,
				positive=[article_id],
				limit=self.cfg.top_k + 1,
				query_filter=qfilter,
				with_payload=False,
				with_vectors=False,
			)
		except Exception:
			return []
		out: List[str] = []
		for rec in recs or []:
			rid = str(getattr(rec, "id", "") or "")
			if rid and rid != article_id:
				out.append(rid)
		return out

	def _vector_search_neighbors(self, article_id: str, qfilter: Optional[Any]) -> List[str]:
		try:
			recs = self.client.retrieve(
				collection_name=self.cfg.qdrant_collection,
				ids=[article_id],
				with_payload=False,
				with_vectors=True,
			)
		except Exception:
			return []
		if not recs:
			return []
		vec = getattr(recs[0], "vector", None)
		if vec is None:
			return []

		hits: List[Any] = []
		if hasattr(self.client, "search"):
			try:
				hits = self.client.search(
					collection_name=self.cfg.qdrant_collection,
					query_vector=vec,
					limit=self.cfg.top_k + 1,
					query_filter=qfilter,
					with_payload=False,
					with_vectors=False,
				)
			except Exception:
				hits = []
		elif hasattr(self.client, "search_points"):
			try:
				hits = self.client.search_points(
					collection_name=self.cfg.qdrant_collection,
					query_vector=vec,
					limit=self.cfg.top_k + 1,
					query_filter=qfilter,
					with_payload=False,
					with_vectors=False,
				)
			except Exception:
				hits = []
		else:
			try:
				resp = self.client.query_points(
					collection_name=self.cfg.qdrant_collection,
					query=vec,
					limit=self.cfg.top_k + 1,
					query_filter=qfilter,
					with_payload=False,
					with_vectors=False,
				)
				hits = list(getattr(resp, "points", []) or [])
			except Exception:
				hits = []

		out: List[str] = []
		for rec in hits:
			rid = str(getattr(rec, "id", "") or "")
			if rid and rid != article_id:
				out.append(rid)
		return out

	def neighbors(self, hit: PreparedHit) -> List[str]:
		article_id = str(hit.article_id or "").strip()
		if not article_id:
			return []

		cache_key = (article_id, hit.lang, hit.published_at[:10])
		cached = self._cache.get(cache_key)
		if cached is not None:
			return cached

		qfilter = self._build_filter(hit)
		out = self._recommend_neighbors(article_id, qfilter=qfilter)
		if not out:
			out = self._vector_search_neighbors(article_id, qfilter=qfilter)
		out = out[: self.cfg.top_k]
		self._cache[cache_key] = out
		return out


def run_dedup_stage3_qdrant(
	hits: List[PreparedHit],
	interest: str,
	finder: QdrantNeighborFinder,
	cfg: Stage3Config,
) -> List[PreparedHit]:
	kept: List[PreparedHit] = []
	kept_by_article_id: Dict[str, int] = {}
	fallback_index = BlockingIndex()
	reason_counts: Dict[str, int] = {}
	removed = 0
	total_shortlist = 0
	total_checks = 0
	qdrant_queries = 0
	qdrant_with_neighbors = 0
	fallback_used = 0

	for cand in hits:
		qdrant_queries += 1
		neighbor_ids = finder.neighbors(cand)
		if neighbor_ids:
			qdrant_with_neighbors += 1

		shortlist: List[int] = []
		seen_shortlist: Set[int] = set()
		for nid in neighbor_ids:
			idx = kept_by_article_id.get(str(nid))
			if idx is not None and idx not in seen_shortlist:
				shortlist.append(idx)
				seen_shortlist.add(idx)
			if len(shortlist) >= cfg.max_intersection_candidates:
				break

		if not shortlist:
			fallback_used += 1
			shortlist = fallback_index.shortlist(cand, max_candidates=cfg.fallback_candidates)

		total_shortlist += len(shortlist)
		matched_reason: Optional[str] = None
		for kept_idx in shortlist:
			prev = kept[kept_idx]
			total_checks += 1
			matched_reason = is_same_story_duplicate_light(prev, cand)
			if matched_reason:
				break

		if matched_reason:
			removed += 1
			reason_counts[matched_reason] = reason_counts.get(matched_reason, 0) + 1
			continue

		new_idx = len(kept)
		kept.append(cand)
		if cand.article_id:
			kept_by_article_id[cand.article_id] = new_idx
		fallback_index.add(cand, new_idx)

	avg_shortlist = (total_shortlist / len(hits)) if hits else 0.0
	avg_checks = (total_checks / len(hits)) if hits else 0.0
	print(
		f"[dedup:{interest}] deduplication meme sujet fini | in={len(hits)} out={len(kept)} removed={removed} avg_shortlist={avg_shortlist:.2f} avg_checks={avg_checks:.2f} qdrant_neighbors={qdrant_with_neighbors}/{qdrant_queries} fallback_used={fallback_used}"
	)
	if reason_counts:
		details = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])))
		print(f"[dedup:{interest}] deduplication meme sujet reasons | {details}")
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


def deduplicate_interest_hits(
	raw_hits: List[Dict[str, Any]],
	interest: str,
	source_run_id: int,
	stage2_max_candidates: int = 120,
) -> List[Dict[str, Any]]:
	prepared = [prepare_hit(hit, interest=interest, source_run_id=source_run_id) for hit in raw_hits]
	prepared.sort(key=lambda h: (int(h.raw_hit.get("rank") or 10**9), -h.score))

	print(f"[dedup:{interest}] start | source_run_id={source_run_id} hits={len(prepared)}")	
	stage1 = run_dedup_stage(prepared, is_simple_duplicate, "deduplication simple", interest)
	stage2 = run_dedup_stage2_blocking(stage1, interest=interest, max_candidates=stage2_max_candidates)

	print(f"[dedup:{interest}] done | final_hits={len(stage2)}")
	return build_output_hits(stage2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Deduplicate latest retrieval hits per interest and store them in PostgreSQL")
	parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/pfe_news")
	parser.add_argument("--interest", "--interet", dest="interests", action="append", default=None)
	parser.add_argument("--stage2-max-candidates", type=int, default=120)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	_require_dependencies(require_qdrant=False)

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
					"near_text_duplicate_blocking_rapidfuzz",
				],
				"stage2": {"max_candidates": max(1, int(args.stage2_max_candidates))},
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
		dedup_hits = deduplicate_interest_hits(
			block.get("hits") or [],
			interest=interest,
			source_run_id=run_id,
			stage2_max_candidates=max(1, int(args.stage2_max_candidates)),
		)
		out_blocks.append({"interest": interest, "n": len(dedup_hits), "hits": dedup_hits})

	n_rows = store.insert_dedup_hits(dedup_run_id, out_blocks)
	print(f"[dedup] saved dedup run | dedup_run_id={dedup_run_id} hits={n_rows}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
