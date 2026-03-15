from __future__ import annotations

import argparse
import importlib.util
import os
import re
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
    "Given two news articles, answer yes if they should be merged into the same "
    "news-feed cluster. Treat different wording, outlet framing, translations, "
    "roundups, and minor follow-up updates as the same cluster when the main event, "
    "product update, rumor cycle, launch window, or announcement is essentially the same. "
    "Answer no only if they deserve separate feed slots because the main focus/event differs."
)
PAIRWISE_BODY_CHARS = 600

_NAV_PATTERNS = [
	r"\b(home|menu|navigation|breadcrumb|section|category|categories|tag|tags)\b",
	r"\b(sign in|login|log in|subscribe|newsletter|cookie|privacy|terms|contact|about us)\b",
	r"\b(share|follow us|read more|related|recommended|trending|latest news|breaking)\b",
	r"\b(advertisement|sponsored|©|all rights reserved)\b",
	r"^\s*(world|politics|business|technology|sports|opinion|video|podcast)\s*$",
]

_NOISE_PHRASES = [
	r"stay up to date with notifications",
	r"notifications can be managed",
	r"jump to content",
	r"sign up to our newsletters",
	r"log in register",
	r"all rights reserved",
]


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


def _word_count(text: str) -> int:
	return len(re.findall(r"\b\w+\b", text or "", flags=re.UNICODE))


def _tokens(text: str) -> List[str]:
	return [t.casefold() for t in re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)]


def _title_token_set(title: str) -> set[str]:
	toks = [t for t in _tokens(title) if len(t) >= 3]
	return set(toks)


def _mostly_symbols(text: str) -> bool:
	raw = (text or "").strip()
	if not raw:
		return True
	alnum = sum(1 for ch in raw if ch.isalnum())
	non_space = sum(1 for ch in raw if not ch.isspace())
	if non_space == 0:
		return True
	return (alnum / float(non_space)) < 0.45


def _looks_navigation_like(text: str) -> bool:
	line = (text or "").strip().casefold()
	if not line:
		return True
	if line.count("|") >= 2 or line.count("/") >= 3 or line.count(">") >= 2:
		return True
	for pat in _NAV_PATTERNS:
		if re.search(pat, line, flags=re.IGNORECASE):
			return True
	return False


def _nav_word_ratio(text: str) -> float:
	toks = _tokens(text)
	if not toks:
		return 1.0
	nav_hits = 0
	for tok in toks:
		if tok in {"home", "menu", "navigation", "breadcrumb", "category", "categories", "tags", "tag", "login", "subscribe", "privacy", "cookie", "contact", "about", "share", "related", "recommended", "trending"}:
			nav_hits += 1
	return nav_hits / float(len(toks))


def _repair_broken_words(text: str) -> str:
	t = str(text or "")
	t = t.replace("\xa0", " ")
	t = t.replace("\r\n", "\n").replace("\r", "\n")
	t = t.replace("\\\n", "")
	t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t, flags=re.UNICODE)
	t = re.sub(r"([a-zà-ÿ])\s*\n\s*([a-zà-ÿ])", r"\1\2", t, flags=re.IGNORECASE)
	t = re.sub(r"([a-zà-ÿ])([A-Z])", r"\1 \2", t)
	for pat in _NOISE_PHRASES:
		t = re.sub(pat, " ", t, flags=re.IGNORECASE)
	t = t.replace("\n", " ")
	t = re.sub(r"\s+", " ", t)
	return t


def _segment_text(text: str) -> List[str]:
	t = str(text or "")
	if not t:
		return []
	parts = re.split(
		r"(?<=[.!?;:])\s+|\s+[|/>]{1,}\s+|\s+-\s+|(?<=[a-zà-ÿ0-9])(?=[A-Z])|\s*·\s*",
		t,
	)
	segments: List[str] = []
	for p in parts:
		s = re.sub(r"\s+", " ", p).strip(" -|/>")
		if s:
			segments.append(s)
	return segments


def _body_candidates(hit: Dict[str, Any]) -> List[str]:
	payload = hit.get("payload") or {}
	full = hit.get("full_article") or {}
	keys = ("canonical_text", "text", "content", "body")
	out: List[str] = []
	seen: set[str] = set()
	for src in (full, payload):
		if not isinstance(src, dict):
			continue
		for key in keys:
			v = str(src.get(key) or "").strip()
			if not v:
				continue
			n = re.sub(r"\s+", " ", v).strip().casefold()
			if n in seen:
				continue
			seen.add(n)
			out.append(v)
	return out


def _cleaned_body_quality(text: str, title: str) -> float:
	if not text:
		return -1e9
	wc = _word_count(text)
	if wc <= 0:
		return -1e9
	nav_pen = _nav_word_ratio(text) * 50.0
	overlap = len(_title_token_set(title) & set(_tokens(text)))
	return float(wc) + (2.0 * float(overlap)) - nav_pen


def _segment_bonus_score(seg: str, title_tokens: set[str]) -> float:
	s = seg.strip()
	if not s:
		return -10.0
	wc = _word_count(s)
	if wc < 6:
		return -5.0
	if not any(ch.islower() for ch in s):
		return -3.0
	if _looks_navigation_like(s):
		return -4.0
	if _mostly_symbols(s):
		return -4.0
	if _nav_word_ratio(s) >= 0.35:
		return -3.0
	if s.count(">") >= 2 or s.count("/") >= 3:
		return -3.0

	score = 0.0
	seg_tokens = set(_tokens(s))
	overlap = len(seg_tokens & title_tokens)
	if overlap >= 1:
		score += min(2.0, 0.7 * overlap)
	if re.search(r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", s):
		score += 0.7
	if re.search(r"\b(announced|launch|launched|acquire|acquired|merge|merged|raises|raised|unveiled|introduced|approved|signed|released)\b", s, flags=re.IGNORECASE):
		score += 0.9
	if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", s):
		score += 0.4
	if wc >= 10:
		score += 0.4
	return score


def _pairwise_denoise_body(title: str, summary: str, body: str, max_chars: int) -> str:
	raw = " ".join([summary or "", body or ""]).strip()
	raw = _repair_broken_words(raw)
	segments = _segment_text(raw)
	if not segments:
		return ""

	title_norm = re.sub(r"\s+", " ", (title or "")).strip().casefold()
	tokens_title = _title_token_set(title)
	seen: set[str] = set()
	scored: List[tuple[int, str, float]] = []

	for idx, seg in enumerate(segments):
		s = re.sub(r"\s+", " ", seg).strip()
		if not s:
			continue
		n = s.casefold()
		if n in seen:
			continue
		seen.add(n)
		if n == title_norm or (title_norm and title_norm in n and _word_count(s) <= 8):
			continue
		score = _segment_bonus_score(s, tokens_title)
		if score < 0:
			continue
		scored.append((idx, s, score))

	if not scored:
		return ""

	# keep top segments by score, then restore original order
	scored_sorted = sorted(scored, key=lambda x: (-x[2], x[0]))
	top_k = min(5, max(2, len(scored_sorted)))
	chosen = scored_sorted[:top_k]
	chosen = sorted(chosen, key=lambda x: x[0])

	cleaned = " ".join(seg for _, seg, _ in chosen).strip()
	cleaned = re.sub(r"\s+", " ", cleaned).strip()
	if len(cleaned) > 700:
		cleaned = cleaned[:700].rsplit(" ", 1)[0].strip()
	if len(cleaned) < 400 and len(scored_sorted) > top_k:
		for _, seg, _ in sorted(scored_sorted[top_k:], key=lambda x: x[0]):
			if len(cleaned) >= 400:
				break
			cand = (cleaned + " " + seg).strip()
			if len(cand) > 700:
				break
			cleaned = cand

	if _word_count(cleaned) < 12:
		return ""
	return cleaned


def _extract_pairwise_doc_text(hit: Dict[str, Any], pairwise_denoise: bool = True) -> str:
	title = _pick_text(hit, "title").strip()
	summary = _pick_text(hit, "summary", "description", "excerpt").strip()
	body = _pick_text(hit, "canonical_text", "text", "content", "body").strip()
	if pairwise_denoise:
		best_lead = ""
		best_quality = -1e9
		for cand_body in _body_candidates(hit):
			cand_lead = _pairwise_denoise_body(title=title, summary=summary, body=cand_body, max_chars=PAIRWISE_BODY_CHARS)
			q = _cleaned_body_quality(cand_lead, title=title)
			if q > best_quality:
				best_quality = q
				best_lead = cand_lead
		lead = best_lead
		if not lead:
			lead = _pairwise_denoise_body(title=title, summary=summary, body=body, max_chars=PAIRWISE_BODY_CHARS)
	else:
		lead = body[:PAIRWISE_BODY_CHARS].strip()

	parts: List[str] = []
	if title:
		parts.append(f"TITLE: {title}")
	if summary and not pairwise_denoise:
		parts.append(f"SUMMARY: {summary}")
	if lead:
		parts.append(f"BODY: {lead}")
	if pairwise_denoise and len(lead) < 120 and title:
		# fallback title-only when body is too poor after denoise
		return f"TITLE: {title}"
	if not parts:
		url = _pick_text(hit, "url").strip()
		if url:
			parts.append(f"URL: {url}")
	return "\n\n".join(parts)


def _build_pairwise_prompt(doc_a: str, doc_b: str) -> str:
	query = (
        "Reference news article:\n"
        f"{doc_a}\n\n"
        "Task: decide whether a candidate article should be merged into the same news-feed cluster. "
        "Answer yes if the two articles are about essentially the same main event, product update, "
        "rumor cycle, launch window, or announcement, even if wording, language, outlet framing, "
        "or minor details differ."
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


def _prepare_pairwise_docs(sorted_hits: List[Dict[str, Any]], pairwise_denoise: bool = True) -> List[PairwiseDoc]:
	docs: List[PairwiseDoc] = []
	for i, hit in enumerate(sorted_hits):
		docs.append(
			PairwiseDoc(
				raw_hit=dict(hit),
				sort_order=i,
				rerank_score=float(hit.get("rerank_score") or 0.0),
				dense_rank=int(hit.get("dense_rank") or hit.get("rank") or (i + 1)),
				doc_text=_extract_pairwise_doc_text(hit, pairwise_denoise=pairwise_denoise),
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


def _article_id_from_hit(hit: Dict[str, Any]) -> str:
	payload = hit.get("payload") or {}
	return str(hit.get("id") or payload.get("article_id") or "").strip()


def _load_manual_hits_from_source(
	store: PostgresStore,
	table: str,
	run_id: int,
	id1: str,
	id2: str,
	manual_interest: Optional[str],
	hydrate: bool,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any], str]]:
	if table == "retrieval_hits":
		blocks = store.fetch_retrieval_blocks(run_id)
	else:
		blocks = store.fetch_dedup_blocks(run_id)

	interest_filter = str(manual_interest or "").strip().casefold()
	matches: Dict[str, List[tuple[str, Dict[str, Any]]]] = {str(id1): [], str(id2): []}
	for block in blocks:
		interest = str(block.get("interest") or "").strip()
		if interest_filter and interest.casefold() != interest_filter:
			continue
		for h in block.get("hits") or []:
			hit_id = _article_id_from_hit(h)
			if hit_id == str(id1):
				matches[str(id1)].append((interest, dict(h)))
			elif hit_id == str(id2):
				matches[str(id2)].append((interest, dict(h)))

	left = matches[str(id1)]
	right = matches[str(id2)]
	if not left or not right:
		return None

	left_by_interest = {interest: hit for interest, hit in left}
	right_by_interest = {interest: hit for interest, hit in right}
	common_interests = sorted(set(left_by_interest.keys()) & set(right_by_interest.keys()))
	if not common_interests:
		return None

	if interest_filter:
		picked_interest = next((i for i in common_interests if i.casefold() == interest_filter), None)
		if picked_interest is None:
			return None
	else:
		if len(common_interests) != 1:
			raise RuntimeError(
				"Manual source mode is ambiguous: the two IDs co-occur in multiple interests. "
				f"Please pass --manual-interest. interests={common_interests}"
			)
		picked_interest = common_interests[0]

	h1 = left_by_interest[picked_interest]
	h2 = right_by_interest[picked_interest]
	if hydrate:
		for h in (h1, h2):
			payload = h.get("payload") or {}
			art = store.find_article(article_id=h.get("id") or payload.get("article_id"), url=payload.get("url"))
			if art is not None:
				h["full_article"] = art

	return h1, h2, picked_interest
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
	source_table: Optional[str],
	source_run_id: Optional[int],
	manual_interest: Optional[str],
	hydrate: bool,
	threshold: float,
	smaxtreshold: Optional[float],
	smintreshold: Optional[float],
	pairwise_batch_size: int,
	pairwise_denoise: bool,
) -> int:
	left_id = str(id1 or "").strip()
	right_id = str(id2 or "").strip()
	if not left_id or not right_id:
		raise RuntimeError("Provide both --id1 and --id2 for manual judge mode")

	h1: Optional[Dict[str, Any]] = None
	h2: Optional[Dict[str, Any]] = None
	mode_label = "articles"
	picked_interest: Optional[str] = None
	if source_table is not None and source_run_id is not None:
		loaded = _load_manual_hits_from_source(
			store=store,
			table=source_table,
			run_id=int(source_run_id),
			id1=left_id,
			id2=right_id,
			manual_interest=manual_interest,
			hydrate=hydrate,
		)
		if loaded is not None:
			h1, h2, picked_interest = loaded
			mode_label = f"{source_table}:run={int(source_run_id)}"

	if h1 is None or h2 is None:
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
		mode_label = "articles"

	doc1 = _extract_pairwise_doc_text(h1, pairwise_denoise=pairwise_denoise)
	doc2 = _extract_pairwise_doc_text(h2, pairwise_denoise=pairwise_denoise)
	prompt_ab = _build_pairwise_prompt(doc1, doc2)
	prompt_ba = _build_pairwise_prompt(doc2, doc1)
	print(
		"[pairwise:manual] source | "
		f"mode={mode_label}"
		f" manual_interest={manual_interest if manual_interest else 'None'}"
		f" picked_interest={picked_interest if picked_interest else 'None'}"
		f" hydrate={hydrate}"
	)

	print("[pairwise:manual] ----- DOC A -----")
	print(doc1)
	print("[pairwise:manual] ----- DOC B -----")
	print(doc2)
	print("[pairwise:manual] ----- PROMPT A->B -----")
	print(prompt_ab)
	print("[pairwise:manual] ----- PROMPT B->A -----")
	print(prompt_ba)

	scores = reranker.score_pairs([prompt_ab, prompt_ba], batch_size=max(1, int(pairwise_batch_size)))
	s_ab = float(scores[0]) if len(scores) >= 1 else 0.0
	s_ba = float(scores[1]) if len(scores) >= 2 else s_ab
	score = float((s_ab + s_ba) / 2.0)
	s_max = max(s_ab, s_ba)
	s_min = min(s_ab, s_ba)
	use_dual = (smaxtreshold is not None) and (smintreshold is not None)
	is_yes = score >= float(threshold)
	if (not is_yes) and use_dual:
		is_yes = (s_max >= float(smaxtreshold)) and (s_min >= float(smintreshold))
	verdict = "yes" if is_yes else "no"

	print(
		"[pairwise:manual] result | "
		f"id1={left_id} id2={right_id} score_ab={s_ab:.6f} score_ba={s_ba:.6f} avg_score={score:.6f} "
		f"s_max={s_max:.6f} s_min={s_min:.6f} threshold={float(threshold):.3f} "
		f"smaxtreshold={smaxtreshold if smaxtreshold is not None else 'None'} "
		f"smintreshold={smintreshold if smintreshold is not None else 'None'} "
		f"pairwise_denoise={pairwise_denoise} same_story={verdict}"
	)
	return 0


def _greedy_pairwise_keep(
	docs: List[PairwiseDoc],
	reranker: Any,
	pairwise_threshold: float,
	smaxtreshold: Optional[float],
	smintreshold: Optional[float],
	pairwise_batch_size: int,
	interest: str,
	target: int,
) -> tuple[List[PairwiseDoc], Dict[str, int], int, int]:
	if not docs:
		return [], {}, 0, 0

	threshold = float(pairwise_threshold)
	use_dual = (smaxtreshold is not None) and (smintreshold is not None)
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
			s_max = max(s_ab, s_ba)
			s_min = min(s_ab, s_ba)
			comparisons += 1
			if pair_avg >= threshold:
				is_same_story = True
				reason_counts[f"pairwise_avg>={threshold:.2f}"] += 1
				break
			if use_dual and (s_max >= float(smaxtreshold)) and (s_min >= float(smintreshold)):
				is_same_story = True
				reason_counts[f"pairwise_max>={float(smaxtreshold):.2f}_min>={float(smintreshold):.2f}"] += 1
				break

		if is_same_story:
			rejected += 1
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
	smaxtreshold: Optional[float],
	smintreshold: Optional[float],
	pairwise_batch_size: int,
	pairwise_denoise: bool,
) -> List[Dict[str, Any]]:
	if not hits:
		return []

	target = max(1, int(final_topn))
	scan_k = max(target, int(diversity_scan_k))
	sorted_hits = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
	pool = sorted_hits[: min(len(sorted_hits), scan_k)]

	print(f"[rerank:{interest}] pool created | input={len(hits)} pool={len(pool)}")
	docs = _prepare_pairwise_docs(pool, pairwise_denoise=pairwise_denoise)
	kept_docs, reasons, pair_checks, rejected = _greedy_pairwise_keep(
		docs=docs,
		reranker=reranker,
		pairwise_threshold=pairwise_threshold,
		smaxtreshold=smaxtreshold,
		smintreshold=smintreshold,
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
		f"kept={len(kept_docs)} final={len(final_hits)} pairwise_denoise={pairwise_denoise}"
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
	parser.add_argument("--smaxtreshold", "--smax-threshold", type=float, default=None)
	parser.add_argument("--smintreshold", "--smin-threshold", type=float, default=None)
	parser.add_argument("--pairwise-denoise", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--id1", default=None, help="Manual pairwise judge: article_id #1")
	parser.add_argument("--id2", default=None, help="Manual pairwise judge: article_id #2")
	parser.add_argument("--manual-interest", default=None, help="Manual pairwise judge: constrain lookup to one interest when using --table/--run-id")
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
			source_table=(str(args.table) if args.run_id is not None else None),
			source_run_id=(int(args.run_id) if args.run_id is not None else None),
			manual_interest=(str(args.manual_interest) if args.manual_interest else None),
			hydrate=bool(args.hydrate),
			threshold=float(args.pairwise_threshold),
			smaxtreshold=(float(args.smaxtreshold) if args.smaxtreshold is not None else None),
			smintreshold=(float(args.smintreshold) if args.smintreshold is not None else None),
			pairwise_batch_size=max(1, int(args.pairwise_batch_size)),
			pairwise_denoise=bool(args.pairwise_denoise),
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
			smaxtreshold=(float(args.smaxtreshold) if args.smaxtreshold is not None else None),
			smintreshold=(float(args.smintreshold) if args.smintreshold is not None else None),
			pairwise_batch_size=max(1, int(args.pairwise_batch_size)),
			pairwise_denoise=bool(args.pairwise_denoise),
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
				"smaxtreshold": (float(args.smaxtreshold) if args.smaxtreshold is not None else None),
				"smintreshold": (float(args.smintreshold) if args.smintreshold is not None else None),
				"pairwise_denoise": bool(args.pairwise_denoise),
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
