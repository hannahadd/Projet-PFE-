from __future__ import annotations

import argparse
import importlib.util
import importlib
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Iterable, List, Optional, Set

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

try:
    fuzz = importlib.import_module("rapidfuzz.fuzz")
    _RAPIDFUZZ_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    fuzz = None
    _RAPIDFUZZ_IMPORT_ERROR = exc


def _require_dependencies() -> None:
    if _RAPIDFUZZ_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency 'rapidfuzz'. Install with: pip install rapidfuzz"
        ) from _RAPIDFUZZ_IMPORT_ERROR


def _normalize_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    cleaned = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
    return cleaned.rstrip("/")


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9][a-z0-9'_-]*", _normalize_text(text)) if len(tok) > 1]


def _token_set(tokens: Iterable[str]) -> Set[str]:
    return set(tokens)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _ratio(a: str, b: str, score_cutoff: int = 0) -> float:
    if not a or not b:
        return 0.0
    score = fuzz.ratio(a, b, score_cutoff=score_cutoff)
    if score is None:
        return 0.0
    return float(score) / 100.0


def _extract_hit_text(hit: Dict[str, Any]) -> Dict[str, str]:
    payload = hit.get("payload") or {}
    full_article = hit.get("full_article") or {}

    title = str(payload.get("title") or full_article.get("title") or "")

    summary = str(
        payload.get("summary")
        or payload.get("description")
        or payload.get("excerpt")
        or full_article.get("summary")
        or full_article.get("description")
        or full_article.get("excerpt")
        or ""
    )

    body = str(
        payload.get("canonical_text")
        or payload.get("text")
        or payload.get("content")
        or full_article.get("canonical_text")
        or full_article.get("text")
        or full_article.get("content")
        or ""
    )

    url = str(payload.get("url") or full_article.get("url") or "")
    fingerprint = str(payload.get("fingerprint") or full_article.get("fingerprint") or "").strip()
    lang = str(payload.get("lang") or full_article.get("lang") or "").strip().lower()

    return {
        "title": title,
        "summary": summary,
        "body": body,
        "url": url,
        "fingerprint": fingerprint,
        "lang": lang,
    }


def _extract_anchor_tokens(title: str, summary: str, body: str, title_tokens: List[str], body_tokens: List[str]) -> Set[str]:
    out: Set[str] = set()

    for tok in title_tokens:
        if len(tok) >= 4 or any(ch.isdigit() for ch in tok):
            out.add(tok)

    for tok in body_tokens[:100]:
        if any(ch.isdigit() for ch in tok):
            out.add(tok)

    raw = f"{title} {summary} {body[:1200]}"
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+_.-]{1,}", raw):
        t = token.lower()
        if len(t) >= 4 or any(ch.isdigit() for ch in t):
            out.add(t)

    if len(out) > 80:
        return set(sorted(out)[:80])
    return out


@dataclass
class CollapseDoc:
    hit: Dict[str, Any]
    article_id: str
    url_norm: str
    fingerprint: str
    lang: str
    title_norm: str
    summary_norm: str
    body_norm: str
    title_tokens: List[str]
    body_tokens: List[str]
    title_token_set: Set[str]
    body_token_set: Set[str]
    anchor_token_set: Set[str]
    title_len: int
    body_len: int
    body_1200: str
    body_1600: str


def _prepare_collapse_doc(hit: Dict[str, Any]) -> CollapseDoc:
    payload = hit.get("payload") or {}
    text_fields = _extract_hit_text(hit)
    title_tokens = _tokenize(text_fields["title"])
    body_tokens = _tokenize(f"{text_fields['summary']}\n{text_fields['body']}")[:260]
    title_norm = _normalize_text(text_fields["title"])
    summary_norm = _normalize_text(text_fields["summary"])
    body_norm = _normalize_text(text_fields["body"])[:5000]
    article_id = str(hit.get("id") or payload.get("article_id") or "")

    return CollapseDoc(
        hit=hit,
        article_id=article_id,
        url_norm=_normalize_url(text_fields["url"]),
        fingerprint=text_fields["fingerprint"],
        lang=text_fields["lang"],
        title_norm=title_norm,
        summary_norm=summary_norm,
        body_norm=body_norm,
        title_tokens=title_tokens,
        body_tokens=body_tokens,
        title_token_set=_token_set(title_tokens),
        body_token_set=_token_set(body_tokens),
        anchor_token_set=_extract_anchor_tokens(
            text_fields["title"],
            text_fields["summary"],
            text_fields["body"],
            title_tokens,
            body_tokens,
        ),
        title_len=len(title_norm),
        body_len=len(body_norm),
        body_1200=body_norm[:1200],
        body_1600=body_norm[:1600],
    )


def _cheap_precheck_same_story(selected: CollapseDoc, cand: CollapseDoc) -> bool:
    if selected.title_len and cand.title_len:
        max_title = max(selected.title_len, cand.title_len)
        if abs(selected.title_len - cand.title_len) > max(24, int(0.8 * max_title)):
            if not (selected.anchor_token_set & cand.anchor_token_set):
                return False

    if selected.body_len and cand.body_len:
        max_body = max(selected.body_len, cand.body_len)
        if abs(selected.body_len - cand.body_len) > max(400, int(0.9 * max_body)):
            if not (selected.anchor_token_set & cand.anchor_token_set):
                return False

    return True


def is_post_rerank_same_story(selected: CollapseDoc, cand: CollapseDoc) -> Optional[str]:
    if selected.article_id and cand.article_id and selected.article_id == cand.article_id:
        return "same_article_id"

    if selected.url_norm and cand.url_norm and selected.url_norm == cand.url_norm:
        return "same_url"

    if selected.fingerprint and cand.fingerprint and selected.fingerprint == cand.fingerprint:
        return "same_fingerprint"

    if not _cheap_precheck_same_story(selected, cand):
        return None

    title_ratio = _ratio(selected.title_norm, cand.title_norm, score_cutoff=62)
    summary_ratio = _ratio(selected.summary_norm, cand.summary_norm, score_cutoff=70)
    body_ratio = _ratio(selected.body_1600, cand.body_1600, score_cutoff=45)
    title_j = _jaccard(selected.title_token_set, cand.title_token_set)
    body_j = _jaccard(selected.body_token_set, cand.body_token_set)
    anchor_j = _jaccard(selected.anchor_token_set, cand.anchor_token_set)

    if title_ratio >= 0.965:
        return "title_ratio>=0.965"

    if body_ratio >= 0.985 and min(selected.body_len, cand.body_len) >= 280:
        return "body_ratio>=0.985"

    if title_ratio >= 0.90 and body_j >= 0.46:
        return "title_ratio>=0.90_body_j>=0.46"

    if title_j >= 0.80 and body_j >= 0.43:
        return "title_j>=0.80_body_j>=0.43"

    if summary_ratio >= 0.95 and title_j >= 0.40:
        return "summary_ratio>=0.95_title_j>=0.40"

    if anchor_j >= 0.60 and (title_ratio >= 0.62 or body_j >= 0.35):
        return "anchor_j>=0.60"

    return None


def _post_rerank_story_collapse(
    ranked_hits: List[Dict[str, Any]],
    topn: int,
    diversity_scan_k: int,
    interest: str,
) -> List[Dict[str, Any]]:
    if not ranked_hits:
        print(f"[rerank:{interest}] post-rerank diversity fini | input=0 scan=0 final=0 collapsed=0 comparisons=0")
        return []

    target = int(topn) if int(topn) > 0 else len(ranked_hits)
    if target <= 0:
        target = len(ranked_hits)

    docs = [_prepare_collapse_doc(h) for h in ranked_hits]
    scan_head = min(len(docs), max(int(diversity_scan_k), target))

    selected_docs: List[CollapseDoc] = []
    collapsed = 0
    comparisons = 0
    reason_counts: Counter[str] = Counter()

    def consider_doc(cand: CollapseDoc) -> bool:
        nonlocal collapsed, comparisons
        for prev in selected_docs:
            comparisons += 1
            reason = is_post_rerank_same_story(prev, cand)
            if reason:
                collapsed += 1
                reason_counts[reason] += 1
                return False
        selected_docs.append(cand)
        return True

    for cand in docs[:scan_head]:
        consider_doc(cand)
        if len(selected_docs) >= target:
            break

    if len(selected_docs) < target:
        for cand in docs[scan_head:]:
            consider_doc(cand)
            if len(selected_docs) >= target:
                break

    final_hits = [d.hit for d in selected_docs[:target]]
    print(
        f"[rerank:{interest}] post-rerank diversity fini | "
        f"input={len(ranked_hits)} scan={scan_head} final={len(final_hits)} "
        f"collapsed={collapsed} comparisons={comparisons}"
    )
    if reason_counts:
        details = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])))
        print(f"[rerank:{interest}] post-rerank diversity reasons | {details}")

    return final_hits


def main() -> int:
    _require_dependencies()

    parser = argparse.ArgumentParser(description="Rerank retrieval candidates from PostgreSQL")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
    parser.add_argument("--table", choices=["retrieval_hits", "dedup_hits"], default="retrieval_hits")
    parser.add_argument("--retrieval-run-id", "--run-id", dest="run_id", type=int, required=True)
    parser.add_argument("--model", default=core.DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--diversity-scan-k", type=int, default=80)
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
        augmented = _post_rerank_story_collapse(
            ranked_hits=augmented,
            topn=int(args.topn),
            diversity_scan_k=max(1, int(args.diversity_scan_k)),
            interest=interest,
        )
        for i, h in enumerate(augmented, 1):
            h["rank"] = i

        out_blocks.append({"interest": interest, "n": len(augmented), "hits": augmented})

    rerank_run_id = store.create_rerank_run(
        {
            "retrieval_run_id": representative_retrieval_run_id,
            "model": args.model,
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "topn": int(args.topn),
            "diversity_scan_k": int(args.diversity_scan_k),
            "instruction": args.instruction,
            "source_table": args.table,
            "source_run_id": int(args.run_id),
            "dedup_run_id": dedup_run_id,
            "postprocess": {
                "type": "greedy_story_collapse",
                "diversity_scan_k": int(args.diversity_scan_k),
                "topn_final": int(args.topn),
            },
        }
    )
    n_rows = store.insert_rerank_hits(rerank_run_id, out_blocks)
    print(f"Saved rerank run in PostgreSQL: rerank_run_id={rerank_run_id} | hits={n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())