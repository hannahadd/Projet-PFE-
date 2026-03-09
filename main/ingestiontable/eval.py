from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from db import PostgresStore


DEFAULT_EVAL_FILE = Path(__file__).resolve().parent / "evalarticles" / "evalarticles.json"


def stable_article_id(url: str, title: str, content: str) -> str:
    key = (url or "").strip()
    if not key:
        key = ((title or "") + "||" + (content or "")[:400]).strip()
    hex_id = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()
    return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"


def article_fingerprint(title: str, text: str, url: str = "", n_chars: int = 400) -> str:
    u = (url or "").strip().lower()
    t = (title or "").strip().lower()
    lead = (text or "")[:n_chars].strip().lower()
    raw = (u + "||" + t + "||" + lead).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def _flatten_articles(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        # object format: {"articles": [...]} or single article dict
        if isinstance(node.get("articles"), list):
            for it in node["articles"]:
                yield from _flatten_articles(it)
        elif "title" in node or "url" in node:
            yield node
    elif isinstance(node, list):
        for it in node:
            yield from _flatten_articles(it)


def load_eval_articles(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    aliases: Dict[str, List[str]] = {}
    if isinstance(data, dict) and isinstance(data.get("interest_aliases"), dict):
        for k, vals in data["interest_aliases"].items():
            if isinstance(vals, list):
                aliases[str(k)] = [str(v) for v in vals]

    articles = [a for a in _flatten_articles(data)]
    out: List[Dict[str, Any]] = []
    for a in articles:
        title = str(a.get("title") or "")
        content = str(a.get("content") or "")
        url = str(a.get("url") or "")
        rec_id = str(a.get("article_id") or a.get("id") or stable_article_id(url, title, content))

        expected = a.get("expected_interest") or a.get("interest")
        if isinstance(expected, list):
            expected_interests = [str(x).strip() for x in expected if str(x).strip()]
        elif isinstance(expected, str) and expected.strip():
            expected_interests = [expected.strip()]
        else:
            expected_interests = []

        tags = a.get("tags") if isinstance(a.get("tags"), list) else []

        out.append(
            {
                "article_id": rec_id,
                "title": title,
                "content": content,
                "url": url,
                "published_date": a.get("published_date"),
                "source": a.get("source"),
                "lang": a.get("lang"),
                "expected_interests": expected_interests,
                "tags": [str(t) for t in tags],
                "rank_target_max": int(a.get("rank_target_max") or 20),
            }
        )
    return out, aliases


def normalize_interest_name(name: str, aliases: Dict[str, List[str]]) -> str:
    raw = (name or "").strip()
    low = raw.lower()
    for canonical, vals in aliases.items():
        if low == canonical.lower():
            return canonical
        for v in vals:
            if low == str(v).strip().lower():
                return canonical
    return raw


def _fetch_hits(store: PostgresStore, table: str, run_id: int) -> List[Dict[str, Any]]:
    if table == "retrieval_hits":
        blocks = store.fetch_retrieval_blocks(run_id)
    elif table == "dedup_hits":
        blocks = store.fetch_dedup_blocks(run_id)
    elif table == "rerank_hits":
        blocks = store.fetch_rerank_blocks(run_id)
    else:
        raise ValueError("table must be retrieval_hits, dedup_hits or rerank_hits")

    rows: List[Dict[str, Any]] = []
    for b in blocks:
        interest = str(b.get("interest") or "")
        for h in b.get("hits") or []:
            payload = h.get("payload") or {}
            rows.append(
                {
                    "interest": interest,
                    "rank": int(h.get("rank") or 0),
                    "article_id": str(h.get("id") or payload.get("article_id") or ""),
                    "url": str(payload.get("url") or ""),
                    "title": str(payload.get("title") or ""),
                    "score": float(h.get("score") or 0.0),
                    "rerank_score": float(h.get("rerank_score") or 0.0),
                }
            )
    return rows


def _build_lookup(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        aid = str(r.get("article_id") or "")
        url = str(r.get("url") or "")
        if aid:
            by_id.setdefault(aid, []).append(r)
        if url:
            by_url.setdefault(url, []).append(r)
    return by_id, by_url


def rank_points(rank: int, target_max: int = 20) -> float:
    if rank <= 0:
        return 0.0
    if rank <= min(3, target_max):
        return 1.0
    if rank <= min(10, target_max):
        return 0.7
    if rank <= target_max:
        return 0.4
    return 0.1


def evaluate(
    eval_articles: List[Dict[str, Any]],
    aliases: Dict[str, List[str]],
    hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id, by_url = _build_lookup(hits)

    per_interest: Dict[str, Dict[str, Any]] = {}
    details: List[Dict[str, Any]] = []

    total = 0
    total_ok = 0
    total_rank_points = 0.0

    for item in eval_articles:
        expected_interests = item.get("expected_interests") or []
        if not expected_interests:
            continue

        total += 1
        aid = str(item.get("article_id") or "")
        url = str(item.get("url") or "")
        candidates = []
        if aid:
            candidates.extend(by_id.get(aid, []))
        if (not candidates) and url:
            candidates.extend(by_url.get(url, []))

        found = len(candidates) > 0
        chosen = min(candidates, key=lambda x: x.get("rank", 10**9)) if candidates else None
        found_interest = chosen.get("interest") if chosen else None

        expected_norm = [normalize_interest_name(x, aliases) for x in expected_interests]
        found_norm = normalize_interest_name(str(found_interest or ""), aliases) if found_interest else None
        correct_interest = bool(found_norm and found_norm in expected_norm)

        is_ok = bool(found and correct_interest)
        if is_ok:
            total_ok += 1

        rk_target = int(item.get("rank_target_max") or 20)
        rk_points = rank_points(int(chosen.get("rank") or 0), target_max=rk_target) if is_ok and chosen else 0.0
        total_rank_points += rk_points

        interest_key = expected_norm[0]
        bucket = per_interest.setdefault(
            interest_key,
            {
                "expected": 0,
                "found_correct": 0,
                "rank_points": 0.0,
                "rank_count": 0,
                "articles": [],
            },
        )
        bucket["expected"] += 1
        if is_ok:
            bucket["found_correct"] += 1
            bucket["rank_points"] += rk_points
            bucket["rank_count"] += 1

        row = {
            "article_id": aid,
            "title": item.get("title"),
            "expected_interests": expected_norm,
            "found": found,
            "found_interest": found_interest,
            "found_rank": int(chosen.get("rank") or 0) if chosen else None,
            "ok": is_ok,
            "rank_points": rk_points,
        }
        details.append(row)
        bucket["articles"].append(row)

    interests_report: List[Dict[str, Any]] = []
    for k, v in sorted(per_interest.items(), key=lambda x: x[0].lower()):
        expected = int(v["expected"])
        found_correct = int(v["found_correct"])
        ratio = (found_correct / expected) if expected else 0.0
        rank_avg = (v["rank_points"] / max(1, v["rank_count"])) if v["rank_count"] else 0.0
        interests_report.append(
            {
                "interest": k,
                "score": f"{found_correct}/{expected}",
                "ratio": ratio,
                "rank_score_avg": rank_avg,
                "articles": v["articles"],
            }
        )

    global_ratio = (total_ok / total) if total else 0.0
    global_rank_avg = (total_rank_points / max(1, total_ok)) if total_ok else 0.0

    return {
        "total_expected": total,
        "total_found_correct": total_ok,
        "total_score": f"{total_ok}/{total}",
        "total_ratio": global_ratio,
        "global_rank_score_avg": global_rank_avg,
        "by_interest": interests_report,
        "details": details,
    }


def upsert_eval_articles(store: PostgresStore, eval_articles: List[Dict[str, Any]]) -> int:
    rows = []
    for a in eval_articles:
        title = str(a.get("title") or "")
        content = str(a.get("content") or "")
        url = str(a.get("url") or "")
        rec_id = str(a.get("article_id") or stable_article_id(url, title, content))
        fp = article_fingerprint(title, content, url=url)
        rows.append(
            {
                "id": rec_id,
                "title": title,
                "content": content,
                "published_date": a.get("published_date"),
                "source": a.get("source"),
                "url": url or None,
                "fingerprint": fp,
                "lang": a.get("lang"),
                "raw": a,
            }
        )
    return store.upsert_articles(rows)


def print_report(report: Dict[str, Any]) -> None:
    for b in report.get("by_interest", []):
        print(
            f"- {b['interest']} score {b['score']} | "
            f"rank_score_avg={b['rank_score_avg']:.2f}"
        )
    print(
        f"TOTAL: {report['total_score']} | "
        f"ratio={report['total_ratio']:.2%} | "
        f"global_rank_score_avg={report['global_rank_score_avg']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/rerank runs with eval articles")
    parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/pfe_news")
    parser.add_argument("--run-id", type=int, required=True, help="retrieval_run_id or rerank_run_id")
    parser.add_argument(
        "--table",
        choices=["retrieval_hits", "dedup_hits", "rerank_hits"],
        required=True,
        help="Table to evaluate",
    )
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--out", default=None, help="Optional JSON report output path")
    parser.add_argument(
        "--upsert-articles",
        action="store_true",
        help="Upsert eval articles into articles table before evaluation",
    )
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    eval_articles, aliases = load_eval_articles(eval_file)

    store = PostgresStore(args.db_url)
    store.init_db()

    if args.upsert_articles:
        n = upsert_eval_articles(store, eval_articles)
        print(f"Upserted eval articles into articles table: {n}")

    hits = _fetch_hits(store, args.table, int(args.run_id))
    report = evaluate(eval_articles, aliases, hits)

    print_report(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())