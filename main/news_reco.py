from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from db import PostgresStore


ROOT = Path(__file__).resolve().parents[1]
SRC_NEWS_PATH = Path(__file__).resolve().parent / "news_reco_core.py"


def _load_src_news_module():
    spec = importlib.util.spec_from_file_location("src_news_reco_core", SRC_NEWS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SRC_NEWS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load_src_news_module()


DEFAULT_INTERESTS: List[str] = [
    "war and international conflict",
    "AI and LLMs",
    "french politics",
    "SpaceX",
    "Apple",
]


INTEREST_EXPANSIONS: Dict[str, List[str]] = {}


def _fetch_articles_cache_metadata(store: PostgresStore) -> Dict[str, Any]:
    with store.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MAX(updated_at) FROM articles")
            count, max_updated_at = cur.fetchone()
    return {
        "count": int(count or 0),
        "max_updated_at": max_updated_at.isoformat() if max_updated_at is not None else None,
    }


def _load_postgres_articles_cached(store: PostgresStore, max_articles: Optional[int]) -> tuple[List[Any], int, bool]:
    meta = _fetch_articles_cache_metadata(store)
    signature = "|".join(
        [
            f"v={core.CACHE_VERSION}",
            f"db={core.namespace_hash(store.db_url)}",
            f"count={meta['count']}",
            f"max_updated_at={meta['max_updated_at']}",
            f"limit={max_articles if max_articles is not None else 'all'}",
        ]
    )
    cache_dir = core.default_cache_dir()
    cache_path = cache_dir / f"articles_pg_{core.namespace_hash(store.db_url)}_{max_articles if max_articles is not None else 'all'}.pkl"
    cached = core.load_pickle_cache(cache_path)
    if cached and cached.get("signature") == signature and isinstance(cached.get("articles"), list):
        return cached["articles"], int(cached.get("raw_count") or len(cached["articles"])), True

    db_rows = store.fetch_articles(limit=max_articles)
    raw_articles = [core.Article.from_dict(r) for r in db_rows]
    articles = core.dedup_articles(raw_articles)
    core.save_pickle_cache(
        cache_path,
        {
            "signature": signature,
            "raw_count": len(raw_articles),
            "articles": articles,
        },
    )
    return articles, len(raw_articles), False


def _candidate_expansion_dirs() -> List[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "expand" / "interest",
        here / "expandmodule" / "interest",
    ]


def _normalize_item_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def load_interest_expansions(expansions_dir: Optional[str]) -> Dict[str, List[str]]:
    if expansions_dir:
        base = Path(expansions_dir)
    else:
        candidates = [p for p in _candidate_expansion_dirs() if p.exists()]
        base = candidates[0] if candidates else _candidate_expansion_dirs()[0]

    if not base.exists() or not base.is_dir():
        print(f"[WARN] Expansions directory not found: {base}. Using raw interest text only.")
        return {}

    mapping: Dict[str, List[str]] = {}
    for path in sorted(base.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        interest = str((payload or {}).get("interest") or "").strip()
        items = (payload or {}).get("items") or []
        if not interest or not isinstance(items, list):
            continue

        normalized = _normalize_item_list([str(x) for x in items])
        if normalized:
            mapping[interest] = normalized

    print(f"Loaded expansions for {len(mapping)} interests from: {base}")
    return mapping


def _build_query_specs_for_interest(
    interest: str,
    embedder: Any,
    tags: List[str],
    max_expansions_per_interest: int,
) -> List[Any]:
    specs = core.build_interest_query_specs(
        interest=interest,
        expansions_map=INTEREST_EXPANSIONS,
        embedder=embedder,
        max_expansions=max_expansions_per_interest,
    )
    if tags:
        specs = core.merge_query_specs(specs + core.build_tag_query_specs(tags, embedder))
    return specs


def _build_aggregate_query_specs(
    interests: List[str],
    embedder: Any,
    tags: List[str],
    max_expansions_per_interest: int,
) -> List[Any]:
    specs: List[Any] = []
    for interest in interests:
        specs.extend(_build_query_specs_for_interest(interest, embedder, tags, max_expansions_per_interest))
    return core.merge_query_specs(specs)


def _resolve_default_interests() -> List[str]:
    from_expansions = [str(k).strip() for k in INTEREST_EXPANSIONS.keys() if str(k).strip()]
    if from_expansions:
        return from_expansions
    return list(DEFAULT_INTERESTS)


def _resolve_retrieval_parameters(args: argparse.Namespace) -> Dict[str, int]:
    dense_per_anchor = int(
        args.dense_per_anchor
        if args.dense_per_anchor is not None
        else (args.dense_per_center if args.dense_per_center is not None else 300)
    )
    if args.dense_per_expansion is not None:
        dense_per_expansion = int(args.dense_per_expansion)
    elif args.dense_per_center is not None:
        dense_per_expansion = max(1, int(round(args.dense_per_center * 0.4)))
    else:
        dense_per_expansion = 120

    if args.bm25_title_k is not None:
        bm25_title_k = int(args.bm25_title_k)
    elif args.bm25_k is not None:
        bm25_title_k = int(args.bm25_k)
    else:
        bm25_title_k = 80

    if args.bm25_body_k is not None:
        bm25_body_k = int(args.bm25_body_k)
    elif args.bm25_k is not None:
        bm25_body_k = int(args.bm25_k)
    else:
        bm25_body_k = 120

    return {
        "max_expansions_per_interest": int(args.max_expansions_per_interest),
        "dense_per_anchor": dense_per_anchor,
        "dense_per_expansion": dense_per_expansion,
        "bm25_title_k": bm25_title_k,
        "bm25_body_k": bm25_body_k,
        "rrf_k": int(args.rrf_k),
        "candidate_cap": int(args.candidate_cap),
    }


def _build_export_blocks(
    interests: List[str],
    aggregate: bool,
    qindex: Any,
    bm25: Any,
    id_to_article: Dict[str, Any],
    embedder: Any,
    tags: List[str],
    days: int,
    topk: int,
    lang_filter: Optional[Set[str]],
    min_sim: float,
    min_bm25: float,
    dense_only: bool,
    max_expansions_per_interest: int,
    dense_per_anchor: int,
    dense_per_expansion: int,
    bm25_title_k: int,
    bm25_body_k: int,
    rrf_k: int,
    candidate_cap: int,
    debug_retrieval: bool,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    if aggregate or len(interests) <= 1:
        query_specs = _build_aggregate_query_specs(
            interests=interests,
            embedder=embedder,
            tags=tags,
            max_expansions_per_interest=max_expansions_per_interest,
        )
        scores_map: Dict[str, float] = {}
        feed = core.retrieve_feed_multiquery_for_interest(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            top_k=topk,
            query_specs=query_specs,
            days=days,
            dense_per_anchor=dense_per_anchor,
            dense_per_expansion=dense_per_expansion,
            bm25_title_k=bm25_title_k,
            bm25_body_k=bm25_body_k,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            rrf_k=rrf_k,
            candidate_cap=candidate_cap,
            scores_out=scores_map,
            debug=debug_retrieval,
            debug_label="aggregate",
        )
        blocks.append(
            {
                "interest": "; ".join(interests) if interests else "aggregate",
                "n": len(feed),
                "hits": [
                    {
                        "rank": i,
                        "id": a.id,
                        "score": float(scores_map.get(a.id, 0.0)),
                        "payload": {
                            "article_id": a.id,
                            "title": a.title,
                            "description": a.description,
                            "url": a.url,
                            "domain": a.domain,
                            "date": a.date.isoformat() if a.date else None,
                            "lang": a.lang,
                            "fingerprint": a.fingerprint,
                            "canonical_text": a.canonical_text,
                        },
                    }
                    for i, a in enumerate(feed, 1)
                ],
            }
        )
        return blocks

    for interest in interests:
        query_specs = _build_query_specs_for_interest(
            interest=interest,
            embedder=embedder,
            tags=tags,
            max_expansions_per_interest=max_expansions_per_interest,
        )
        scores_map: Dict[str, float] = {}
        feed = core.retrieve_feed_multiquery_for_interest(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            top_k=topk,
            query_specs=query_specs,
            days=days,
            dense_per_anchor=dense_per_anchor,
            dense_per_expansion=dense_per_expansion,
            bm25_title_k=bm25_title_k,
            bm25_body_k=bm25_body_k,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            rrf_k=rrf_k,
            candidate_cap=candidate_cap,
            scores_out=scores_map,
            debug=debug_retrieval,
            debug_label=interest,
        )
        blocks.append(
            {
                "interest": interest,
                "n": len(feed),
                "hits": [
                    {
                        "rank": i,
                        "id": a.id,
                        "score": float(scores_map.get(a.id, 0.0)),
                        "payload": {
                            "article_id": a.id,
                            "title": a.title,
                            "description": a.description,
                            "url": a.url,
                            "domain": a.domain,
                            "date": a.date.isoformat() if a.date else None,
                            "lang": a.lang,
                            "fingerprint": a.fingerprint,
                            "canonical_text": a.canonical_text,
                        },
                    }
                    for i, a in enumerate(feed, 1)
                ],
            }
        )
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="News retrieval with PostgreSQL output")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
    parser.add_argument(
        "--expansions-dir",
        default=None,
        help="Directory containing one expansion JSON per interest (default: main/expand/interest or main/expandmodule/interest)",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="news_dense")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--interest", action="append", default=None)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--resume-index", action="store_true")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--min-sim", type=float, default=0.15)
    parser.add_argument("--min-bm25", type=float, default=0.05)
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--dense-per-center", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bm25-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-expansions-per-interest", type=int, default=6)
    parser.add_argument("--dense-per-anchor", type=int, default=None)
    parser.add_argument("--dense-per-expansion", type=int, default=None)
    parser.add_argument("--bm25-title-k", type=int, default=None)
    parser.add_argument("--bm25-body-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-cap", type=int, default=800)
    parser.add_argument("--debug-retrieval", action="store_true")
    args = parser.parse_args()

    if args.reindex and args.resume_index:
        parser.error("--reindex and --resume-index are mutually exclusive")

    global INTEREST_EXPANSIONS
    INTEREST_EXPANSIONS = load_interest_expansions(args.expansions_dir)

    store = PostgresStore(args.db_url)
    store.init_db()

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] if args.tags else []

    qindex = core.DenseIndexQdrant(url=args.qdrant_url, collection=args.collection)
    collection_exists = qindex.collection_exists()
    points_count = qindex.get_points_count() if collection_exists else 0
    need_index = bool(args.reindex or args.resume_index or points_count == 0)

    need_article_corpus = bool(need_index or not args.dense_only)
    articles: List[Any] = []
    id_to_article: Dict[str, Any] = {}
    bm25 = None

    if need_article_corpus:
        articles, raw_count, articles_from_cache = _load_postgres_articles_cached(store, args.max_articles)
        if not articles:
            raise RuntimeError("No articles found in PostgreSQL table 'articles'. Run normalization/upsert first.")
        cache_msg = " | cache=articles" if articles_from_cache else ""
        print(f"Loaded from PostgreSQL: {raw_count} | After dedup: {len(articles)}{cache_msg}")
        id_to_article = {a.id: a for a in articles}
    else:
        print("Dense-only fast path: skipping full PostgreSQL article load and local BM25 rebuild.")

    if need_index:
        articles_to_index = articles
        if args.resume_index and not args.reindex and collection_exists:
            existing_ids = qindex.get_existing_ids()
            articles_to_index = [a for a in articles if a.id not in existing_ids]
            print(f"Qdrant resume mode: existing={len(existing_ids)} | missing={len(articles_to_index)}")

        if articles_to_index:
            embedder = core.Embedder("BAAI/bge-m3", use_fp16=True)
            texts = [a.canonical_text for a in articles_to_index]
            all_vecs = []
            bs = 32
            for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
                vecs = embedder.encode(texts[i : i + bs], batch_size=bs)
                all_vecs.append(vecs)
            all_vecs = np.vstack(all_vecs)
            for a, v in zip(articles_to_index, all_vecs):
                a.embedding = v
            dim = int(all_vecs.shape[1])
            if args.reindex:
                qindex.recreate(vector_dim=dim)
            else:
                qindex.ensure_collection(vector_dim=dim)
            qindex.upsert_articles(articles_to_index, batch_size=256)
            print(f"Qdrant upsert done. Indexed {len(articles_to_index)} articles.")
        else:
            print(f"Qdrant already contains all {len(articles)} deduplicated articles. Skipping re-embedding.")
    else:
        print(f"Qdrant already has {points_count} points. Skipping re-embedding.")

    if not args.dense_only:
        bm25, bm25_from_cache = core.load_or_build_bm25(
            articles,
            cache_key=f"postgres:{args.db_url}:{args.max_articles if args.max_articles is not None else 'all'}",
        )
        if bm25_from_cache:
            print("Loaded BM25 from cache.")
        else:
            print("Built BM25 and saved cache.")

    embedder = core.Embedder("BAAI/bge-m3", use_fp16=True)

    interests = [i for i in (args.interest or []) if i and i.strip()]
    if not interests:
        interests = _resolve_default_interests()
        print(f"No --interest provided. Using all configured interests: {len(interests)}")

    lang_filter = None
    if args.lang:
        lang_filter = {l.strip().lower() for l in args.lang.split(",") if l.strip()}
    min_bm25 = 0.0 if args.dense_only else args.min_bm25
    retrieval_params = _resolve_retrieval_parameters(args)

    blocks = _build_export_blocks(
        interests=interests,
        aggregate=bool(args.aggregate),
        qindex=qindex,
        bm25=bm25,
        id_to_article=id_to_article,
        embedder=embedder,
        tags=tags,
        days=int(args.days),
        topk=int(args.topk),
        lang_filter=lang_filter,
        min_sim=float(args.min_sim),
        min_bm25=float(min_bm25),
        dense_only=bool(args.dense_only),
        max_expansions_per_interest=int(retrieval_params["max_expansions_per_interest"]),
        dense_per_anchor=int(retrieval_params["dense_per_anchor"]),
        dense_per_expansion=int(retrieval_params["dense_per_expansion"]),
        bm25_title_k=int(retrieval_params["bm25_title_k"]),
        bm25_body_k=int(retrieval_params["bm25_body_k"]),
        rrf_k=int(retrieval_params["rrf_k"]),
        candidate_cap=int(retrieval_params["candidate_cap"]),
        debug_retrieval=bool(args.debug_retrieval),
    )

    run_id = store.create_retrieval_run(
        {
            "data_path": "postgresql://articles",
            "qdrant_url": args.qdrant_url,
            "collection": args.collection,
            "dense_model": "BAAI/bge-m3",
            "days": int(args.days),
            "topk": int(args.topk),
            "dense_only": bool(args.dense_only),
            "aggregate": bool(args.aggregate),
            "tags": tags,
            "lang": args.lang,
            "min_sim": float(args.min_sim),
            "min_bm25": float(min_bm25),
            "interests": interests,
        }
    )
    n_hits = store.insert_retrieval_hits(run_id, blocks)

    print(f"Saved retrieval run in PostgreSQL: retrieval_run_id={run_id} | hits={n_hits}")
    for b in blocks:
        print(f"- {b['interest']}: {b['n']} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
