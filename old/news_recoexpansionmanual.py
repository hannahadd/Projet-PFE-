from __future__ import annotations

import argparse
import importlib.util
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


INTEREST_EXPANSIONS: Dict[str, List[str]] = {
    "AI and LLMs": [
        "Artificial Intelligence news",
        "Large language models usage",
        "Generative AI capabilities",
        "Deep learning algorithms",
        "Machine intelligence ethics",
    ],
    "SpaceX": [
        "SpaceX launch schedule",
        "SpaceX rocket failures",
        "SpaceX Starship tests",
        "SpaceX Elon Musk news",
        "SpaceX satellite launches",
    ],
    "Apple": [
        "Apple stock price",
        "Apple product launch",
        "Apple earnings report",
        "Apple CEO interview",
        "Apple supply chain",
    ],
    "french politics": [
        "French election results",
        "National parliamentary votes",
        "Government formation process",
        "European political alliances",
        "Legislative committee reports",
    ],
    "war and international conflict": [
        "War and peace treaties",
        "Military intervention operations",
        "International conflict resolutions",
        "Arms control agreements",
        "Global security crises",
    ],
}


def _expand_interest(interest: str) -> List[str]:
    base = (interest or "").strip()
    if not base:
        return []
    expanded = INTEREST_EXPANSIONS.get(base, [base])
    out: List[str] = []
    for s in expanded:
        s2 = str(s).strip()
        if s2 and s2 not in out:
            out.append(s2)
    return out


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
    dense_per_center: int,
    bm25_k: int,
    mmr_lambda_div: float,
    mmr_near_dup_threshold: float,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    if aggregate or len(interests) <= 1:
        expanded: List[str] = []
        for interest in interests:
            expanded.extend(_expand_interest(interest))
        if not expanded:
            expanded = interests
        user = core.UserProfile(interests_text=expanded, interests_tags=tags)
        user.build_from_onboarding(embedder.encode, k_max=min(8, len(expanded) or 1))
        scores_map: Dict[str, float] = {}
        feed = core.retrieve_feed(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            user=user,
            dense_per_center=dense_per_center,
            bm25_k=bm25_k,
            days=days,
            top_k=topk,
            use_rerank=True,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            scores_out=scores_map,
            mmr_lambda_div=mmr_lambda_div,
            mmr_near_dup_threshold=mmr_near_dup_threshold,
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
        expanded = _expand_interest(interest)
        user = core.UserProfile(interests_text=expanded, interests_tags=tags)
        user.build_from_onboarding(embedder.encode, k_max=min(6, len(expanded) or 1))
        scores_map: Dict[str, float] = {}
        feed = core.retrieve_feed(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            user=user,
            dense_per_center=dense_per_center,
            bm25_k=bm25_k,
            days=days,
            top_k=topk,
            use_rerank=True,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            scores_out=scores_map,
            mmr_lambda_div=mmr_lambda_div,
            mmr_near_dup_threshold=mmr_near_dup_threshold,
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
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="news_dense")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--interest", action="append", default=None)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--lang", default=None)
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--min-sim", type=float, default=0.15)
    parser.add_argument("--min-bm25", type=float, default=0.05)
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--dense-per-center", type=int, default=400)
    parser.add_argument("--bm25-k", type=int, default=400)
    parser.add_argument("--mmr-lambda", type=float, default=0.82)
    parser.add_argument("--mmr-near-dup-threshold", type=float, default=0.92)
    args = parser.parse_args()

    store = PostgresStore(args.db_url)
    store.init_db()

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] if args.tags else []

    db_rows = store.fetch_articles(limit=args.max_articles)
    if not db_rows:
        raise RuntimeError("No articles found in PostgreSQL table 'articles'. Run normalization/upsert first.")

    raw_articles = [core.Article.from_dict(r) for r in db_rows]
    articles = core.dedup_articles(raw_articles)
    print(f"Loaded from PostgreSQL: {len(raw_articles)} | After dedup: {len(articles)}")

    qindex = core.DenseIndexQdrant(url=args.qdrant_url, collection=args.collection)
    points_count = qindex.get_points_count() if qindex.collection_exists() else 0
    need_index = bool(args.reindex or points_count == 0)

    if need_index:
        embedder = core.Embedder("BAAI/bge-m3", use_fp16=True)
        texts = [a.canonical_text for a in articles]
        all_vecs = []
        bs = 32
        for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
            vecs = embedder.encode(texts[i : i + bs], batch_size=bs)
            all_vecs.append(vecs)
        all_vecs = np.vstack(all_vecs)
        for a, v in zip(articles, all_vecs):
            a.embedding = v
        dim = int(all_vecs.shape[1])
        if args.reindex:
            qindex.recreate(vector_dim=dim)
        else:
            qindex.ensure_collection(vector_dim=dim)
        qindex.upsert_articles(articles, batch_size=256)
        print("Qdrant upsert done.")
    else:
        print(f"Qdrant already has {points_count} points. Skipping re-embedding.")

    bm25 = core.BM25Index(articles)
    id_to_article = {a.id: a for a in articles}
    embedder = core.Embedder("BAAI/bge-m3", use_fp16=True)

    interests = [i for i in (args.interest or []) if i and i.strip()]
    if not interests:
        interests = [
            "war and international conflict",
            "AI and LLMs",
            "french politics",
            "SpaceX",
            "Apple",
        ]

    lang_filter = None
    if args.lang:
        lang_filter = {l.strip().lower() for l in args.lang.split(",") if l.strip()}
    min_bm25 = 0.0 if args.dense_only else args.min_bm25

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
        dense_per_center=int(args.dense_per_center),
        bm25_k=int(args.bm25_k),
        mmr_lambda_div=float(args.mmr_lambda),
        mmr_near_dup_threshold=float(args.mmr_near_dup_threshold),
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
