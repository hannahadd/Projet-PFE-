from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank retrieval candidates from PostgreSQL")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
    parser.add_argument("--retrieval-run-id", type=int, required=True)
    parser.add_argument("--model", default=core.DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--instruction", default=core.DEFAULT_INSTRUCTION)
    parser.add_argument("--hydrate", action="store_true", help="Attach full article from PostgreSQL articles table")
    args = parser.parse_args()

    store = PostgresStore(args.db_url)
    store.init_db()

    blocks = store.fetch_retrieval_blocks(args.retrieval_run_id)
    if not blocks:
        raise RuntimeError(f"No retrieval hits found for retrieval_run_id={args.retrieval_run_id}")

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
        if args.topn and args.topn > 0:
            augmented = augmented[: int(args.topn)]
        for i, h in enumerate(augmented, 1):
            h["rank"] = i

        out_blocks.append({"interest": interest, "n": len(augmented), "hits": augmented})

    rerank_run_id = store.create_rerank_run(
        {
            "retrieval_run_id": int(args.retrieval_run_id),
            "model": args.model,
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "topn": int(args.topn),
            "instruction": args.instruction,
        }
    )
    n_rows = store.insert_rerank_hits(rerank_run_id, out_blocks)
    print(f"Saved rerank run in PostgreSQL: rerank_run_id={rerank_run_id} | hits={n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
