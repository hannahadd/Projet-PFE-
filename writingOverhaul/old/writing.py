from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from db import PostgresStore


ROOT = Path(__file__).resolve().parents[1]
SRC_WRITING_PATH = Path(__file__).resolve().parent / "writing_core.py"


def _load_src_writing_module():
    spec = importlib.util.spec_from_file_location("src_writing_core", SRC_WRITING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SRC_WRITING_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load_src_writing_module()


class OllamaClientCompat:
    def __init__(self, cfg: Any, debug: bool = False):
        self.cfg = cfg
        self.debug = debug

    def chat(self, messages: List[Dict[str, str]]) -> str:
        base = self.cfg.ollama_url.rstrip("/")

        chat_payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if getattr(self.cfg, "enforce_json", False):
            chat_payload["format"] = "json"

        chat_url = base + "/api/chat"
        response = requests.post(chat_url, json=chat_payload, timeout=self.cfg.timeout_s)
        if response.status_code == 404:
            prompt = "\n".join([
                f"[{m.get('role','user')}]\n{m.get('content','')}" for m in messages
            ])
            gen_payload: Dict[str, Any] = {
                "model": self.cfg.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.num_predict,
                    "num_ctx": self.cfg.num_ctx,
                },
            }
            if getattr(self.cfg, "enforce_json", False):
                gen_payload["format"] = "json"
            gen_url = base + "/api/generate"
            response = requests.post(gen_url, json=gen_payload, timeout=self.cfg.timeout_s)
            response.raise_for_status()
            data = response.json()
            return data.get("response") or ""

        response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content") or ""


def _pick_meta(hit: Dict[str, Any]) -> Dict[str, Any]:
    payload = hit.get("payload") or {}
    full = hit.get("full_article") or {}
    return {
        "article_id": payload.get("article_id") or full.get("id") or hit.get("id"),
        "title": payload.get("title") or full.get("title"),
        "url": payload.get("url") or full.get("url"),
        "source": payload.get("domain") or full.get("source"),
        "published_date": full.get("published_date") or payload.get("date"),
        "lang": payload.get("lang") or full.get("lang"),
        "rerank_score": hit.get("rerank_score"),
        "dense_score": hit.get("score"),
        "rank": hit.get("rank"),
    }


def _stable_article_text(hit: Dict[str, Any], max_chars: int) -> str:
    payload = hit.get("payload") or {}
    full = hit.get("full_article") or {}
    txt = full.get("content") or payload.get("canonical_text") or payload.get("description") or ""
    txt = (txt or "").strip()
    if len(txt) <= max_chars:
        return txt
    head = txt[: int(max_chars * 0.7)]
    tail = txt[-int(max_chars * 0.3):]
    return head.rstrip() + "\n...\n" + tail.lstrip()


def _group_and_dedup(blocks: List[Dict[str, Any]], top_n: int) -> Dict[str, List[Dict[str, Any]]]:
    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        interest = (block or {}).get("interest") or "unknown_interest"
        bucket.setdefault(interest, []).extend((block or {}).get("hits") or [])

    out: Dict[str, List[Dict[str, Any]]] = {}
    for interest, hits in bucket.items():
        seen = set()
        dedup = []
        for h in hits:
            meta = _pick_meta(h)
            key = meta.get("article_id") or meta.get("url") or h.get("id")
            if not key:
                key = str(hash(json.dumps(h, ensure_ascii=False, sort_keys=True)))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(h)

        def sk(h: Dict[str, Any]) -> Tuple[float, int]:
            rr = h.get("rerank_score")
            rr = float(rr) if rr is not None else -1.0
            rank = h.get("rank")
            rank = int(rank) if rank is not None else 10**9
            return (-rr, rank)

        dedup.sort(key=sk)
        out[interest] = dedup[:top_n]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize reranked articles from PostgreSQL using Ollama")
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
    ap.add_argument("--rerank-run-id", type=int, required=True)
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=9000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_predict", type=int, default=500)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--no_enforce_json", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--interest-batch-size", type=int, default=10, help="Number of interests to process in this run")
    ap.add_argument("--offset", type=int, default=0, help="Offset in interest list (for next batches)")
    args = ap.parse_args()

    store = PostgresStore(args.db_url)
    store.init_db()

    blocks = store.fetch_rerank_blocks(args.rerank_run_id)
    if not blocks:
        raise RuntimeError(f"No rerank hits found for rerank_run_id={args.rerank_run_id}")

    grouped = _group_and_dedup(blocks, top_n=int(args.top_n))
    interests = sorted(grouped.keys())
    start = max(0, int(args.offset))
    end = start + max(1, int(args.interest_batch_size))
    selected_interests = interests[start:end]

    if not selected_interests:
        print("No interests selected for this batch. Nothing to process.")
        return 0

    cfg = core.OllamaConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=float(args.temperature),
        num_predict=int(args.num_predict),
        num_ctx=int(args.num_ctx),
        timeout_s=int(args.timeout),
        sleep_s=float(args.sleep),
        enforce_json=(not args.no_enforce_json),
    )
    client = OllamaClientCompat(cfg, debug=bool(args.debug))

    writing_run_id = store.create_writing_run(
        {
            "rerank_run_id": int(args.rerank_run_id),
            "model": args.model,
            "ollama_url": args.ollama_url,
            "interest_batch_size": int(args.interest_batch_size),
            "offset_interests": int(args.offset),
            "top_n": int(args.top_n),
            "max_chars": int(args.max_chars),
            "temperature": float(args.temperature),
            "num_predict": int(args.num_predict),
            "num_ctx": int(args.num_ctx),
        }
    )

    print(
        f"Writing run created: writing_run_id={writing_run_id} | "
        f"interests={len(selected_interests)} ({start}:{end})"
    )

    for idx, interest in enumerate(selected_interests, 1):
        hits = grouped.get(interest, [])
        print(f"[RUN] {idx}/{len(selected_interests)} {interest} (articles={len(hits)})")
        for k, h in enumerate(hits, 1):
            meta = _pick_meta(h)
            content = _stable_article_text(h, max_chars=int(args.max_chars))

            if not content:
                llm_out = {
                    "summary_fr": "",
                    "points_cles": [],
                    "notes": "Contenu article absent dans la base.",
                }
            else:
                messages = core.build_article_messages(interest, meta, content)
                raw = client.chat(messages)
                parsed = core.safe_json_loads(raw)
                llm_out = core.normalize_llm_article_output(parsed)
                llm_out["raw_llm"] = parsed if isinstance(parsed, dict) else {"raw": raw}

            row = {
                "interest": interest,
                "rank": int(meta.get("rank") or k),
                "article_id": meta.get("article_id"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "source": meta.get("source"),
                "published_date": meta.get("published_date"),
                "lang": meta.get("lang"),
                "rerank_score": meta.get("rerank_score"),
                "dense_score": meta.get("dense_score"),
                "summary_fr": llm_out.get("summary_fr", ""),
                "points_cles": llm_out.get("points_cles", []),
                "notes": llm_out.get("notes"),
                "raw_llm": llm_out.get("raw_llm", {}),
            }
            store.upsert_article_summary(writing_run_id, row)
            print(f"  [OK] {interest} [{k}/{len(hits)}] saved")
            time.sleep(float(args.sleep))

    print(f"Done. Summaries stored in PostgreSQL with writing_run_id={writing_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
