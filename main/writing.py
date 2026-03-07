from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self.last_thinking: str = ""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        base = self.cfg.ollama_url.rstrip("/")
        self.last_thinking = ""

        chat_payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
                "num_ctx": self.cfg.num_ctx,
            },
        }

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
                "think": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.num_predict,
                    "num_ctx": self.cfg.num_ctx,
                },
            }
            gen_url = base + "/api/generate"
            response = requests.post(gen_url, json=gen_payload, timeout=self.cfg.timeout_s)
            response.raise_for_status()
            data = response.json()
            self.last_thinking = str(data.get("thinking") or "")
            return data.get("response") or ""

        response.raise_for_status()
        data = response.json()
        self.last_thinking = str((data.get("message") or {}).get("thinking") or "")
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


def _debug_raw_output(debug: bool, interest: str, meta: Dict[str, Any], raw: str, thinking: str = "") -> None:
    if not debug:
        return
    print("\n" + "=" * 80)
    print(f"[DEBUG] interest={interest} | rank={meta.get('rank')} | title={meta.get('title')}")
    if thinking.strip():
        print("[DEBUG] thinking returned by Ollama despite think=false:")
        print(thinking)
        print("-" * 80)
    print("-" * 80)
    print(raw)
    print("=" * 80 + "\n")


def _missing_fields(payload: Dict[str, Any]) -> List[str]:
    missing = []
    if not str(payload.get("title") or "").strip():
        missing.append("title")
    if not str(payload.get("summary_fr") or "").strip():
        missing.append("summary")
    return missing


def _build_llm_output(
    client: OllamaClientCompat,
    interest: str,
    meta: Dict[str, Any],
    content: str,
    debug: bool,
) -> Tuple[str, Dict[str, Any]]:
    if not content:
        return "WARN", {
            "title": "",
            "summary_fr": "",
            "points_cles": [],
            "notes": "Article content missing in PostgreSQL.",
            "raw_llm": {"reason": "missing_content"},
        }

    summary_raw = ""
    summary_thinking = ""
    title_raw = ""
    title_thinking = ""

    summary_out: Dict[str, Any] = {
        "summary_fr": "",
        "points_cles": [],
        "notes": "Empty summary response",
    }
    title_out: Dict[str, Any] = {
        "title": "",
        "notes": "Empty title response",
    }

    for attempt_idx in range(1, 3):
        if attempt_idx == 1:
            messages = core.build_summary_messages(interest, meta, content)
        else:
            messages = core.build_summary_retry_messages(interest, meta, content, summary_raw)

        raw = client.chat(messages)
        thinking = getattr(client, "last_thinking", "")
        _debug_raw_output(debug, interest, meta, raw, thinking)

        current_summary = core.parse_summary_output(raw)
        current_summary["summary_fr"] = core.clean_generated_summary(current_summary.get("summary_fr"))
        summary_raw = raw
        summary_thinking = thinking
        summary_out = current_summary

        if core.is_usable_summary_output(summary_out.get("summary_fr"), raw):
            break

    for attempt_idx in range(1, 3):
        if attempt_idx == 1:
            messages = core.build_title_messages(interest, meta, content, summary_out.get("summary_fr", ""))
        else:
            messages = core.build_title_retry_messages(interest, meta, content, summary_out.get("summary_fr", ""), title_raw)

        raw = client.chat(messages)
        thinking = getattr(client, "last_thinking", "")
        _debug_raw_output(debug, interest, meta, raw, thinking)

        current_title = core.parse_title_output(raw)
        current_title["title"] = core.clean_generated_title(current_title.get("title"))
        title_raw = raw
        title_thinking = thinking
        title_out = current_title

        if core.is_usable_title_output(title_out.get("title"), raw, meta.get("title"), meta.get("lang")):
            break

    llm_out: Dict[str, Any] = {
        "title": core.clean_generated_title(title_out.get("title")),
        "summary_fr": core.clean_generated_summary(summary_out.get("summary_fr")),
        "points_cles": [],
        "raw_llm": {
            "summary": {"raw": summary_raw, "thinking": summary_thinking},
            "title": {"raw": title_raw, "thinking": title_thinking},
        },
    }

    contaminated = core.looks_like_contaminated_output(summary_raw, {"summary_fr": llm_out.get("summary_fr")}) or core.looks_like_contaminated_output(title_raw, {"title": llm_out.get("title")})
    missing = _missing_fields(llm_out)
    notes: List[str] = []
    if not core.is_usable_summary_output(llm_out.get("summary_fr"), summary_raw):
        notes.append("summary invalid or too short")
    if not core.is_usable_title_output(llm_out.get("title"), title_raw, meta.get("title"), meta.get("lang")):
        notes.append("title not translated or rewritten")
    if contaminated:
        notes.append("invalid echoed/instructional output")
    if missing:
        notes.append(f"missing {', '.join(missing)}")
    if summary_thinking.strip() or title_thinking.strip():
        notes.append("thinking text returned by model")
    if notes:
        llm_out["notes"] = "; ".join(dict.fromkeys(notes))
    status = "OK" if not notes else ("FAIL" if contaminated or missing else "WARN")
    return status, llm_out


def _repair_hit_from_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(ctx.get("payload") or {})
    full_article = dict(ctx.get("full_article") or {})
    if ctx.get("article_id") and not payload.get("article_id"):
        payload["article_id"] = ctx.get("article_id")
    if ctx.get("url") and not payload.get("url"):
        payload["url"] = ctx.get("url")
    if ctx.get("title") and not payload.get("title"):
        payload["title"] = ctx.get("title")
    if ctx.get("source") and not payload.get("domain"):
        payload["domain"] = ctx.get("source")
    if ctx.get("published_date") and not payload.get("date"):
        payload["date"] = ctx.get("published_date")
    if ctx.get("lang") and not payload.get("lang"):
        payload["lang"] = ctx.get("lang")
    if ctx.get("article_id") and not full_article.get("id"):
        full_article["id"] = ctx.get("article_id")
    if ctx.get("title") and not full_article.get("title"):
        full_article["title"] = ctx.get("title")
    if ctx.get("url") and not full_article.get("url"):
        full_article["url"] = ctx.get("url")
    if ctx.get("source") and not full_article.get("source"):
        full_article["source"] = ctx.get("source")
    if ctx.get("published_date") and not full_article.get("published_date"):
        full_article["published_date"] = ctx.get("published_date")
    if ctx.get("lang") and not full_article.get("lang"):
        full_article["lang"] = ctx.get("lang")
    return {"id": ctx.get("article_id"), "rank": ctx.get("rank"), "payload": payload, "full_article": full_article}


def _maybe_enrich_repair_hit(store: PostgresStore, ctx: Dict[str, Any], hit: Dict[str, Any]) -> Dict[str, Any]:
    content = _stable_article_text(hit, max_chars=10**9)
    if content:
        return hit

    article = store.find_article(ctx.get("article_id"), ctx.get("url"))
    if not article:
        return hit

    payload = dict(hit.get("payload") or {})
    full_article = dict(hit.get("full_article") or {})
    payload.setdefault("article_id", article.get("id"))
    payload.setdefault("title", article.get("title"))
    payload.setdefault("url", article.get("url"))
    payload.setdefault("domain", article.get("source"))
    payload.setdefault("date", article.get("published_date"))
    payload.setdefault("lang", article.get("lang"))
    full_article.setdefault("id", article.get("id"))
    full_article.setdefault("title", article.get("title"))
    full_article.setdefault("url", article.get("url"))
    full_article.setdefault("source", article.get("source"))
    full_article.setdefault("published_date", article.get("published_date"))
    full_article.setdefault("lang", article.get("lang"))
    full_article.setdefault("content", article.get("content"))
    return {"id": article.get("id"), "rank": ctx.get("rank"), "payload": payload, "full_article": full_article}


def _run_repair(store: PostgresStore, client: OllamaClientCompat, repair_id: int, args: argparse.Namespace) -> int:
    ctx = store.fetch_article_summary_repair_context(repair_id)
    if not ctx:
        raise RuntimeError(f"No article_summaries row found with id={repair_id}")

    missing_before = _missing_fields(ctx)
    if not missing_before:
        print(f"[REPAIR] article_summary_id={repair_id} already complete. Nothing to do.")
        return 0

    hit = _repair_hit_from_context(ctx)
    hit = _maybe_enrich_repair_hit(store, ctx, hit)
    meta = _pick_meta(hit)
    meta["article_id"] = meta.get("article_id") or ctx.get("article_id")
    meta["title"] = meta.get("title") or ctx.get("title")
    meta["url"] = meta.get("url") or ctx.get("url")
    meta["source"] = meta.get("source") or ctx.get("source")
    meta["published_date"] = meta.get("published_date") or ctx.get("published_date")
    meta["lang"] = meta.get("lang") or ctx.get("lang")
    meta["rank"] = meta.get("rank") or ctx.get("rank")
    meta["rerank_score"] = ctx.get("rerank_score")
    meta["dense_score"] = ctx.get("dense_score")

    print(
        f"[REPAIR] article_summary_id={repair_id} | writing_run_id={ctx.get('writing_run_id')} "
        f"| interest={ctx.get('interest')} | rank={ctx.get('rank')} | missing={','.join(missing_before)}"
    )

    try:
        status, llm_out = _build_llm_output(
            client=client,
            interest=str(ctx.get("interest") or "unknown_interest"),
            meta=meta,
            content=_stable_article_text(hit, max_chars=int(args.max_chars)),
            debug=bool(args.debug),
        )
    except Exception as exc:
        status = "FAIL"
        llm_out = {
            "title": "",
            "summary_fr": ctx.get("summary_fr") or "",
            "points_cles": [],
            "notes": f"Repair generation failed: {exc}",
            "raw_llm": {"error": str(exc)},
        }

    store.update_article_summary_by_id(repair_id, llm_out)
    missing_after = _missing_fields(llm_out)
    suffix = f" | still_missing={','.join(missing_after)}" if missing_after else ""
    print(f"[{status}] article_summary_id={repair_id} repaired{suffix}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize reranked articles from PostgreSQL using Ollama")
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pfe_news"))
    ap.add_argument("--rerank-run-id", type=int)
    ap.add_argument("--repair", type=int, help="Repair one row in article_summaries by numeric id")
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=9000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_predict", type=int, default=900)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--interest-batch-size", type=int, default=10, help="Number of interests to process in this run")
    ap.add_argument("--offset", type=int, default=0, help="Offset in interest list (for next batches)")
    args = ap.parse_args()

    if args.repair is None and args.rerank_run_id is None:
        ap.error("Either --rerank-run-id or --repair must be provided.")
    if args.repair is not None and args.rerank_run_id is not None:
        ap.error("Use either --rerank-run-id or --repair, not both.")

    store = PostgresStore(args.db_url)
    store.init_db()

    cfg = core.OllamaConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=float(args.temperature),
        num_predict=int(args.num_predict),
        num_ctx=int(args.num_ctx),
        timeout_s=int(args.timeout),
        sleep_s=float(args.sleep),
        enforce_json=False,
    )
    client = OllamaClientCompat(cfg, debug=bool(args.debug))

    if args.repair is not None:
        return _run_repair(store, client, int(args.repair), args)

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

            try:
                status, llm_out = _build_llm_output(
                    client=client,
                    interest=interest,
                    meta=meta,
                    content=content,
                    debug=bool(args.debug),
                )
            except Exception as exc:
                status = "FAIL"
                llm_out = {
                    "title": "",
                    "summary_fr": "",
                    "points_cles": [],
                    "notes": f"Generation failed: {exc}",
                    "raw_llm": {"error": str(exc)},
                }

            row = {
                "interest": interest,
                "rank": int(meta.get("rank") or k),
                "article_id": meta.get("article_id"),
                "title": core.clean_generated_title(llm_out.get("title")),
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
            article_summary_id = store.upsert_article_summary(writing_run_id, row)
            missing = _missing_fields(row)
            suffix = f" | missing={','.join(missing)}" if missing else ""
            print(
                f"  [{status}] {interest} [{k}/{len(hits)}] saved "
                f"(article_summary_id={article_summary_id}{suffix})"
            )
            time.sleep(float(args.sleep))

    print(f"Done. Summaries stored in PostgreSQL with writing_run_id={writing_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
