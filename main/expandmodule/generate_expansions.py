from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence


PROMPT_TEMPLATE = """Give me exactly {count} 2025 trending words directly linked with "{topic}".

Strict rules:
- exactly {count} items
- exactly {min_words} words per item
- one item per line
- no numbering
- no bullets
- no explanation
- no intro
- no outro
- no full sentence
- no countries
- no nationalities
- each item must be directly usable to search news articles about this topic
- each item must be specific, concrete, and low-ambiguity

Reply only with the {count} items, one per line."""


def slugify(value: str, max_len: int = 120) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return (s or "topic")[:max_len]


def _strip_noise(line: str) -> str:
    text = (line or "").strip()
    text = re.sub(r"^[-*•]+\s*", "", text)
    text = re.sub(r"^\d+[\).:-]\s*", "", text)
    text = text.strip("`\"' ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_ascii(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_candidates(raw: str) -> List[str]:
    txt = (raw or "").replace("\r", "\n")
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.IGNORECASE | re.DOTALL)
    txt = re.sub(r"^```(?:text|markdown|json)?\s*|\s*```$", "", txt.strip(), flags=re.IGNORECASE | re.MULTILINE)

    lines: List[str] = []
    for line in txt.split("\n"):
        clean = _strip_noise(line)
        if clean:
            lines.append(clean)

    # fallback: if model returns comma/semicolon-separated text
    if len(lines) <= 1 and lines:
        one = lines[0]
        parts = [p.strip() for p in re.split(r"[,;|]", one) if p.strip()]
        if len(parts) > 1:
            lines = parts

    return lines


def _is_valid_item(item: str, min_words: int, max_words: int) -> bool:
    if not item:
        return False
    words = [w for w in re.findall(r"[\w][\w'\-]*", item, flags=re.UNICODE) if w]
    hard_max = max(int(max_words), 3)
    if len(words) < min_words or len(words) > hard_max:
        return False
    if any(len(w) == 1 for w in words):
        return False
    # ban very generic noisy lines
    low = item.lower().strip()
    banned = {
        "keyword",
        "keywords",
        "item",
        "items",
        "final items",
        "topic",
        "n/a",
        "could not parse",
        "last model output",
    }
    if low in banned:
        return False
    if "could not parse" in low or "last model output" in low:
        return False
    return True


def parse_items(raw: str, count: int, min_words: int, max_words: int) -> List[str]:
    candidates = _split_candidates(raw)
    out: List[str] = []
    seen = set()

    for c in candidates:
        c = _to_ascii(c)
        if not _is_valid_item(c, min_words=min_words, max_words=max_words):
            continue
        key = c.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= count:
            break

    if len(out) >= count:
        return out

    # Salvage pass: tolerate small over-length outputs (e.g. 3 words when max=2)
    # WITHOUT truncating phrases (avoid artifacts like "Crise de la").
    salvage: List[str] = []
    for c in candidates:
        words = [w for w in re.findall(r"[\w][\w'\-]*", c, flags=re.UNICODE) if w]
        if len(words) < min_words:
            continue
        hard_max = max(int(max_words), 3)
        if len(words) > hard_max:
            continue
        compact = c
        compact = _to_ascii(_strip_noise(compact))
        if not _is_valid_item(compact, min_words=min_words, max_words=max_words):
            continue
        key = compact.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        salvage.append(compact)
        if len(out) + len(salvage) >= count:
            break

    out.extend(salvage)

    return out


def call_ollama(model: str, prompt: str, timeout_s: int) -> str:
    cmd: Sequence[str] = ["ollama", "run", model, "--think=false", prompt]
    res = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ollama command failed (code={res.returncode}): {res.stderr.strip()}")
    return (res.stdout or "").strip()


def generate_items_for_topic(
    topic: str,
    model: str,
    count: int,
    min_words: int,
    max_words: int,
    timeout_s: int,
    retries: int,
) -> tuple[List[str], str]:
    if not topic.strip():
        raise ValueError("topic cannot be empty")

    prompt = PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        count=count,
        min_words=min_words,
        max_words=max_words,
    )

    last_raw = ""
    best_items: List[str] = []
    best_raw: str = ""
    for attempt in range(1, retries + 1):
        raw = call_ollama(model=model, prompt=prompt, timeout_s=timeout_s)
        last_raw = raw
        items = parse_items(raw, count=count, min_words=min_words, max_words=max_words)
        if len(items) > len(best_items):
            best_items = items
            best_raw = raw
        if len(items) == count:
            return items, raw

        repair_prompt = (
            f"Rewrite your previous answer.\n"
            f"Need exactly {count} items about \"{topic}\".\n"
            f"Each item must be {min_words} to {max_words} words.\n"
            f"One item per line.\n"
            f"No numbering. No bullets. No explanation. No countries.\n\n"
            f"Previous output:\n{raw}\n\n"
            f"Return only the corrected {count} items, one per line."
        )
        raw = call_ollama(model=model, prompt=repair_prompt, timeout_s=timeout_s)
        last_raw = raw
        items = parse_items(raw, count=count, min_words=min_words, max_words=max_words)
        if len(items) > len(best_items):
            best_items = items
            best_raw = raw
        if len(items) == count:
            return items, raw

    # Soft-fail policy: returning fewer items is acceptable.
    if best_items:
        return best_items, (best_raw or last_raw)

    raise RuntimeError(
        f"Could not parse any valid item for topic={topic!r}. "
        f"Last model output:\n{last_raw}"
    )


def write_interest_json(out_dir: Path, topic: str, model: str, items: List[str], raw_output: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = slugify(topic) + ".json"
    out_path = out_dir / filename
    payload = {
        "interest": topic,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "n_items": len(items),
        "items": items,
        "raw_output": raw_output,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_interest_error_json(out_dir: Path, topic: str, model: str, partial_items: List[str], error: str, raw_output: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = slugify(topic) + ".json"
    out_path = out_dir / filename
    payload = {
        "interest": topic,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "error": error,
        "n_items": len(partial_items),
        "items": partial_items,
        "raw_output": raw_output,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate keyword expansions per interest with Ollama")
    parser.add_argument("--topic", action="append", required=True, help="Interest topic (repeat for multiple)")
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--count", type=int, default=10, help="Number of expansion items")
    parser.add_argument("--min-words", type=int, default=1, help="Minimum words per item")
    parser.add_argument("--max-words", type=int, default=2, help="Maximum words per item")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per ollama call (seconds)")
    parser.add_argument("--retries", type=int, default=3, help="Retries for robust parsing")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "interest"),
        help="Output directory for one JSON file per interest",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing remaining topics if one topic fails.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.min_words <= 0 or args.max_words <= 0 or args.min_words > args.max_words:
        raise ValueError("invalid --min-words/--max-words")

    out_dir = Path(args.out_dir)
    topics = [t.strip() for t in (args.topic or []) if t and t.strip()]

    failed: List[str] = []

    for topic in topics:
        try:
            items, raw = generate_items_for_topic(
                topic=topic,
                model=args.model,
                count=int(args.count),
                min_words=int(args.min_words),
                max_words=int(args.max_words),
                timeout_s=int(args.timeout),
                retries=int(args.retries),
            )
            out_path = write_interest_json(out_dir=out_dir, topic=topic, model=args.model, items=items, raw_output=raw)
            print(f"Saved: {out_path}")
            for i, item in enumerate(items, 1):
                print(f"  {i:02d}. {item}")
            if len(items) < int(args.count):
                print(f"[WARN] {topic}: only {len(items)}/{args.count} valid items parsed")
        except Exception as exc:
            failed.append(topic)
            message = str(exc)
            out_path = write_interest_error_json(
                out_dir=out_dir,
                topic=topic,
                model=args.model,
                partial_items=[],
                error=message,
                raw_output=message,
            )
            print(f"[WARN] Topic failed: {topic}")
            print(f"[WARN] Saved partial/error JSON: {out_path}")
            if not args.continue_on_error:
                return 1

    if failed:
        print(f"Completed with warnings. Failed topics: {len(failed)}")
        for t in failed:
            print(f"  - {t}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
