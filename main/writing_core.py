#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core helpers for the DB-backed writing stage.

The active pipeline reads reranked articles from PostgreSQL, calls Ollama one
article at a time, parses a plain-text response, then stores the generated
English title and English summary back into PostgreSQL.

Expected LLM output:
    - line 1: English title
    - line 2+: English summary

The historical column name `summary_fr` is kept for compatibility, even though
the generated summary is now in English.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# Robust parsing / cleaning
# -------------------------

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
TITLE_PREFIX_RE = re.compile(r"^(?:title|headline|titre)\s*:\s*", re.IGNORECASE)
SUMMARY_PREFIX_RE = re.compile(r"^(?:summary|résumé|resume)\s*:\s*", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
SINGLE_LINE_LABELED_RE = re.compile(
    r"^\s*(?:title|headline|titre)\s*:\s*(?P<title>.+?)\s+"
    r"(?:summary|résumé|resume)\s*:\s*(?P<summary>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

BAD_OUTPUT_PATTERNS = [
    "constraint checklist",
    "confidence score",
    "drafting the title",
    "drafting the summary",
    "input data:",
    "output plain text only",
    "first line only",
    "second line onward",
    "no json",
    "no bullet points",
    "do not reveal reasoning",
    "i should use a clean english news title",
    "key points from text",
]


def strip_think_and_fences(text: str) -> str: 
    text = THINK_RE.sub("", text or "")
    text = FENCE_RE.sub("", text).strip()
    return text


def extract_json_object(text: str) -> Optional[str]:
    """
    Essaye d'extraire un objet JSON même si le modèle ajoute du texte autour.
    """
    t = strip_think_and_fences(text)
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        return t
    i = t.find("{")
    j = t.rfind("}")
    if i >= 0 and j > i:
        return t[i : j + 1].strip()
    return None


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    raw = extract_json_object(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def clean_generated_title(title: Optional[str]) -> str:
    text = strip_think_and_fences(title or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    if "\n" in text:
        text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    text = TITLE_PREFIX_RE.sub("", text)
    text = re.sub(r"^[#>*\-\s]+", "", text).strip()
    text = text.strip("\"'“”‘’ ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :-–—")


def clean_generated_summary(summary: Optional[str]) -> str:
    # 1. Nettoyage initial et phrases interdites
    text = (summary or "").replace("\r\n", "\n").strip()
    forbidden_starts = ["According to", "Based on", "Here is", "This article", "I have analyzed"]
    
    for phrase in forbidden_starts:
        if text.lower().startswith(phrase.lower()):
            parts = text.split('.', 1)
            if len(parts) > 1:
                text = parts[1].strip()
    
    # 2. Formatage des lignes
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
        
    cleaned_lines = []
    for idx, line in enumerate(lines):
        if idx == 0:
            line = SUMMARY_PREFIX_RE.sub("", line).strip()
        cleaned_lines.append(line)
        
    return " ".join(cleaned_lines).strip()


def normalized_text_key(text: Optional[str]) -> str:
    value = unicodedata.normalize("NFKD", (text or "").strip().lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_is_translated_or_rewritten(
    generated_title: Optional[str],
    original_title: Optional[str],
    original_lang: Optional[str],
) -> bool:
    generated = clean_generated_title(generated_title)
    if not generated:
        return False

    original = clean_generated_title(original_title)
    lang = (original_lang or "").strip().lower()
    if not original:
        return True

    if normalized_text_key(generated) != normalized_text_key(original):
        return True

    return lang.startswith("en")


def _looks_like_summary(text: str) -> bool:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) >= 140:
        return True
    sentence_marks = compact.count(". ") + compact.count("! ") + compact.count("? ")
    return sentence_marks >= 2


def parse_llm_article_output(raw: str, fallback_title: Optional[str] = None) -> Dict[str, Any]:
    # 1. Nettoyage initial : suppression des balises <think> et blocs de code
    cleaned_raw = strip_think_and_fences(raw).replace("\r\n", "\n").strip()

    if not cleaned_raw:
        return {"title": "", "summary_fr": "", "points_cles": [], "notes": "Empty model response"}

    # 2. Tentative "Brutale" : Extraction de JSON n'importe où dans le texte
    json_start = cleaned_raw.find('{')
    json_end = cleaned_raw.rfind('}')
    if json_start != -1 and json_end != -1:
        try:
            json_str = cleaned_raw[json_start : json_end + 1]
            data = json.loads(json_str)
            return normalize_llm_article_output(data)
        except json.JSONDecodeError:
            pass # On continue si le JSON est corrompu vers le parsing manuel

    # 3. Parsing manuel (Fall-back) : Si le modèle n'a pas renvoyé de JSON valide
    labeled = SINGLE_LINE_LABELED_RE.match(cleaned_raw)
    if labeled:
        title = clean_generated_title(labeled.group("title"))
        summary = clean_generated_summary(labeled.group("summary"))
        out = {"title": title, "summary_fr": summary, "points_cles": []}
        if not title or not summary:
            out["notes"] = "Partial model response"
        return out

    lines = [line.rstrip() for line in cleaned_raw.splitlines()]
    first_idx = next((idx for idx, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        return {"title": "", "summary_fr": "", "points_cles": [], "notes": "Empty model response"}

    title = clean_generated_title(lines[first_idx])
    summary_lines = [line.strip() for line in lines[first_idx + 1 :] if line.strip()]
    
    # Application du nettoyeur de summary (avec les phrases interdites)
    summary = clean_generated_summary("\n".join(line for line in summary_lines if line))

    if not summary and _looks_like_summary(title):
        summary = cleaned_raw
        title = ""

    out = {"title": title, "summary_fr": summary, "points_cles": []}
    missing = []
    if not out["title"]:
        missing.append("title")
    if not out["summary_fr"]:
        missing.append("summary")
    if missing:
        out["notes"] = f"Missing parsed fields: {', '.join(missing)}"
    
    return out


def looks_like_contaminated_output(raw: str, parsed: Optional[Dict[str, Any]] = None) -> bool:
    cleaned = strip_think_and_fences(raw).strip()
    lowered = cleaned.lower()
    if any(marker in lowered for marker in BAD_OUTPUT_PATTERNS):
        return True

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    bullet_lines = sum(1 for line in lines if LIST_MARKER_RE.match(line))
    if bullet_lines >= 3:
        return True

    if parsed:
        title = str(parsed.get("title") or "").strip().lower()
        summary = str(parsed.get("summary_fr") or "").strip().lower()
        if any(marker in title for marker in BAD_OUTPUT_PATTERNS):
            return True
        if any(marker in summary for marker in BAD_OUTPUT_PATTERNS):
            return True
        if summary and bullet_lines >= 1 and ("*" in summary or "- " in summary or "1." in summary):
            return True

    return False


def is_usable_summary_output(summary: Optional[str], raw: str) -> bool:
    cleaned_summary = clean_generated_summary(summary)
    if not cleaned_summary:
        return False
    if len(cleaned_summary) < 350:
        return False
    if looks_like_contaminated_output(raw, {"summary_fr": cleaned_summary}):
        return False
    return True


def is_usable_title_output(title: Optional[str], raw: str, original_title: Optional[str], original_lang: Optional[str]) -> bool:
    cleaned_title = clean_generated_title(title)
    if not cleaned_title:
        return False
    if len(cleaned_title) < 8:
        return False
    if len(cleaned_title) > 160:
        return False
    if looks_like_contaminated_output(raw, {"title": cleaned_title}):
        return False
    if not title_is_translated_or_rewritten(cleaned_title, original_title, original_lang):
        return False
    return True


def parse_summary_output(raw: str) -> Dict[str, Any]:
    cleaned = clean_generated_summary(strip_think_and_fences(raw))
    out: Dict[str, Any] = {"summary_fr": cleaned, "points_cles": []}
    if not cleaned:
        out["notes"] = "Empty summary response"
    return out


def parse_title_output(raw: str) -> Dict[str, Any]:
    cleaned = clean_generated_title(raw)
    out: Dict[str, Any] = {"title": cleaned}
    if not cleaned:
        out["notes"] = "Empty title response"
    return out


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def slugify(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s or "unknown_interest")[:max_len]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_input(data: Any) -> List[Dict[str, Any]]:
    """
    Supporte:
      - {"results": [...]} (attendu)
      - [...] (liste de blocks)
    """
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Format JSON inattendu: attendu {'results': [...]} ou une liste [...]")


def pick_article_meta(hit: Dict[str, Any]) -> Dict[str, Any]:
    payload = hit.get("payload") or {}
    full_article = hit.get("full_article") or {}
    return {
        "article_id": payload.get("article_id") or full_article.get("id") or hit.get("id"),
        "title": payload.get("title") or full_article.get("title"),
        "url": payload.get("url") or full_article.get("url"),
        "source": payload.get("domain") or full_article.get("source"),
        "published_date": full_article.get("published_date") or payload.get("date"),
        "lang": payload.get("lang"),
        "rerank_score": hit.get("rerank_score"),
        "dense_score": hit.get("score"),
        "rank": hit.get("rank"),
    }


def stable_article_text(hit: Dict[str, Any], max_chars: int) -> str:
    payload = hit.get("payload") or {}
    full_article = hit.get("full_article") or {}
    text = full_article.get("content") or payload.get("canonical_text") or payload.get("description") or ""
    text = (text or "").strip()

    if len(text) <= max_chars:
        return text

    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head.rstrip() + "\n...\n" + tail.lstrip()


def group_by_interest(blocks: List[Dict[str, Any]], top_n: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Concatène les hits par intérêt, dédoublonne, trie, garde top_n.
    """
    bucket: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        interest = (block or {}).get("interest") or "unknown_interest"
        hits = (block or {}).get("hits") or []
        bucket.setdefault(interest, []).extend(hits)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for interest, hits in bucket.items():
        seen = set()
        dedup: List[Dict[str, Any]] = []

        for h in hits:
            meta = pick_article_meta(h)
            key = meta.get("article_id") or meta.get("url") or h.get("id")
            if not key:
                key = sha1(json.dumps(h, ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(h)

        def sort_key(h: Dict[str, Any]) -> Tuple[float, int]:
            rr = h.get("rerank_score")
            rr = float(rr) if rr is not None else -1.0
            rank = h.get("rank")
            rank = int(rank) if rank is not None else 10**9
            return (-rr, rank)

        dedup.sort(key=sort_key)
        out[interest] = dedup[:top_n]

    return out


# -------------------------
# Ollama client
# -------------------------

@dataclass
class OllamaConfig:
    model: str
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.2
    num_predict: int = 500
    num_ctx: int = 4096
    timeout_s: int = 180
    sleep_s: float = 0.05
    enforce_json: bool = True  # utilise "format": "json" si possible


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, debug: bool = False) -> None:
        self.cfg = cfg
        self.debug = debug

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = self.cfg.ollama_url.rstrip("/") + "/api/chat"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        # IMPORTANT : force une réponse JSON si Ollama le supporte
        if self.cfg.enforce_json:
            payload["format"] = "json"

        r = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content") or ""


# -------------------------
# Prompting (article-by-article)
# -------------------------

SUMMARY_SYSTEM_PROMPT = """You are a strictly constrained text processing engine.
Task: Generate one English summary paragraph (8-12 sentences) from the provided article.
CRITICAL RULES:
1. OUTPUT ONLY a raw JSON object with the exact keys: {"summary_fr": "...", "points_cles": [...]}.
2. CONTENT RULE: The field "summary_fr" MUST contain the summary written in ENGLISH.
3. DO NOT generate summaries in other languages (no French, no Ukrainian, etc.).
4. FORBIDDEN: Do not generate fields like "summary_en", "summary_uk", "summary_ru", "summary_de", etc.
5. FORBIDDEN: Do not output text in any language other than English.
6. DO NOT output any introductory text, analysis, or conversational filler.
7. DO NOT repeat the prompt instructions.
8. Output MUST be a single raw JSON object. Do not write a single word before '{' or after '}'.
9. If the article contains multiple languages, process the content but output ONLY English.
"""

SUMMARY_RETRY_SYSTEM_PROMPT = """Your previous response failed the formatting rules.
Task: Generate one English summary paragraph.
CRITICAL RULES:
1. OUTPUT ONLY the JSON format: {"summary_fr": "...", "points_cles": [...]}.
2. ZERO conversational filler.
3. ZERO analysis of the input.
4. ONLY the requested JSON structure.
"""

TITLE_SYSTEM_PROMPT = """Return only a short English news title for the article.
The title must be concise, natural, and informative.
Use one line only.
No analysis.
No checklist.
No bullet list.
No labels.
No markdown.
Do not mention the prompt.
Do not mention instructions.
If the source headline is not English, rewrite it in natural English instead of copying it.
"""

TITLE_RETRY_SYSTEM_PROMPT = """Your previous answer was invalid.
Retry now.
Return only one short English news title.
One line only.
No analysis.
No checklist.
No bullet list.
No labels.
No markdown.
Do not mention the prompt.
Do not copy a non-English source headline verbatim.
"""


def _article_prompt_body(interest: str, meta: Dict[str, Any], content: str) -> str:
    return "\n".join(
        [
            f"Interest: {interest}",
            f"Original title: {meta.get('title') or ''}",
            f"Source: {meta.get('source') or ''}",
            f"Published date: {meta.get('published_date') or ''}",
            f"Original language: {meta.get('lang') or ''}",
            "",
            "Use only the article text below.",
            "<article>",
            content,
            "</article>",
        ]
    )


def _title_prompt_body(interest: str, meta: Dict[str, Any], content: str, summary: str) -> str:
    parts = [
        f"Interest: {interest}",
        f"Original title: {meta.get('title') or ''}",
        f"Original language: {meta.get('lang') or ''}",
        f"Source: {meta.get('source') or ''}",
    ]
    if summary.strip():
        parts.extend([
            "",
            "Generated summary:",
            summary,
        ])
    parts.extend([
        "",
        "Article text:",
        "<article>",
        content,
        "</article>",
    ])
    return "\n".join(parts)


def build_summary_messages(interest: str, meta: Dict[str, Any], content: str) -> List[Dict[str, str]]:
    user_payload = _article_prompt_body(interest, meta, content)
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def build_summary_retry_messages(interest: str, meta: Dict[str, Any], content: str, previous_raw: str) -> List[Dict[str, str]]:
    user_payload = "\n\n".join(
        [
            _article_prompt_body(interest, meta, content),
            "Invalid previous answer to ignore:",
            previous_raw[:2000],
        ]
    )
    return [
        {"role": "system", "content": SUMMARY_RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def build_title_messages(interest: str, meta: Dict[str, Any], content: str, summary: str) -> List[Dict[str, str]]:
    user_payload = _title_prompt_body(interest, meta, content, summary)
    return [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def build_title_retry_messages(interest: str, meta: Dict[str, Any], content: str, summary: str, previous_raw: str) -> List[Dict[str, str]]:
    user_payload = "\n\n".join(
        [
            _title_prompt_body(interest, meta, content, summary),
            "Invalid previous answer to ignore:",
            previous_raw[:1000],
        ]
    )
    return [
        {"role": "system", "content": TITLE_RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]


def build_article_messages(interest: str, meta: Dict[str, Any], content: str) -> List[Dict[str, str]]:
    return build_summary_messages(interest, meta, content)


def build_article_retry_messages(interest: str, meta: Dict[str, Any], content: str, previous_raw: str) -> List[Dict[str, str]]:
    return build_summary_retry_messages(interest, meta, content, previous_raw)


def normalize_llm_article_output(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    On tolère quelques variantes, mais on garantit la présence de summary_fr & points_cles.
    """
    if not parsed or not isinstance(parsed, dict):
        return {"summary_fr": "", "points_cles": [], "notes": "Réponse non-JSON ou vide"}

    # Le modèle peut parfois renvoyer "articles":[{...}] => on prend le premier
    if isinstance(parsed.get("articles"), list) and parsed["articles"]:
        first = parsed["articles"][0]
        if isinstance(first, dict):
            parsed = first

    summary = parsed.get("summary_fr")
    points = parsed.get("points_cles")
    notes = parsed.get("notes")

    if not isinstance(summary, str):
        summary = ""

    if isinstance(points, list):
        points = [str(x) for x in points if str(x).strip()]
    else:
        points = []

    if notes is not None and not isinstance(notes, str):
        notes = str(notes)

    out = {"summary_fr": summary.strip(), "points_cles": points}
    if notes:
        out["notes"] = notes.strip()
    return out


# -------------------------
# Main (incremental writing)
# -------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Legacy article-by-article export mode via Ollama.")
    ap.add_argument("--input", required=True, help="Chemin du JSON qwenrerankbge (ex: qwenrerankbge.json)")
    ap.add_argument("--outdir", default="interest", help="Dossier de sortie (default: ./interest)")
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=9000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_predict", type=int, default=500)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--overwrite", action="store_true", help="Réécrire les fichiers d'intérêt (sinon resume/reprise)")
    ap.add_argument("--no_enforce_json", action="store_true", help="Legacy option kept for compatibility")
    ap.add_argument("--debug", action="store_true", help="Affiche la sortie brute du LLM dans la console")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] input introuvable: {in_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    cfg = OllamaConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        timeout_s=args.timeout,
        sleep_s=args.sleep,
        enforce_json=(not args.no_enforce_json),
    )
    client = OllamaClient(cfg, debug=args.debug)

    blocks = normalize_input(load_json(in_path))
    grouped = group_by_interest(blocks, top_n=args.top_n)

    interests = list(grouped.keys())
    print(f"[INFO] {len(interests)} intérêts détectés. Sortie: {outdir.resolve()}")
    print(f"[INFO] model={cfg.model} url={cfg.ollama_url} num_ctx={cfg.num_ctx} num_predict={cfg.num_predict} enforce_json={cfg.enforce_json}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(in_path),
        "ollama_url": cfg.ollama_url,
        "model": cfg.model,
        "params": {
            "top_n": args.top_n,
            "max_chars": args.max_chars,
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
            "num_ctx": cfg.num_ctx,
            "enforce_json": cfg.enforce_json,
        },
        "interests": [],
    }

    for idx, interest in enumerate(interests, 1):
        slug = slugify(interest)
        out_path = outdir / f"{slug}.json"

        # init / resume
        if out_path.exists() and not args.overwrite:
            try:
                state = load_json(out_path)
                if not isinstance(state, dict):
                    raise ValueError("fichier pas un objet JSON")
            except Exception:
                # si fichier corrompu -> on repart
                state = {"interest": interest, "articles": []}
        else:
            state = {"interest": interest, "articles": []}

        # build processed set
        processed = set()
        for a in state.get("articles", []) if isinstance(state.get("articles"), list) else []:
            if isinstance(a, dict):
                aid = a.get("article_id") or a.get("url")
                if aid:
                    processed.add(aid)

        hits = grouped[interest]
        print(f"[RUN ] {idx}/{len(interests)} {interest} (top={len(hits)}) already={len(processed)} file={out_path.name}")

        # meta (updated continuously)
        state["_meta"] = {
            "interest": interest,
            "slug": slug,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": cfg.model,
            "ollama_url": cfg.ollama_url,
            "input_file": str(in_path),
        }
        if "articles" not in state or not isinstance(state["articles"], list):
            state["articles"] = []

        # process each article
        for k, h in enumerate(hits, 1):
            meta = pick_article_meta(h)
            key = meta.get("article_id") or meta.get("url") or meta.get("title") or sha1(json.dumps(meta, ensure_ascii=False))
            if key in processed:
                continue

            content = stable_article_text(h, max_chars=args.max_chars)

            # If no content at all, don't call LLM
            if not content:
                article_out = {
                    **meta,
                    "title": "",
                    "summary_fr": "",
                    "points_cles": [],
                    "notes": "Article content missing in the source payload.",
                }
                state["articles"].append(article_out)
                processed.add(key)
                atomic_write_json(out_path, state)
                print(f"  [OK] {interest} [{k}/{len(hits)}] (no content) -> wrote")
                continue

            messages = build_article_messages(interest, meta, content)
            raw = client.chat(messages)

            if args.debug:
                print("\n" + "=" * 80)
                print(f"[DEBUG] interest={interest} | rank={meta.get('rank')} | title={meta.get('title')}")
                print("-" * 80)
                print(raw)
                print("=" * 80 + "\n")

            llm_out = parse_llm_article_output(raw)

            article_out = {**meta, **llm_out}
            if not isinstance(article_out.get("title"), str):
                article_out["title"] = ""
            # safety: ensure types
            if not isinstance(article_out.get("summary_fr"), str):
                article_out["summary_fr"] = ""
            if not isinstance(article_out.get("points_cles"), list):
                article_out["points_cles"] = []

            state["articles"].append(article_out)
            processed.add(key)

            # write after each article
            atomic_write_json(out_path, state)
            print(f"  [OK] {interest} [{k}/{len(hits)}] -> wrote")

            time.sleep(cfg.sleep_s)

        # update final meta per interest
        state["_meta"]["n_articles"] = len(state["articles"])
        state["_meta"]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(out_path, state)

        manifest["interests"].append({
            "interest": interest,
            "file": out_path.name,
            "n_articles_written": len(state.get("articles", [])),
        })

    atomic_write_json(outdir / "index.json", manifest)
    print("[DONE] index.json écrit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())