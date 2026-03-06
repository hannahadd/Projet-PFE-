#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rewrite.py
----------
Résumé (FR) des 10 articles finaux par intérêt, à partir du JSON qwenrerankbge,
en utilisant Ollama (Qwen 3.5 9B Instruct recommandé en q4/q8).

IMPORTANT : traite TOUJOURS article par article et écrit au fur et à mesure.

Sortie :
  interest/<slug_interet>.json  (1 fichier par intérêt, mis à jour après chaque article)
  interest/index.json           (manifest global)

Exemples :
  python rewrite.py --input qwenrerankbge.json --model "qwen3.5:9b-q4_K_M"
  python rewrite.py --input qwenrerankbge.json --model "qwen3.5:9b-q8_0" --num_ctx 8192
  python rewrite.py --input qwenrerankbge.json --model "qwen3.5:9b-q4_K_M" --debug

Prérequis :
  pip install requests
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

SYSTEM_PROMPT = """Tu es un assistant de synthèse d'articles de presse.
Ta mission : produire des résumés en français, factuels, compacts et lisibles.

RÈGLES IMPORTANTES (mode non-thinking) :
- N'affiche PAS ton raisonnement.
- N'écris pas de balises <think> (ni rien de similaire).
- Réponds UNIQUEMENT avec du JSON valide (sans texte autour, sans markdown).

FORMAT DE SORTIE (objet JSON) :
{
  "summary_fr": "3 à 6 phrases max, 1 paragraphe.",
  "points_cles": ["3 à 6 points courts"],
  "notes": "optionnel"
}

Si le contenu est absent / illisible :
- mets summary_fr="" et explique pourquoi dans notes.
"""


def build_article_messages(interest: str, meta: Dict[str, Any], content: str) -> List[Dict[str, str]]:
    user_payload = {
        "interest": interest,
        "article": {
            "article_id": meta.get("article_id"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "published_date": meta.get("published_date"),
            "source": meta.get("source"),
            "lang": meta.get("lang"),
            "content": content,
        },
        "constraints": {
            "language": "fr",
            "json_only": True,
            "no_reasoning": True,
            "summary_sentences": "3-6",
            "points_cles": "3-6",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


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
    ap = argparse.ArgumentParser(description="Résumé FR article-par-article via Ollama (écrit progressivement).")
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
    ap.add_argument("--no_enforce_json", action="store_true", help="N'utilise pas format=json côté Ollama")
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
                    "summary_fr": "",
                    "points_cles": [],
                    "notes": "Contenu article absent dans l'entrée (full_article/content/canonical_text/description vide).",
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

            parsed = safe_json_loads(raw)
            llm_out = normalize_llm_article_output(parsed)

            article_out = {**meta, **llm_out}
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