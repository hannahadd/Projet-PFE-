from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from db import PostgresStore


def resolve_input_path(filename: str) -> Path:
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        ROOT_DIR / "ingestion" / filename,
        ROOT_DIR / "ingestionTest" / filename,
        ROOT_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Tried: {[str(p) for p in candidates]}")


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


def parse_date_column(series: pd.Series) -> pd.Series:
    if series.dtype in ["int64", "float64"]:
        series = series.astype(str)
    parsed = pd.to_datetime(series, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(series.loc[mask], errors="coerce", utc=True)
    return parsed


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "headline" in df.columns:
        df["title"] = df["headline"]
    elif "title" not in df.columns:
        df["title"] = None

    if "text" in df.columns:
        df["content"] = df["text"]
    elif "article" in df.columns:
        df["content"] = df["article"]
    elif "description" in df.columns:
        df["content"] = df["description"]
    else:
        df["content"] = None

    if "date" in df.columns:
        df["published_date"] = parse_date_column(df["date"])
    elif "published_date" in df.columns:
        df["published_date"] = parse_date_column(df["published_date"])
    elif "published" in df.columns:
        df["published_date"] = parse_date_column(df["published"])
    else:
        df["published_date"] = pd.NaT

    if "source_name" in df.columns:
        df["source"] = df["source_name"]
    elif "domain" in df.columns:
        df["source"] = df["domain"]
    elif "source" not in df.columns:
        df["source"] = None

    if "link" in df.columns:
        df["url"] = df["link"]
    elif "url" not in df.columns:
        df["url"] = None

    if "lang" not in df.columns:
        df["lang"] = None

    out = df[["title", "content", "published_date", "source", "url", "lang"]].copy()
    out["title"] = out["title"].astype(str).str.strip()
    out["content"] = out["content"].astype(str).str.strip()

    out = out[(out["title"] != "") & (out["content"] != "")]
    return out


def to_records(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, r in merged_df.iterrows():
        published = r.get("published_date")
        if pd.isna(published):
            published_val: Optional[str] = None
        else:
            published_val = pd.Timestamp(published).isoformat()

        title = str(r.get("title") or "")
        content = str(r.get("content") or "")
        url = str(r.get("url") or "")
        rec_id = stable_article_id(url, title, content)
        fp = article_fingerprint(title, content, url=url)

        rec = {
            "id": rec_id,
            "title": title,
            "content": content,
            "published_date": published_val,
            "source": str(r.get("source") or "") or None,
            "url": url or None,
            "fingerprint": fp,
            "lang": str(r.get("lang") or "") or None,
        }
        rec["raw"] = dict(rec)
        records.append(rec)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize datasets and upsert into PostgreSQL")
    parser.add_argument("--csv", default="dataset_top20.csv", help="CSV dataset filename/path")
    parser.add_argument("--jsonl", default="ccnews_dataset.jsonl", help="JSONL dataset filename/path")
    parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/pfe_news")
    parser.add_argument("--no-db", action="store_true", help="Skip PostgreSQL upsert")
    parser.add_argument("--export-json", default=str(ROOT_DIR / "main" / "ingestiontable" / "merged_dataset.normalized.json"))
    args = parser.parse_args()

    csv_path = resolve_input_path(args.csv)
    jsonl_path = resolve_input_path(args.jsonl)

    df_csv = pd.read_csv(csv_path)
    df_jsonl = pd.read_json(jsonl_path, lines=True, encoding="utf-8")

    csv_norm = normalize_dataframe(df_csv)
    jsonl_norm = normalize_dataframe(df_jsonl)

    merged = pd.concat([csv_norm, jsonl_norm], ignore_index=True)
    merged = merged.drop_duplicates(subset=["url"], keep="first")

    records = to_records(merged)

    if args.export_json:
        out = Path(args.export_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_json(out, orient="records", force_ascii=False, indent=2)
        print(f"Export JSON: {out}")

    if not args.no_db:
        store = PostgresStore(args.db_url)
        store.init_db()
        n = store.upsert_articles(records)
        print(f"Upsert PostgreSQL articles: {n}")

    print(f"Fusion terminée. Nombre total d'articles: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
