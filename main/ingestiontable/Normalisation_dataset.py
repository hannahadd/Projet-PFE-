from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
        ROOT_DIR / "main" / "ingestiontable" / filename,
        ROOT_DIR / "ingestion" / filename,
        ROOT_DIR / "ingestionTest" / filename,
        ROOT_DIR / "ccnews_warc_by_day" / filename,
        ROOT_DIR / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Tried: {[str(p) for p in candidates]}")


def expand_input_paths(input_spec: str) -> List[Path]:
    path = resolve_input_path(input_spec)
    if path.is_dir():
        files: List[Path] = []
        for pattern in ("*.csv", "*.jsonl", "*.json"):
            files.extend(sorted(p for p in path.glob(pattern) if p.is_file()))
        if not files:
            raise FileNotFoundError(f"No supported input files found in directory: {path}")
        return files
    return [path]


def iter_input_dataframes(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=chunk_size)
        return

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= chunk_size:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for start in range(0, len(payload), chunk_size):
                yield pd.DataFrame(payload[start : start + chunk_size])
            return
        if isinstance(payload, dict):
            yield pd.DataFrame([payload])
            return
        raise ValueError(f"Unsupported JSON structure for {path}. Expected an object or a list of objects.")

    raise ValueError(f"Unsupported input format for {path}. Expected .csv, .jsonl or .json")


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


class JsonArrayWriter:
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", encoding="utf-8")
        self._fh.write("[\n")
        self._first = True

    def write_records(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            if not self._first:
                self._fh.write(",\n")
            self._fh.write(json.dumps(rec, ensure_ascii=False, indent=2))
            self._first = False

    def close(self) -> None:
        self._fh.write("\n]\n")
        self._fh.close()


def flush_batch(
    batch: List[Dict[str, Any]],
    store: Optional[PostgresStore],
    writer: Optional[JsonArrayWriter],
) -> int:
    if not batch:
        return 0
    if store is not None:
        store.upsert_articles(batch)
    if writer is not None:
        writer.write_records(batch)
    flushed = len(batch)
    batch.clear()
    return flushed


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize datasets and upsert into PostgreSQL")
    parser.add_argument(
        "--in",
        dest="inputs",
        action="append",
        default=None,
        help="Input file or directory. Can be repeated, e.g. --in ccnews_warc_by_day/20260211json --in main/ingestiontable/dataset_top20.csv",
    )
    parser.add_argument("--csv", default="dataset_top20.csv", help="CSV dataset filename/path")
    parser.add_argument("--jsonl", default="ccnews_dataset.jsonl", help="JSONL dataset filename/path")
    parser.add_argument("--db-url", default="postgresql://postgres:postgres@localhost:5432/pfe_news")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of normalized articles to upsert/export per batch")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of raw rows to load at a time from each input file")
    parser.add_argument("--no-db", action="store_true", help="Skip PostgreSQL upsert")
    parser.add_argument(
        "--export-json",
        default=None,
        help="Optional output JSON path. If omitted, no JSON file is exported.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    input_specs = args.inputs if args.inputs else [args.csv, args.jsonl]

    input_paths: List[Path] = []
    seen: set[Path] = set()
    for spec in input_specs:
        for path in expand_input_paths(spec):
            resolved = path.resolve()
            if resolved not in seen:
                input_paths.append(path)
                seen.add(resolved)

    if not input_paths:
        raise ValueError("No input data loaded. Provide at least one supported --in path or use the default inputs.")

    store = None if args.no_db else PostgresStore(args.db_url)
    if store is not None:
        store.init_db()

    writer = JsonArrayWriter(Path(args.export_json)) if args.export_json else None
    batch: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    total_records = 0
    inserted_records = 0

    try:
        for input_path in input_paths:
            print(f"Processing input: {input_path}")
            for df in iter_input_dataframes(input_path, chunk_size=args.chunk_size):
                normalized = normalize_dataframe(df)
                for rec in to_records(normalized):
                    url_key = str(rec.get("url") or "").strip()
                    if url_key:
                        if url_key in seen_urls:
                            continue
                        seen_urls.add(url_key)
                    else:
                        rec_id = str(rec.get("id") or "").strip()
                        if rec_id in seen_ids:
                            continue
                        seen_ids.add(rec_id)

                    batch.append(rec)
                    total_records += 1

                    if len(batch) >= args.batch_size:
                        inserted_records += flush_batch(batch, store, writer)

        inserted_records += flush_batch(batch, store, writer)
    finally:
        if writer is not None:
            writer.close()

    if args.export_json:
        print(f"Export JSON: {args.export_json}")

    if store is not None:
        print(f"Upsert PostgreSQL articles: {inserted_records}")

    print(f"Fusion terminée. Nombre total d'articles: {total_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
