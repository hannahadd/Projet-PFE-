import hashlib
from pathlib import Path

import pandas as pd
import json


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_input_path(filename: str) -> Path:
    """Resolve input file path regardless of current working directory."""
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        ROOT_DIR / filename,
        ROOT_DIR / "ingestion" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}. Tried: {[str(p) for p in candidates]}")


def stable_article_id(url: str, title: str, content: str) -> str:
    """Deterministic id (uuid-like) based on URL when available."""
    key = (url or "").strip()
    if not key:
        key = ((title or "") + "||" + (content or "")[:400]).strip()
    hex_id = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()
    return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"


# ==============================
# DATE PARSING ROBUSTE
# ==============================
def parse_date_column(series):

    if series.dtype in ["int64", "float64"]:
        series = series.astype(str)

    parsed = pd.to_datetime(
        series,
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            series.loc[mask],
            errors="coerce"
        )

    return parsed


# ==============================
# NORMALISATION GENERIQUE
# ==============================
def normalize_dataframe(df):

    df = df.copy()

    # TITLE
    if "headline" in df.columns:
        df["title"] = df["headline"]
    elif "title" not in df.columns:
        df["title"] = None

    # CONTENT priorité stricte
    if "text" in df.columns:
        df["content"] = df["text"]
    elif "article" in df.columns:
        df["content"] = df["article"]
    elif "description" in df.columns:
        df["content"] = df["description"]
    else:
        df["content"] = None

    # DATE
    if "date" in df.columns:
        df["published_date"] = parse_date_column(df["date"])
    elif "published_date" in df.columns:
        df["published_date"] = parse_date_column(df["published_date"])
    elif "published" in df.columns:
        df["published_date"] = parse_date_column(df["published"])
    else:
        df["published_date"] = pd.NaT

    # SOURCE
    if "source_name" in df.columns:
        df["source"] = df["source_name"]
    elif "domain" in df.columns:
        df["source"] = df["domain"]
    elif "source" not in df.columns:
        df["source"] = None

    # URL
    if "link" in df.columns:
        df["url"] = df["link"]
    elif "url" not in df.columns:
        df["url"] = None

    df_final = df[[
        "title",
        "content",
        "published_date",
        "source",
        "url"
    ]].copy()

    # Nettoyage
    df_final["content"] = df_final["content"].astype(str).str.strip()
    df_final["title"] = df_final["title"].astype(str).str.strip()

    df_final = df_final[
        (df_final["content"] != "") &
        (df_final["title"] != "")
    ]

    return df_final


# ==============================
# CHARGEMENT DATASET 1 (CSV)
# ==============================
csv_path = resolve_input_path("dataset_top20.csv")
df_csv = pd.read_csv(csv_path)

df_csv_normalized = normalize_dataframe(df_csv)


# ==============================
# CHARGEMENT DATASET 2 (JSONL)
# ==============================
jsonl_path = resolve_input_path("ccnews_dataset.jsonl")
df_jsonl = pd.read_json(jsonl_path, lines=True, encoding="utf-8")

df_jsonl_normalized = normalize_dataframe(df_jsonl)


# ==============================
# FUSION
# ==============================
merged_df = pd.concat(
    [df_csv_normalized, df_jsonl_normalized],
    ignore_index=True
)

# Suppression doublons sur URL (clé fiable)
merged_df = merged_df.drop_duplicates(subset=["url"])


# ==============================
# EXPORT JSON FINAL
# ==============================
merged_df["published_date"] = merged_df["published_date"].astype(str)

# Stable unique id for each article
merged_df["id"] = merged_df.apply(
    lambda r: stable_article_id(
        str(r.get("url") or ""),
        str(r.get("title") or ""),
        str(r.get("content") or ""),
    ),
    axis=1,
)

# Reorder columns (keep compatibility)
cols = ["id", "title", "content", "published_date", "source", "url"]
merged_df = merged_df[cols]

out_root = ROOT_DIR / "merged_dataset.json"
out_src = ROOT_DIR / "src" / "merged_dataset.json"

merged_df.to_json(out_root, orient="records", force_ascii=False, indent=2)
out_src.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_json(out_src, orient="records", force_ascii=False, indent=2)

print("Fusion terminée.")
print("Nombre total d’articles :", len(merged_df))
print("Export:")
print(" -", out_root)
print(" -", out_src)