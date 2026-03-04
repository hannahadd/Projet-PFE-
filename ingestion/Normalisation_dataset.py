import pandas as pd
import json


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
df_csv = pd.read_csv("dataset_top20.csv")

df_csv_normalized = normalize_dataframe(df_csv)


# ==============================
# CHARGEMENT DATASET 2 (JSONL)
# ==============================
df_jsonl = pd.read_json(
    "ccnews_dataset.jsonl",
    lines=True,
    encoding="utf-8"
)

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

merged_df.to_json(
    "merged_dataset.json",
    orient="records",
    force_ascii=False,
    indent=2
)

print("Fusion terminée.")
print("Nombre total d’articles :", len(merged_df))