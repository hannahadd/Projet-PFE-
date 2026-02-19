import pandas as pd
import json
from pathlib import Path

# ==========================
# CONFIGURATION
# ==========================

JSONL_PATH = "ccnews_dataset.jsonl"
CSV_PATH = "dataset_top20.csv"
OUTPUT_PATH = "merged_dataset.json"

# Colonnes finales souhaitées
FINAL_COLUMNS = [
    "title",
    "content",
    "published_date",
    "source",
    "url"
]

# ==========================
# FONCTIONS
# ==========================

def normalize_text(x):
    if x is None:
        return None

    # Si c’est déjà une string
    if isinstance(x, str):
        return x.strip()

    # Si c’est une liste
    if isinstance(x, list):
        return " ".join([str(i) for i in x])

    # Si c’est un dict
    if isinstance(x, dict):
        return " ".join([str(v) for v in x.values()])

    # Si c’est un NaN pandas
    if pd.isna(x):
        return None

    # Sinon on force en string
    return str(x).strip()



def load_jsonl(path):
    df = pd.read_json(path, lines=True)
    return df


def load_csv(path):
    df = pd.read_csv(path)
    return df


def normalize_dataframe(df):

    # Standardiser noms
    df.columns = [c.lower().strip() for c in df.columns]

    column_mapping = {
        "headline": "title",
        "article": "content",
        "text": "content",
        "description": "content",
        "date": "published_date",
        "published": "published_date",
        "publication_date": "published_date",
        "source_name": "source",
        "link": "url",
    }

    df = df.rename(columns=column_mapping)

    # 🔥 Supprimer colonnes dupliquées (garde la première)
    df = df.loc[:, ~df.columns.duplicated()]

    # Ajouter colonnes manquantes
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Ne garder que colonnes utiles
    df = df[FINAL_COLUMNS]

    # Forcer type string propre
    for col in ["title", "content", "source", "url"]:
        df[col] = df[col].astype(str).str.strip()

    # Conversion date
    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce"
    )

    # Supprimer contenu vide
    df = df[df["content"].notna()]
    df = df[df["content"] != "None"]
    df = df[df["content"] != ""]

    return df


# ==========================
# PIPELINE PRINCIPAL
# ==========================

def main():
    print("Chargement des fichiers...")
    
    df_jsonl = load_jsonl(JSONL_PATH)
    df_csv = load_csv(CSV_PATH)

    print("Normalisation...")
    
    df_jsonl = normalize_dataframe(df_jsonl)
    df_csv = normalize_dataframe(df_csv)

    print("Fusion...")
    
    merged_df = pd.concat([df_jsonl, df_csv], ignore_index=True)

    print("Suppression doublons...")
    
    merged_df = merged_df.drop_duplicates(subset=["title", "content"])

    print("Tri par date...")
    
    merged_df = merged_df.sort_values(by="published_date", ascending=False)

    print("Export JSON...")
    
    merged_df.to_json(
        OUTPUT_PATH,
        orient="records",
        date_format="iso",
        force_ascii=False,
        indent=2
    )

    print(f"Dataset fusionné sauvegardé : {OUTPUT_PATH}")
    print(f"Nombre final d'articles : {len(merged_df)}")


if __name__ == "__main__":
    main()