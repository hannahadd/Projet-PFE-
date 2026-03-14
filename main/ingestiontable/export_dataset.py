import requests
import io
import csv
import sys
import os
import argparse
import zipfile
from urllib.parse import urlparse
from newspaper import Article

# Augmenter la limite de taille des champs CSV pour éviter les erreurs avec GDELT
csv.field_size_limit(2**31 - 1)

MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# -----------------------------
# Configuration des domaines Top 20
# -----------------------------
TOP_20_DOMAINS = {
    "yahoo.com", "yahoo.co.jp", "turbopages.org", "naver.com", "qq.com",
    "msn.com", "news.yahoo.co.jp", "globo.com", "bbc.co.uk", "cnn.com",
    "news.google.com", "nytimes.com", "bbc.com", "dailymail.co.uk",
    "foxnews.com", "theguardian.com", "uol.com.br", "hindustantimes.com",
    "indiatimes.com", "infobae.com"
}

def extract_domain(url):
    """Extrait le domaine de base d'une URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return None

def is_top20_domain(domain):
    """Vérifie si le domaine appartient à la liste définie."""
    if not domain: return False
    for top_domain in TOP_20_DOMAINS:
        if domain == top_domain or domain.endswith("." + top_domain):
            return True
    return False

def clean_themes(theme_string):
    """Nettoie les thèmes GDELT pour ne garder que les étiquettes principales."""
    if not theme_string: return []
    themes = theme_string.split(";")
    return list(set(t.split("_")[0] for t in themes))

# -----------------------------
# Extraction et Parsing de contenu
# -----------------------------
def extract_article_text(url):
    """Télécharge et parse l'article pour extraire titre et texte."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {"title": article.title, "text": article.text}
    except Exception:
        return {"title": None, "text": None}

def download_and_parse_gkg_url(gkg_url, max_rows=None):
    """Télécharge un ZIP GKG et extrait les articles correspondant au Top 20."""
    try:
        resp = requests.get(gkg_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Erreur téléchargement {gkg_url}: {e}")
        return []

    zipped = io.BytesIO(resp.content)
    articles = []

    try:
        with zipfile.ZipFile(zipped) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                reader = csv.reader(
                    io.TextIOWrapper(f, encoding="utf-8", errors="ignore"),
                    delimiter="\t"
                )
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows: break
                    if len(row) < 8: continue

                    url = row[4]
                    domain = extract_domain(url)
                    if not is_top20_domain(domain): continue

                    content = extract_article_text(url)
                    articles.append({
                        "date": row[1],
                        "source": row[3],
                        "domain": domain,
                        "url": url,
                        "themes": clean_themes(row[7]),
                        "title": content["title"],
                        "text": content["text"]
                    })
    except Exception as e:
        print(f"Erreur ZIP {gkg_url}: {e}")
    return articles

def get_week_gkg_urls(start_date, end_date):
    """Récupère la liste des fichiers GKG sur GDELT pour l'intervalle donné."""
    resp = requests.get(MASTER_URL)
    resp.raise_for_status()
    urls = []
    for line in resp.text.strip().split("\n"):
        if ".gkg.csv.zip" in line:
            url = line.split()[-1]
            date_part = url.split("/")[-1][:8]
            if start_date <= date_part <= end_date:
                urls.append(url)
    return urls

def export_to_csv(articles, filename):
    """Exporte les articles collectés dans un fichier CSV."""
    fieldnames = ["date", "source", "domain", "url", "title", "text", "themes"]
    
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            if not article["text"]: continue
            writer.writerow({
                "date": article["date"],
                "source": article["source"],
                "domain": article["domain"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"],
                "themes": ";".join(article["themes"])
            })
    print(f"\n[OK] Dataset exporté dans : {filename}")

# -----------------------------
# MAIN avec gestion des arguments
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'export GDELT Top 20 News")
    
    # Arguments obligatoires
    parser.add_argument("--start_date", required=True, help="Date de début (YYYYMMDD)")
    parser.add_argument("--end_date", required=True, help="Date de fin (YYYYMMDD)")
    
    # Chemin forcé par défaut
    DEFAULT_PATH = "/home/pfe/Documents/PFE/main/ingestiontable/dataset_top20.csv"
    parser.add_argument("--output", default=DEFAULT_PATH, help="Chemin complet du fichier de sortie")
    parser.add_argument("--limit_per_file", type=int, default=20, help="Max de lignes à parser par fichier ZIP")

    args = parser.parse_args()

    # Création du dossier parent s'il n'existe pas
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"--- Ingestion GDELT du {args.start_date} au {args.end_date} ---")
    
    gkg_urls = get_week_gkg_urls(args.start_date, args.end_date)
    print(f"Fichiers trouvés : {len(gkg_urls)}")

    all_articles = []
    success = 0
    fail = 0

    for idx, url in enumerate(gkg_urls):
        print(f"[{idx+1}/{len(gkg_urls)}] Traitement de : {url.split('/')[-1]}")
        articles = download_and_parse_gkg_url(url, max_rows=args.limit_per_file)

        for art in articles:
            if art["text"]:
                success += 1
            else:
                fail += 1
        all_articles.extend(articles)

    print("\n" + "="*20)
    print(f"Articles réussis : {success}")
    print(f"Articles échoués : {fail}")
    print(f"Total conservés  : {len(all_articles)}")
    print("="*20)

    export_to_csv(all_articles, args.output)