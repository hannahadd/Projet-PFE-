import requests
import io
import csv
import sys
from urllib.parse import urlparse
from newspaper import Article
import zipfile

# Augmenter la limite de taille des champs CSV (fix GDELT)
csv.field_size_limit(2**31 - 1)

MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# -----------------------------
# Top 20 sites de news influents / les plus visités
# -----------------------------
TOP_20_DOMAINS = {
    "yahoo.com",
    "yahoo.co.jp",
    "turbopages.org",
    "naver.com",
    "qq.com",
    "msn.com",
    "news.yahoo.co.jp",
    "globo.com",
    "bbc.co.uk",
    "cnn.com",
    "news.google.com",
    "nytimes.com",
    "bbc.com",
    "dailymail.co.uk",
    "foxnews.com",
    "theguardian.com",
    "uol.com.br",
    "hindustantimes.com",
    "indiatimes.com",
    "infobae.com"
}

# -----------------------------
# Extraction du domaine
# -----------------------------
def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except:
        return None


# -----------------------------
# Vérifier si domaine appartient au Top 20
# -----------------------------
def is_top20_domain(domain):
    if not domain:
        return False

    for top_domain in TOP_20_DOMAINS:
        if domain == top_domain or domain.endswith("." + top_domain):
            return True

    return False


# -----------------------------
# Nettoyage des thèmes GDELT
# -----------------------------
def clean_themes(theme_string):
    if not theme_string:
        return []
    themes = theme_string.split(";")
    return list(set(t.split("_")[0] for t in themes))


# -----------------------------
# Extraction texte article
# -----------------------------
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "text": article.text
        }
    except Exception:
        return {
            "title": None,
            "text": None
        }


# -----------------------------
# Téléchargement et parsing GKG
# -----------------------------
def download_and_parse_gkg_url(gkg_url, max_rows=None):
    try:
        resp = requests.get(gkg_url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Fichier introuvable (404) ignoré : {gkg_url}")
        return []
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
                    if max_rows and i >= max_rows:
                        break

                    if len(row) < 8:
                        continue

                    url = row[4]
                    domain = extract_domain(url)

                    if not is_top20_domain(domain):
                        continue

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
        print(f"Erreur lecture ZIP {gkg_url}: {e}")
        return []

    return articles



# -----------------------------
# Récupération URLs semaine
# -----------------------------
def get_week_gkg_urls(start_date, end_date):
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


# -----------------------------
# Export CSV
# -----------------------------
def export_to_csv(articles, filename="dataset_top20.csv"):
    fieldnames = ["date", "source", "domain", "url", "title", "text", "themes"]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for article in articles:
            if not article["text"]:
                continue

            writer.writerow({
                "date": article["date"],
                "source": article["source"],
                "domain": article["domain"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"],
                "themes": ";".join(article["themes"])
            })

    print(f"Dataset exporté dans {filename}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    start_date = "20260113"
    end_date = "20260201"

    print("Recherche des fichiers GKG...")
    gkg_urls = get_week_gkg_urls(start_date, end_date)

    print(f"{len(gkg_urls)} fichiers trouvés")

    all_articles = []
    success = 0
    fail = 0

    for url in gkg_urls:
        print("Processing:", url)
        articles = download_and_parse_gkg_url(url, max_rows=20)

        for art in articles:
            if art["text"]:
                success += 1
            else:
                fail += 1

        all_articles.extend(articles)

    print("\n===== RÉSULTATS =====")
    print(f"Articles récupérés avec succès : {success}")
    print(f"Articles échoués / bloqués : {fail}")
    print(f"Total articles conservés (Top 20 uniquement) : {len(all_articles)}")

    export_to_csv(all_articles)