import os
import re
import gzip
import io
import argparse
import requests

BASE = "https://data.commoncrawl.org/"
TS_RE = re.compile(r"CC-NEWS-(\d{14})-\d{5}\.warc\.gz$")

def fetch_warc_paths(year: int, month: int) -> list[str]:
    url = f"{BASE}crawl-data/CC-NEWS/{year:04d}/{month:02d}/warc.paths.gz"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
        return [line.decode("utf-8").strip() for line in f if line.strip()]

def warc_paths_for_date(date_str: str) -> list[str]:
    if not re.fullmatch(r"\d{8}", date_str):
        raise ValueError("La date doit être au format YYYYMMDD, ex: 20260211")

    year = int(date_str[:4])
    month = int(date_str[4:6])

    paths = fetch_warc_paths(year, month)

    selected = []
    for path in paths:
        filename = os.path.basename(path)
        m = TS_RE.match(filename)
        if m and m.group(1).startswith(date_str):
            selected.append(path)

    return sorted(selected)

def download_file(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"SKIP exists: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        downloaded = 0

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

    if total:
        print(f"OK  {out_path} ({downloaded/1e6:.1f} MB / {total/1e6:.1f} MB)")
    else:
        print(f"OK  {out_path} ({downloaded/1e6:.1f} MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Date au format YYYYMMDD, ex: 20260211")
    parser.add_argument(
        "--out-dir",
        default="ccnews_warc_by_day",
        help="Dossier de sortie racine"
    )
    args = parser.parse_args()

    warc_paths = warc_paths_for_date(args.date)

    print(f"Date: {args.date}")
    print(f"Nombre de WARC trouvés: {len(warc_paths)}")

    if not warc_paths:
        print("Aucun WARC trouvé pour cette date.")
        return

    day_out_dir = os.path.join(os.getcwd(), args.out_dir, args.date)

    for path in warc_paths:
        url = BASE + path
        filename = os.path.basename(path)
        out_path = os.path.join(day_out_dir, filename)

        print("GET", url)
        download_file(url, out_path)

if __name__ == "__main__":
    main()