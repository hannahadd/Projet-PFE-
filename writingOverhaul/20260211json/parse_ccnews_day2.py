import os
import re
import gzip
import json
import zlib
import argparse
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, List

from warcio.archiveiterator import ArchiveIterator
import trafilatura
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

try:
    import brotli
except Exception:
    brotli = None


# -------------------------
# Dates / extraction
# -------------------------

def safe_dt(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        s2 = s.strip().replace("Z", "")
        dt = datetime.fromisoformat(s2)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def pick(extracted: Any, key: str, default=None):
    if extracted is None:
        return default

    if isinstance(extracted, dict):
        return extracted.get(key, default)

    if hasattr(extracted, "as_dict"):
        try:
            d = extracted.as_dict()
            if isinstance(d, dict):
                return d.get(key, default)
        except Exception:
            pass

    return getattr(extracted, key, default)


def normalize_date_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if hasattr(v, "strftime"):
        try:
            return v.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    if isinstance(v, str):
        return safe_dt(v)
    return None


# -------------------------
# Décodage HTTP robuste
# -------------------------

CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def extract_charset(content_type: str) -> Optional[str]:
    if not content_type:
        return None
    m = re.search(r"charset\s*=\s*([^\s;]+)", content_type, flags=re.I)
    if not m:
        return None
    cs = m.group(1).strip().strip('"').strip("'")
    return cs or None


def decode_http_body(raw: bytes, content_encoding: str, content_type: str) -> Optional[str]:
    if not raw:
        return None

    raw = raw.replace(b"\x00", b"")

    enc = (content_encoding or "").lower()
    try:
        if "br" in enc:
            if brotli is None:
                return None
            raw = brotli.decompress(raw)
        elif "gzip" in enc:
            raw = gzip.decompress(raw)
        elif "deflate" in enc:
            raw = zlib.decompress(raw)
    except Exception:
        return None

    charset = extract_charset(content_type or "")
    text: Optional[str] = None

    if charset:
        try:
            text = raw.decode(charset, errors="replace")
        except Exception:
            text = None

    if not text:
        try:
            best = from_bytes(raw).best()
            text = str(best) if best else raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    text = CTRL_CHARS_RE.sub("", text)
    return text


def sanitize_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        return s.encode("utf-8", "replace").decode("utf-8", "replace")
    except Exception:
        return None


def sanitize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.replace("\x00", "").strip()
    return u or None


# -------------------------
# Fallback meta tags
# -------------------------

def meta_fallback(html: str) -> Dict[str, Optional[str]]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    def get_meta(*keys: str) -> Optional[str]:
        for k in keys:
            tag = soup.find("meta", attrs={"property": k}) or soup.find("meta", attrs={"name": k})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = get_meta("og:title", "twitter:title")
    desc = get_meta("og:description", "twitter:description", "description")
    img = get_meta("og:image", "twitter:image")

    if not title:
        t = soup.find("title")
        title = t.get_text(strip=True) if t else None

    return {"title": title, "description": desc, "image_url": img}


# -------------------------
# Normalisation d'un record
# -------------------------

def normalize_record(url: str, warc_date: Optional[str], html: str) -> Optional[Dict[str, Any]]:
    url = sanitize_url(url) or ""
    parsed = urlparse(url)
    domain = parsed.netloc or ""

    try:
        extracted = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception:
        return None

    if not extracted:
        return None

    text = (pick(extracted, "text", "") or "").strip()
    title = pick(extracted, "title", None)
    description = pick(extracted, "description", None)
    image_url = pick(extracted, "image", None)
    pub_date = pick(extracted, "date", None)

    date_norm = normalize_date_value(pub_date) or safe_dt(warc_date)

    if not title or not description or not image_url:
        fb = meta_fallback(html)
        title = title or fb["title"]
        description = description or fb["description"]
        image_url = image_url or fb["image_url"]

    if not description and text:
        description = (text[:280] + "…") if len(text) > 280 else text

    if len(text) < 200:
        return None
    if not title:
        return None

    text = sanitize_str(text) or ""
    title = sanitize_str(title) or ""
    description = sanitize_str(description)
    image_url = sanitize_str(image_url)
    domain = sanitize_str(domain) or ""
    url = sanitize_str(url) or ""

    return {
        "date": date_norm,
        "description": description,
        "domain": domain,
        "image_url": image_url,
        "text": text,
        "title": title,
        "url": url,
    }


# -------------------------
# Logging erreurs
# -------------------------

def log_error(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as g:
            g.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -------------------------
# Lecture WARC
# -------------------------

def iter_warc_records(warc_gz_path: str, errors_path: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    with gzip.open(warc_gz_path, "rb") as stream:
        for rec in ArchiveIterator(stream):
            if rec.rec_type != "response":
                continue

            url = rec.rec_headers.get_header("WARC-Target-URI")
            warc_date = rec.rec_headers.get_header("WARC-Date")

            url = sanitize_url(url)
            if not url:
                continue

            content_type = ""
            content_encoding = ""
            if rec.http_headers:
                content_type = rec.http_headers.get_header("Content-Type") or ""
                content_encoding = rec.http_headers.get_header("Content-Encoding") or ""

            if "text/html" not in (content_type or "").lower():
                continue

            try:
                raw = rec.content_stream().read()
            except Exception:
                if errors_path:
                    log_error(errors_path, {
                        "warc_file": os.path.basename(warc_gz_path),
                        "url": url,
                        "warc_date": warc_date,
                        "content_type": content_type,
                        "content_encoding": content_encoding,
                        "note": "read_body_failed",
                    })
                continue

            html = decode_http_body(raw, content_encoding, content_type)
            if not html:
                if errors_path:
                    log_error(errors_path, {
                        "warc_file": os.path.basename(warc_gz_path),
                        "url": url,
                        "warc_date": warc_date,
                        "content_type": content_type,
                        "content_encoding": content_encoding,
                        "note": "decode_http_body_failed",
                        "body_head_hex": raw[:200].hex() if raw else None,
                    })
                continue

            if html.lstrip().startswith("<?xml"):
                continue

            item = normalize_record(url, warc_date, html)
            if item:
                yield item


# -------------------------
# 1 JSON par WARC
# -------------------------

def warc_to_json_file(
    warc_file: str,
    out_json_path: str,
    errors_path: Optional[str] = None
) -> int:
    items = []
    seen_urls = set()

    for item in iter_warc_records(warc_file, errors_path=errors_path):
        u = item.get("url")
        if not u or u in seen_urls:
            continue
        seen_urls.add(u)
        items.append(item)

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return len(items)


def build_one_json_per_warc(
    input_dir: str,
    output_dir: str,
    errors_path: Optional[str] = None,
    skip_existing: bool = False
) -> None:
    warc_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".warc.gz")
    )

    print(f"WARC files found: {len(warc_files)}")

    if not warc_files:
        raise RuntimeError(f"Aucun .warc.gz trouvé dans {input_dir}")

    for wf in warc_files:
        base = os.path.basename(wf)
        json_name = base.replace(".warc.gz", ".json")
        out_json_path = os.path.join(output_dir, json_name)

        if skip_existing and os.path.exists(out_json_path) and os.path.getsize(out_json_path) > 0:
            print(f"warc {base} skip")
            continue

        try:
            count = warc_to_json_file(wf, out_json_path, errors_path=errors_path)
            print(f"warc {base} ok ({count} items)")
        except Exception as e:
            print(f"warc {base} failed: {e}")
            if errors_path:
                log_error(errors_path, {
                    "warc_file": base,
                    "note": "warc_processing_failed",
                    "error": str(e),
                })


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Date au format YYYYMMDD, ex: 20260211")
    parser.add_argument(
        "--base-dir",
        default="ccnews_warc_by_day",
        help="Dossier racine contenant les sous-dossiers par date"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip les JSON déjà présents"
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}", args.date):
        raise ValueError("La date doit être au format YYYYMMDD")

    input_dir = os.path.join(args.base_dir, args.date)
    output_dir = os.path.join(args.base_dir, f"{args.date}json")
    errors_path = os.path.join(output_dir, "errors.jsonl")

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Dossier introuvable: {input_dir}")

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    build_one_json_per_warc(
        input_dir=input_dir,
        output_dir=output_dir,
        errors_path=errors_path,
        skip_existing=args.skip_existing,
    )

    print("Terminé.")


if __name__ == "__main__":
    main()