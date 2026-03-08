import os
import re
import gzip
import json
import zlib
import time
import argparse
from urllib.parse import urlparse, urljoin
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

try:
    import brotli
except Exception:
    brotli = None

try:
    import trafilatura
    from trafilatura.settings import DEFAULT_CONFIG
except Exception:
    trafilatura = None
    DEFAULT_CONFIG = None


BASE_DIR_DEFAULT = "ccnews_warc_by_day"
CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
MULTISPACE_RE = re.compile(r"\s+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")

SKIP_URL_PATTERNS = [
    "/tag/",
    "/tags/",
    "/topic/",
    "/topics/",
    "/author/",
    "/authors/",
    "/search",
    "?s=",
    "&s=",
    "/login",
    "/signin",
    "/signup",
    "/register",
    "/account",
    "/video/",
    "/videos/",
    "/gallery/",
    "/galleries/",
    "/slideshow/",
    "/comments/",
]

TRAFI_CONFIG = None
if DEFAULT_CONFIG is not None:
    from copy import deepcopy
    TRAFI_CONFIG = deepcopy(DEFAULT_CONFIG)
    TRAFI_CONFIG["DEFAULT"]["EXTENSIVE_DATE_SEARCH"] = "off"


# -------------------------
# Utils
# -------------------------

def safe_dt(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        s2 = s.strip().replace("Z", "+00:00")
        dt_obj = datetime.fromisoformat(s2)
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


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


def extract_charset(content_type: str) -> Optional[str]:
    if not content_type:
        return None
    m = re.search(r"charset\s*=\s*([^\s;]+)", content_type, flags=re.I)
    if not m:
        return None
    cs = m.group(1).strip().strip('"').strip("'")
    return cs or None


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = MULTISPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def log_error(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as g:
            g.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -------------------------
# HTTP decode
# -------------------------

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


# -------------------------
# HTML helpers
# -------------------------

def get_meta_content(soup: BeautifulSoup, *keys: str) -> Optional[str]:
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            val = tag.get("content", "").strip()
            if val:
                return val
    return None


def guess_date_from_html(soup: BeautifulSoup) -> Optional[str]:
    candidates = [
        get_meta_content(
            soup,
            "article:published_time",
            "article:modified_time",
            "og:updated_time",
            "pubdate",
            "publish-date",
            "date",
            "dc.date",
            "dc.date.issued",
        )
    ]

    time_tag = soup.find("time")
    if time_tag:
        candidates.append(time_tag.get("datetime"))
        candidates.append(time_tag.get_text(" ", strip=True))

    for c in candidates:
        d = safe_dt(c)
        if d:
            return d

    return None


def clean_soup_for_text(soup: BeautifulSoup) -> None:
    for tag_name in [
        "script", "style", "noscript", "template", "svg", "canvas", "iframe",
        "form", "button", "input", "select", "option", "footer", "nav", "aside"
    ]:
        for t in soup.find_all(tag_name):
            t.decompose()

    # supprimer blocs très bruités
    for sel in [
        ".advert", ".ads", ".ad", ".cookie", ".cookies", ".newsletter", ".promo",
        ".related", ".recommended", ".share", ".social", ".sidebar", ".menu",
        "#advert", "#ads", "#cookie", "#newsletter", "#sidebar", "#menu",
    ]:
        for t in soup.select(sel):
            t.decompose()


def best_text_container(soup: BeautifulSoup):
    selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article-body",
        ".article-content",
        ".post",
        ".post-content",
        ".entry-content",
        ".story-body",
        ".content",
    ]

    candidates = []
    for sel in selectors:
        try:
            candidates.extend(soup.select(sel))
        except Exception:
            pass

    body = soup.body or soup
    if body:
        candidates.append(body)

    best_node = None
    best_len = -1

    for node in candidates:
        txt = normalize_whitespace(node.get_text("\n", strip=True))
        score = len(txt)
        if score > best_len:
            best_len = score
            best_node = node

    return best_node or body or soup


# -------------------------
# Extraction rapide
# -------------------------

def extract_fast_article(
    url: str,
    warc_date: Optional[str],
    html: str,
    min_text_chars: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None, "bs4_parse_failed"

    clean_soup_for_text(soup)

    parsed = urlparse(url)
    domain = parsed.netloc or ""

    title = (
        get_meta_content(soup, "og:title", "twitter:title")
        or (soup.title.get_text(" ", strip=True) if soup.title else None)
    )
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else None

    description = get_meta_content(
        soup,
        "og:description",
        "twitter:description",
        "description",
    )

    image_url = get_meta_content(soup, "og:image", "twitter:image")
    if image_url:
        try:
            image_url = urljoin(url, image_url)
        except Exception:
            pass

    pub_date = guess_date_from_html(soup) or safe_dt(warc_date)

    node = best_text_container(soup)
    text = normalize_whitespace(node.get_text("\n", strip=True) if node else "")

    if len(text) < min_text_chars:
        return None, "fast_text_too_short"

    if not title:
        title = domain or "untitled"

    if not description and text:
        description = text[:280] + "…" if len(text) > 280 else text

    item = {
        "date": sanitize_str(pub_date),
        "description": sanitize_str(description),
        "domain": sanitize_str(domain) or "",
        "image_url": sanitize_str(image_url),
        "text": sanitize_str(text) or "",
        "title": sanitize_str(title) or "untitled",
        "url": sanitize_str(url) or "",
    }
    return item, "fast_ok"


# -------------------------
# Fallback trafilatura
# -------------------------

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


def extract_trafilatura_article(
    url: str,
    warc_date: Optional[str],
    html: str,
    min_text_chars: int,
    max_tree_size: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if trafilatura is None:
        return None, "trafilatura_unavailable"

    parsed = urlparse(url)
    domain = parsed.netloc or ""

    try:
        extracted = trafilatura.bare_extraction(
            html,
            url=url,
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=False,
            fast=True,
            max_tree_size=max_tree_size,
            config=TRAFI_CONFIG,
        )
    except Exception as e:
        return None, f"trafilatura_exception:{type(e).__name__}"

    if not extracted:
        return None, "trafilatura_none"

    text = normalize_whitespace((pick(extracted, "text", "") or "").strip())
    title = pick(extracted, "title", None)
    description = pick(extracted, "description", None)
    image_url = pick(extracted, "image", None)
    pub_date = pick(extracted, "date", None)

    if len(text) < min_text_chars:
        return None, "trafilatura_text_too_short"

    if not title:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        title = (
            get_meta_content(soup, "og:title", "twitter:title")
            or (soup.title.get_text(" ", strip=True) if soup.title else None)
            or domain
            or "untitled"
        )
        if not description:
            description = get_meta_content(
                soup, "og:description", "twitter:description", "description"
            )
        if not image_url:
            image_url = get_meta_content(soup, "og:image", "twitter:image")

    if not description and text:
        description = text[:280] + "…" if len(text) > 280 else text

    item = {
        "date": sanitize_str(safe_dt(str(pub_date)) if pub_date else safe_dt(warc_date)),
        "description": sanitize_str(description),
        "domain": sanitize_str(domain) or "",
        "image_url": sanitize_str(image_url),
        "text": sanitize_str(text) or "",
        "title": sanitize_str(title) or "untitled",
        "url": sanitize_str(url) or "",
    }
    return item, "trafilatura_ok"


# -------------------------
# Normalisation article
# -------------------------

def normalize_record(
    url: str,
    warc_date: Optional[str],
    html: str,
    min_text_chars: int,
    max_tree_size: int,
    use_trafilatura_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    item, reason = extract_fast_article(
        url=url,
        warc_date=warc_date,
        html=html,
        min_text_chars=min_text_chars,
    )
    if item is not None:
        return item, reason

    if use_trafilatura_fallback:
        return extract_trafilatura_article(
            url=url,
            warc_date=warc_date,
            html=html,
            min_text_chars=min_text_chars,
            max_tree_size=max_tree_size,
        )

    return None, reason


# -------------------------
# Filtres
# -------------------------

def should_skip_before_decode(
    url: str,
    raw: bytes,
    content_type: str,
    min_html_bytes: int,
    max_html_bytes: int,
) -> bool:
    ct = (content_type or "").lower()
    if "text/html" not in ct:
        return True

    u = (url or "").lower()
    if any(p in u for p in SKIP_URL_PATTERNS):
        return True

    size = len(raw)
    if size < min_html_bytes:
        return True
    if size > max_html_bytes:
        return True

    return False


def should_skip_after_decode(url: str, html: str) -> bool:
    if not html:
        return True

    h = html.lstrip().lower()
    if h.startswith("<?xml"):
        return True

    head = html[:3000].lower()
    noise_signals = [
        "<title>login",
        "<title>sign in",
        "<title>search",
        "<title>tag:",
        "<title>author:",
        "captcha",
        "access denied",
    ]
    if any(sig in head for sig in noise_signals):
        return True

    u = (url or "").lower()
    if any(p in u for p in SKIP_URL_PATTERNS):
        return True

    return False


# -------------------------
# WARC iteration
# -------------------------

def iter_warc_records(
    warc_gz_path: str,
    errors_path: Optional[str],
    min_html_bytes: int,
    max_html_bytes: int,
    min_text_chars: int,
    max_tree_size: int,
    use_trafilatura_fallback: bool,
    stats: Dict[str, int],
) -> Iterable[Dict[str, Any]]:
    with gzip.open(warc_gz_path, "rb") as stream:
        for rec in ArchiveIterator(stream):
            if rec.rec_type != "response":
                continue

            stats["response_records"] += 1

            url = sanitize_url(rec.rec_headers.get_header("WARC-Target-URI"))
            warc_date = rec.rec_headers.get_header("WARC-Date")

            if not url:
                stats["missing_url"] += 1
                continue

            content_type = ""
            content_encoding = ""
            if rec.http_headers:
                content_type = rec.http_headers.get_header("Content-Type") or ""
                content_encoding = rec.http_headers.get_header("Content-Encoding") or ""

            try:
                raw = rec.content_stream().read()
                stats["body_read_ok"] += 1
            except Exception:
                stats["body_read_failed"] += 1
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

            if should_skip_before_decode(url, raw, content_type, min_html_bytes, max_html_bytes):
                stats["skipped_before_decode"] += 1
                continue

            stats["passed_before_decode"] += 1

            html = decode_http_body(raw, content_encoding, content_type)
            if not html:
                stats["decode_failed"] += 1
                if errors_path:
                    log_error(errors_path, {
                        "warc_file": os.path.basename(warc_gz_path),
                        "url": url,
                        "warc_date": warc_date,
                        "content_type": content_type,
                        "content_encoding": content_encoding,
                        "note": "decode_http_body_failed",
                    })
                continue

            stats["decode_ok"] += 1

            if should_skip_after_decode(url, html):
                stats["skipped_after_decode"] += 1
                continue

            stats["passed_after_decode"] += 1

            item, reason = normalize_record(
                url=url,
                warc_date=warc_date,
                html=html,
                min_text_chars=min_text_chars,
                max_tree_size=max_tree_size,
                use_trafilatura_fallback=use_trafilatura_fallback,
            )

            if item is None:
                stats["normalize_failed"] += 1
                stats[f"reason::{reason}"] = stats.get(f"reason::{reason}", 0) + 1
                continue

            stats["normalize_ok"] += 1
            stats[f"reason::{reason}"] = stats.get(f"reason::{reason}", 0) + 1
            yield item


# -------------------------
# JSONL per WARC
# -------------------------

def warc_to_jsonl_file(
    warc_file: str,
    out_jsonl_path: str,
    errors_path: Optional[str],
    min_html_bytes: int,
    max_html_bytes: int,
    min_text_chars: int,
    max_tree_size: int,
    use_trafilatura_fallback: bool,
) -> Tuple[int, Dict[str, int]]:
    seen_urls = set()
    count = 0

    stats: Dict[str, int] = {
        "response_records": 0,
        "missing_url": 0,
        "body_read_ok": 0,
        "body_read_failed": 0,
        "skipped_before_decode": 0,
        "passed_before_decode": 0,
        "decode_failed": 0,
        "decode_ok": 0,
        "skipped_after_decode": 0,
        "passed_after_decode": 0,
        "normalize_failed": 0,
        "normalize_ok": 0,
        "dedup_skipped": 0,
        "written": 0,
    }

    os.makedirs(os.path.dirname(out_jsonl_path) or ".", exist_ok=True)

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for item in iter_warc_records(
            warc_gz_path=warc_file,
            errors_path=errors_path,
            min_html_bytes=min_html_bytes,
            max_html_bytes=max_html_bytes,
            min_text_chars=min_text_chars,
            max_tree_size=max_tree_size,
            use_trafilatura_fallback=use_trafilatura_fallback,
            stats=stats,
        ):
            u = item.get("url")
            if not u or u in seen_urls:
                stats["dedup_skipped"] += 1
                continue

            seen_urls.add(u)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["written"] += 1
            count += 1

    return count, stats


# -------------------------
# Multiprocessing worker
# -------------------------

def process_one_warc(task: Tuple[str, str, bool, int, int, int, int, bool]) -> Tuple[str, str, int, float, Dict[str, int]]:
    (
        warc_file,
        output_dir,
        skip_existing,
        min_html_bytes,
        max_html_bytes,
        min_text_chars,
        max_tree_size,
        use_trafilatura_fallback,
    ) = task

    base = os.path.basename(warc_file)
    out_name = base.replace(".warc.gz", ".jsonl")
    out_jsonl_path = os.path.join(output_dir, out_name)
    errors_path = os.path.join(output_dir, base.replace(".warc.gz", ".errors.jsonl"))

    if skip_existing and os.path.exists(out_jsonl_path) and os.path.getsize(out_jsonl_path) > 0:
        return base, "skip", 0, 0.0, {}

    t0 = time.perf_counter()

    count, stats = warc_to_jsonl_file(
        warc_file=warc_file,
        out_jsonl_path=out_jsonl_path,
        errors_path=errors_path,
        min_html_bytes=min_html_bytes,
        max_html_bytes=max_html_bytes,
        min_text_chars=min_text_chars,
        max_tree_size=max_tree_size,
        use_trafilatura_fallback=use_trafilatura_fallback,
    )

    dt_sec = time.perf_counter() - t0
    return base, "ok", count, dt_sec, stats


def build_one_jsonl_per_warc_parallel(
    input_dir: str,
    output_dir: str,
    workers: int,
    skip_existing: bool,
    min_html_bytes: int,
    max_html_bytes: int,
    min_text_chars: int,
    max_tree_size: int,
    use_trafilatura_fallback: bool,
) -> None:
    warc_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".warc.gz")
    )

    print(f"WARC files found: {len(warc_files)}")
    if not warc_files:
        raise RuntimeError(f"Aucun .warc.gz trouvé dans {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (
            wf,
            output_dir,
            skip_existing,
            min_html_bytes,
            max_html_bytes,
            min_text_chars,
            max_tree_size,
            use_trafilatura_fallback,
        )
        for wf in warc_files
    ]

    total = len(tasks)
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_warc, task) for task in tasks]

        for fut in as_completed(futures):
            done += 1
            try:
                base, status, count, dt_sec, stats = fut.result()
                if status == "skip":
                    print(f"[{done}/{total}] warc {base} skip")
                else:
                    print(f"[{done}/{total}] warc {base} ok ({count} items, {dt_sec:.1f}s)")
                    print(f"    debug: {stats}")
            except Exception as e:
                print(f"[{done}/{total}] warc failed: {e}")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Date au format YYYYMMDD, ex: 20260211")
    parser.add_argument("--base-dir", default=BASE_DIR_DEFAULT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-html-bytes", type=int, default=500)
    parser.add_argument("--max-html-bytes", type=int, default=5_000_000)
    parser.add_argument("--min-text-chars", type=int, default=120)
    parser.add_argument("--max-tree-size", type=int, default=500_000)
    parser.add_argument(
        "--use-trafilatura-fallback",
        action="store_true",
        help="Utilise trafilatura seulement si l'extraction rapide échoue",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}", args.date):
        raise ValueError("La date doit être au format YYYYMMDD")

    input_dir = os.path.join(args.base_dir, args.date)
    output_dir = os.path.join(args.base_dir, f"{args.date}json")

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Dossier introuvable: {input_dir}")

    print(f"Input dir                : {input_dir}")
    print(f"Output dir               : {output_dir}")
    print(f"Workers                  : {args.workers}")
    print(f"Min HTML bytes           : {args.min_html_bytes}")
    print(f"Max HTML bytes           : {args.max_html_bytes}")
    print(f"Min text chars           : {args.min_text_chars}")
    print(f"Max tree size            : {args.max_tree_size}")
    print(f"Use trafilatura fallback : {args.use_trafilatura_fallback}")

    build_one_jsonl_per_warc_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=args.workers,
        skip_existing=args.skip_existing,
        min_html_bytes=args.min_html_bytes,
        max_html_bytes=args.max_html_bytes,
        min_text_chars=args.min_text_chars,
        max_tree_size=args.max_tree_size,
        use_trafilatura_fallback=args.use_trafilatura_fallback,
    )

    print("Terminé.")


if __name__ == "__main__":
    main()