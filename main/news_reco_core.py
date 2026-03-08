import os
import re
import json
import hashlib
import pickle
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta

import numpy as np
from dateutil.parser import isoparse
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rank_bm25 import BM25Okapi

# Optional language detection
try:
    from langdetect import detect as _lang_detect
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

# ----------------------------
# Utils
# ----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_date(d: Any) -> Optional[datetime]:
    if not d:
        return None
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc)
    try:
        return isoparse(str(d)).astimezone(timezone.utc)
    except Exception:
        return None

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def simple_tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", (s or "").lower())

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def article_fingerprint(title: str, text: str, url: str = "", n_chars: int = 400) -> str:
    u = norm_text(url)
    t = norm_text(title)
    lead = norm_text((text or "")[:n_chars])
    raw = (u + "||" + t + "||" + lead).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()

def article_id_from_url(url: str) -> str:
    hex_id = hashlib.md5((url or "").encode("utf-8", errors="ignore")).hexdigest()
    return f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"


def detect_lang(text: str) -> Optional[str]:
    if not HAS_LANGDETECT:
        return None
    t = (text or "").strip()
    if len(t) < 20:
        return None
    try:
        return _lang_detect(t)
    except Exception:
        return None


CACHE_VERSION = 2


def default_cache_dir() -> Path:
    path = Path(__file__).resolve().parent / ".cache" / "news_reco"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pickle_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def save_pickle_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def articles_signature(articles: List["Article"]) -> str:
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|n={len(articles)}".encode("utf-8"))
    for a in articles:
        h.update(str(a.id).encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(str(a.fingerprint or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def namespace_hash(value: str) -> str:
    return hashlib.sha1((value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Article:
    id: str
    title: str
    description: str
    text: str
    url: str
    domain: str
    date: Optional[datetime]
    lang: Optional[str] = None
    canonical_text: str = ""
    fingerprint: str = ""
    embedding: Optional[np.ndarray] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Article":
        title = d.get("title") or ""
        desc = d.get("description") or d.get("summary") or ""
        text = d.get("text") or d.get("content") or d.get("article") or ""
        url = d.get("url") or ""
        domain = d.get("domain") or ""
        if not domain and url:
            domain = urlparse(url).netloc
        if not domain:
            domain = d.get("source") or ""

        dt = parse_date(
            d.get("date")
            or d.get("published_at")
            or d.get("published_date")
            or d.get("datetime")
        )
        if d.get("id"):
            aid = d.get("id")
        else:
            aid = article_id_from_url(url)
            if not aid:
                hex_id = hashlib.md5((title + desc).encode()).hexdigest()
                aid = f"{hex_id[:8]}-{hex_id[8:12]}-{hex_id[12:16]}-{hex_id[16:20]}-{hex_id[20:]}"

        lead = (text or "")[:1200]
        canonical = f"{title}\n{desc}\n{lead}".strip()

        fp = d.get("fingerprint") or article_fingerprint(title, text, url=url)

        lang = d.get("lang")
        if not lang:
            lang = detect_lang(f"{title}. {desc}. {text[:500]}")

        return Article(
            id=str(aid),
            title=title,
            description=desc,
            text=text,
            url=url,
            domain=domain,
            date=dt,
            lang=lang,
            canonical_text=canonical,
            fingerprint=fp,
        )

    @staticmethod
    def from_qdrant_payload(payload: Dict[str, Any], article_id: Optional[str] = None) -> "Article":
        data = dict(payload or {})
        if article_id:
            data["id"] = article_id
        elif data.get("article_id") and not data.get("id"):
            data["id"] = data.get("article_id")
        if data.get("text") and not data.get("content"):
            data["content"] = data.get("text")
        if data.get("date") and not data.get("published_date"):
            data["published_date"] = data.get("date")
        return Article.from_dict(data)


# ----------------------------
# User profile (multi-interests)
# ----------------------------
@dataclass
class ProfileCenter:
    vec: np.ndarray
    weight: float = 1.0
    updated_at: datetime = field(default_factory=now_utc)

@dataclass
class UserProfile:
    interests_text: List[str]
    interests_tags: List[str] = field(default_factory=list)
    centers: List[ProfileCenter] = field(default_factory=list)

    def build_from_onboarding(self, embed_fn, k_max: int = 5):
        """
        Create initial centers from onboarding phrases (1 center per phrase, up to k_max)
        """
        phrases = [p.strip() for p in self.interests_text if p and p.strip()]
        phrases = phrases[:k_max]
        if not phrases:
            phrases = ["general news"]  # fallback
        vecs = embed_fn(phrases)  # (n, dim)
        self.centers = [ProfileCenter(vec=v, weight=1.0) for v in vecs]

    def update_with_article(self, article_vec: np.ndarray, signal: float = 1.0, alpha: float = 0.90):
        """
        Online update: assign to nearest center then EMA update.
        signal can be: click=0.2, like=1.0, long_read=0.6, etc.
        """
        if not self.centers:
            self.centers = [ProfileCenter(vec=article_vec.copy(), weight=signal)]
            return

        sims = [cosine(c.vec, article_vec) for c in self.centers]
        j = int(np.argmax(sims))
        c = self.centers[j]

        # EMA update
        beta = 1.0 - (1.0 - alpha) * max(0.1, min(1.0, signal))
        c.vec = beta * c.vec + (1.0 - beta) * article_vec
        c.weight = 0.95 * c.weight + 0.05 * signal
        c.updated_at = now_utc()

    def query_text(self) -> str:
        """
        Used for BM25 + optional rerank prompt.
        """
        t = " ; ".join(self.interests_text or [])
        if self.interests_tags:
            t += " ; tags: " + ", ".join(self.interests_tags)
        return t.strip()


@dataclass
class QuerySpec:
    text: str
    weight: float
    kind: str
    vec: Optional[np.ndarray] = None


# ----------------------------
# Embedding (BGE-M3 dense)
# ----------------------------
class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        device = None
        try:
            import torch

            # Prefer CUDA on Linux/Windows when available.
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                device = "cuda"
            # Apple Silicon accelerator (macOS).
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = None

        # fp16 only makes sense on accelerators
        use_fp16 = bool(use_fp16 and device in {"cuda", "mps"})

        self.backend = "flagembedding"
        self.model = None

        # Try FlagEmbedding (BGE-M3)
        try:
            from FlagEmbedding.inference.embedder import BGEM3FlagModel

            if device:
                self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
            else:
                self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
            self.backend = "flagembedding"
        except Exception:
            # Fallback to sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer

                st_model = "BAAI/bge-m3"
                if model_name and model_name.strip():
                    st_model = model_name
                if device:
                    self.model = SentenceTransformer(st_model, device=device)
                else:
                    self.model = SentenceTransformer(st_model)
                self.backend = "sentence-transformers"
            except Exception as e:
                raise RuntimeError("No embedding backend available. Install FlagEmbedding or sentence-transformers.") from e

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if self.backend == "flagembedding":
            out = self.model.encode(texts, batch_size=batch_size, max_length=4096)
            dense = out["dense_vecs"]
        else:
            dense = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        dense = np.asarray(dense, dtype=np.float32)
        # normalize for cosine
        dense /= (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
        return dense


def merge_query_specs(query_specs: List[QuerySpec]) -> List[QuerySpec]:
    merged: Dict[str, QuerySpec] = {}
    order: List[str] = []

    for spec in query_specs:
        text = str(spec.text or "").strip()
        if not text:
            continue
        key = norm_text(text)
        weight = 1.0 if spec.kind == "anchor" else float(spec.weight)

        if key not in merged:
            merged[key] = QuerySpec(text=text, weight=weight, kind=spec.kind, vec=spec.vec)
            order.append(key)
            continue

        current = merged[key]
        if spec.kind == "anchor" and current.kind != "anchor":
            current.kind = "anchor"
            current.weight = 1.0
            current.text = text
            current.vec = spec.vec
            continue

        if weight > current.weight:
            current.weight = weight
            current.text = text
            if spec.vec is not None:
                current.vec = spec.vec

    return [merged[key] for key in order]


def build_interest_query_specs(
    interest: str,
    expansions_map: Dict[str, List[str]],
    embedder: "Embedder",
    max_expansions: int = 6,
) -> List[QuerySpec]:
    anchor = str(interest or "").strip()
    if not anchor:
        return []

    texts: List[str] = [anchor]
    seen = {norm_text(anchor)}
    expansions = expansions_map.get(anchor, []) or []

    for item in expansions:
        text = str(item or "").strip()
        key = norm_text(text)
        if not text or key in seen:
            continue
        texts.append(text)
        seen.add(key)
        if len(texts) >= 1 + max(0, int(max_expansions)):
            break

    vecs = embedder.encode(texts, batch_size=min(16, max(1, len(texts))))
    anchor_vec = np.asarray(vecs[0], dtype=np.float32)

    query_specs: List[QuerySpec] = [
        QuerySpec(text=anchor, weight=1.0, kind="anchor", vec=anchor_vec)
    ]

    for text, vec in zip(texts[1:], vecs[1:]):
        exp_vec = np.asarray(vec, dtype=np.float32)
        try:
            cosine_sim = float(np.dot(anchor_vec, exp_vec))
            if not np.isfinite(cosine_sim):
                raise ValueError("non-finite cosine")
            weight = clamp(cosine_sim, 0.15, 0.45)
        except Exception:
            weight = 0.30

        query_specs.append(
            QuerySpec(
                text=text,
                weight=min(0.999999, float(weight)),
                kind="expansion",
                vec=exp_vec,
            )
        )

    return query_specs


def build_tag_query_specs(tags: List[str], embedder: "Embedder", weight: float = 0.20) -> List[QuerySpec]:
    texts: List[str] = []
    seen: Set[str] = set()
    for tag in tags:
        text = str(tag or "").strip()
        key = norm_text(text)
        if not text or key in seen:
            continue
        texts.append(text)
        seen.add(key)

    if not texts:
        return []

    vecs = embedder.encode(texts, batch_size=min(16, max(1, len(texts))))
    tag_weight = clamp(float(weight), 0.15, 0.45)
    return [
        QuerySpec(text=text, weight=tag_weight, kind="expansion", vec=np.asarray(vec, dtype=np.float32))
        for text, vec in zip(texts, vecs)
    ]


# ----------------------------
# Qdrant index (dense)
# ----------------------------
class DenseIndexQdrant:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "news_dense"):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection))
        except Exception:
            return False

    def get_points_count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception:
            return 0

    def get_existing_ids(self, batch_size: int = 2048) -> Set[str]:
        ids: Set[str] = set()
        offset = None

        while True:
            resp = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            if isinstance(resp, tuple):
                records, next_offset = resp
            else:
                records = list(getattr(resp, "points", []) or [])
                next_offset = getattr(resp, "next_page_offset", None)

            if not records:
                break

            for r in records:
                rid = getattr(r, "id", None)
                if rid is not None:
                    ids.add(str(rid))

            if next_offset is None:
                break
            offset = next_offset

        return ids

    def _build_date_filter(self, days: int) -> Optional[qm.Filter]:
        if not days or days <= 0:
            return None
        gte = (now_utc() - timedelta(days=days)).isoformat()

        if hasattr(qm, "DatetimeRange"):
            return qm.Filter(
                must=[
                    qm.FieldCondition(key="date", range=qm.DatetimeRange(gte=gte))
                ]
            )

        try:
            return qm.Filter(
                must=[
                    qm.FieldCondition(key="date", range=qm.Range(gte=gte))
                ]
            )
        except Exception:
            return None

    def ensure_collection(self, vector_dim: int):
        if self.collection_exists():
            return
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=vector_dim,
                    distance=qm.Distance.COSINE,
                ),
            )
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=vector_dim,
                    distance=qm.Distance.COSINE,
                ),
            )

    def recreate(self, vector_dim: int):
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=vector_dim,
                distance=qm.Distance.COSINE
            )
        )

    def upsert_articles(self, articles: List[Article], batch_size: int = 256):
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            points = []
            for a in batch:
                assert a.embedding is not None
                payload = {
                    "article_id": a.id,
                    "title": a.title,
                    "description": a.description,
                    "text": a.text,
                    "canonical_text": a.canonical_text,
                    "url": a.url,
                    "domain": a.domain,
                    "date": a.date.isoformat() if a.date else None,
                    "lang": a.lang,
                    "fingerprint": a.fingerprint,
                }
                points.append(qm.PointStruct(
                    id=a.id,
                    vector=a.embedding.tolist(),
                    payload=payload
                ))
            self.client.upsert(collection_name=self.collection, points=points)

    def retrieve_vectors(self, ids: List[str], batch_size: int = 256) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=batch_ids,
                with_payload=False,
                with_vectors=True,
            )
            for r in records:
                vec = getattr(r, "vector", None)
                if vec is None:
                    continue
                out[str(r.id)] = np.asarray(vec, dtype=np.float32)
        return out

    def search(self, query_vec: np.ndarray, limit: int = 200, days: int = 14) -> List[qm.ScoredPoint]:
        flt = self._build_date_filter(days)
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vec.tolist(),
                limit=limit,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )
        if hasattr(self.client, "search_points"):
            return self.client.search_points(
                collection_name=self.collection,
                query_vector=query_vec.tolist(),
                limit=limit,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )

        # Newer client API (query_points)
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
        return list(getattr(resp, "points", []))


# ----------------------------
# BM25 index (local)
# ----------------------------
class BM25Index:
    def __init__(self, articles: List[Article]):
        self.articles = articles
        self.id_to_pos = {a.id: i for i, a in enumerate(articles)}
        self.title_desc_texts = [f"{a.title}\n{a.description}".strip() for a in articles]
        self.body_texts = [a.canonical_text for a in articles]
        self.title_desc_bm25 = BM25Okapi([simple_tokenize(text) for text in self.title_desc_texts])
        self.body_bm25 = BM25Okapi([simple_tokenize(text) for text in self.body_texts])

    def _topk_from_scores(self, scores: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if k <= 0 or len(scores) == 0:
            return []
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return [(self.articles[i].id, float(scores[i])) for i in idx if scores[i] > 0]

    def topk_title_desc(self, query: str, k: int = 80) -> List[Tuple[str, float]]:
        qt = simple_tokenize(query)
        scores = self.title_desc_bm25.get_scores(qt)
        return self._topk_from_scores(scores, k)

    def topk_body(self, query: str, k: int = 120) -> List[Tuple[str, float]]:
        qt = simple_tokenize(query)
        scores = self.body_bm25.get_scores(qt)
        return self._topk_from_scores(scores, k)

    def topk_weighted(
        self,
        query: str,
        k_title: int = 80,
        k_body: int = 120,
        alpha_title: float = 0.7,
        alpha_body: float = 0.3,
    ) -> List[Tuple[str, float]]:
        qt = simple_tokenize(query)
        title_scores = np.asarray(self.title_desc_bm25.get_scores(qt), dtype=np.float32)
        body_scores = np.asarray(self.body_bm25.get_scores(qt), dtype=np.float32)
        scores = alpha_title * title_scores + alpha_body * body_scores
        return self._topk_from_scores(scores, max(k_title, k_body))

    def topk(self, query: str, k: int = 200) -> List[Tuple[str, float]]:
        return self.topk_body(query, k=k)


def accumulate_weighted_rrf(
    score_map: Dict[str, float],
    article_id: str,
    query_weight: float,
    rank: int,
    rrf_k: int,
) -> None:
    if rank <= 0:
        return
    score_map[article_id] = score_map.get(article_id, 0.0) + (float(query_weight) / float(rrf_k + rank))


def normalize_score_dict(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= 0:
        return {k: 0.0 for k in score_map}
    if hi - lo <= 1e-12:
        return {k: 1.0 for k, v in score_map.items() if v > 0}
    return {k: float((v - lo) / (hi - lo)) for k, v in score_map.items()}


def weighted_rrf(
    candidate_ids: Set[str],
    dense_rrf_scores: Dict[str, float],
    lex_rrf_scores: Dict[str, float],
) -> Dict[str, float]:
    dense_subset = {aid: dense_rrf_scores.get(aid, 0.0) for aid in candidate_ids if dense_rrf_scores.get(aid, 0.0) > 0}
    lex_subset = {aid: lex_rrf_scores.get(aid, 0.0) for aid in candidate_ids if lex_rrf_scores.get(aid, 0.0) > 0}
    dense_norm = normalize_score_dict(dense_subset)
    lex_norm = normalize_score_dict(lex_subset)
    return {
        aid: float(0.65 * dense_norm.get(aid, 0.0) + 0.35 * lex_norm.get(aid, 0.0))
        for aid in candidate_ids
    }


def _article_from_hit(hit: qm.ScoredPoint, id_to_article: Dict[str, Article]) -> Optional[Article]:
    aid = str(hit.id)
    article = id_to_article.get(aid)
    if article is not None:
        return article
    payload = getattr(hit, "payload", None) or {}
    if not payload:
        return None
    article = Article.from_qdrant_payload(payload, article_id=aid)
    id_to_article[aid] = article
    return article


def collect_dense_candidates_multiquery(
    qdrant_index: DenseIndexQdrant,
    id_to_article: Dict[str, Article],
    query_specs: List[QuerySpec],
    days: int,
    dense_per_anchor: int,
    dense_per_expansion: int,
    rrf_k: int,
) -> Tuple[Set[str], Dict[str, float], int]:
    candidate_ids: Set[str] = set()
    rrf_scores: Dict[str, float] = {}
    total_hits = 0

    for spec in query_specs:
        if spec.vec is None:
            continue
        limit = dense_per_anchor if spec.kind == "anchor" else dense_per_expansion
        if limit <= 0:
            continue
        hits = qdrant_index.search(spec.vec, limit=limit, days=days)
        total_hits += len(hits)
        for rank, hit in enumerate(hits, 1):
            article = _article_from_hit(hit, id_to_article)
            if article is None:
                continue
            candidate_ids.add(article.id)
            accumulate_weighted_rrf(rrf_scores, article.id, spec.weight, rank, rrf_k)

    return candidate_ids, rrf_scores, total_hits


def collect_bm25_candidates_multiquery(
    bm25_index: BM25Index,
    id_to_article: Dict[str, Article],
    query_specs: List[QuerySpec],
    bm25_title_k: int,
    bm25_body_k: int,
    rrf_k: int,
) -> Tuple[Set[str], Dict[str, float], Dict[str, float], Dict[str, float], int]:
    candidate_ids: Set[str] = set()
    rrf_scores: Dict[str, float] = {}
    title_best_scores: Dict[str, float] = {}
    body_best_scores: Dict[str, float] = {}
    total_hits = 0

    for spec in query_specs:
        if bm25_title_k > 0:
            title_hits = bm25_index.topk_title_desc(spec.text, k=bm25_title_k)
            total_hits += len(title_hits)
            for rank, (aid, score) in enumerate(title_hits, 1):
                if aid not in id_to_article:
                    continue
                candidate_ids.add(aid)
                weighted_score = float(spec.weight) * float(score)
                title_best_scores[aid] = max(title_best_scores.get(aid, 0.0), weighted_score)
                accumulate_weighted_rrf(rrf_scores, aid, spec.weight, rank, rrf_k)

        if bm25_body_k > 0:
            body_hits = bm25_index.topk_body(spec.text, k=bm25_body_k)
            total_hits += len(body_hits)
            for rank, (aid, score) in enumerate(body_hits, 1):
                if aid not in id_to_article:
                    continue
                candidate_ids.add(aid)
                weighted_score = float(spec.weight) * float(score)
                body_best_scores[aid] = max(body_best_scores.get(aid, 0.0), weighted_score)
                accumulate_weighted_rrf(rrf_scores, aid, spec.weight, rank, rrf_k)

    return candidate_ids, rrf_scores, title_best_scores, body_best_scores, total_hits


def compute_exact_similarity_features(
    candidates: Dict[str, Article],
    query_specs: List[QuerySpec],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    candidate_ids = list(candidates.keys())
    active_specs = [spec for spec in query_specs if spec.vec is not None]
    if not candidate_ids or not active_specs:
        zeros = {aid: 0.0 for aid in candidate_ids}
        return zeros, zeros.copy()

    cand_mat = np.vstack([candidates[aid].embedding for aid in candidate_ids]).astype(np.float32, copy=False)
    query_mat = np.vstack([spec.vec for spec in active_specs]).astype(np.float32, copy=False)
    query_weights = np.asarray([spec.weight for spec in active_specs], dtype=np.float32)

    weighted_sims = (cand_mat @ query_mat.T) * query_weights[None, :]
    dense_max_arr = np.clip(weighted_sims.max(axis=1), 0.0, 1.0)

    if weighted_sims.shape[1] == 1:
        dense_cov_arr = dense_max_arr.copy()
    else:
        top2 = np.sort(weighted_sims, axis=1)[:, -2:]
        dense_cov_arr = np.clip(top2.mean(axis=1), 0.0, 1.0)

    dense_max = {aid: float(val) for aid, val in zip(candidate_ids, dense_max_arr.tolist())}
    dense_cov = {aid: float(val) for aid, val in zip(candidate_ids, dense_cov_arr.tolist())}
    return dense_max, dense_cov


def compute_multiquery_final_scores(
    candidate_ids: List[str],
    s_rrf: Dict[str, float],
    s_dense_max: Dict[str, float],
    s_dense_cov: Dict[str, float],
    s_lex_max: Dict[str, float],
    s_lex_title: Dict[str, float],
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    final_scores: Dict[str, float] = {}
    feature_map: Dict[str, Dict[str, float]] = {}

    for aid in candidate_ids:
        dense_max = float(s_dense_max.get(aid, 0.0))
        lex_max = float(s_lex_max.get(aid, 0.0))
        if dense_max < min_sim and lex_max < min_bm25:
            continue

        dense_cov = float(s_dense_cov.get(aid, dense_max))
        rrf_score = float(s_rrf.get(aid, 0.0))
        lex_title = float(s_lex_title.get(aid, 0.0))
        final_score = (
            0.45 * rrf_score
            + 0.30 * dense_max
            + 0.15 * dense_cov
            + 0.10 * lex_max
        )

        final_scores[aid] = float(final_score)
        feature_map[aid] = {
            "s_rrf": rrf_score,
            "s_dense_max": dense_max,
            "s_dense_cov": dense_cov,
            "s_lex_max": lex_max,
            "s_lex_title": lex_title,
            "final_score": float(final_score),
        }

    return final_scores, feature_map


def retrieve_feed_multiquery_for_interest(
    qdrant_index: DenseIndexQdrant,
    bm25_index: Optional[BM25Index],
    id_to_article: Dict[str, Article],
    query_specs: List[QuerySpec],
    top_k: int,
    days: int = 14,
    dense_per_anchor: int = 300,
    dense_per_expansion: int = 120,
    bm25_title_k: int = 80,
    bm25_body_k: int = 120,
    lang_filter: Optional[Set[str]] = None,
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
    dense_only: bool = False,
    rrf_k: int = 60,
    candidate_cap: int = 800,
    scores_out: Optional[Dict[str, float]] = None,
    debug: bool = False,
    debug_label: Optional[str] = None,
) -> List[Article]:
    label = debug_label or "interest"
    query_specs = merge_query_specs(query_specs)
    if not query_specs:
        return []

    if debug:
        print(f"[retrieval:{label}] query_specs={len(query_specs)}")

    dense_candidate_ids, dense_rrf_raw, dense_hits = collect_dense_candidates_multiquery(
        qdrant_index=qdrant_index,
        id_to_article=id_to_article,
        query_specs=query_specs,
        days=days,
        dense_per_anchor=dense_per_anchor,
        dense_per_expansion=dense_per_expansion,
        rrf_k=rrf_k,
    )

    bm25_candidate_ids: Set[str] = set()
    lex_rrf_raw: Dict[str, float] = {}
    title_best_raw: Dict[str, float] = {}
    body_best_raw: Dict[str, float] = {}
    bm25_hits = 0
    if not dense_only and bm25_index is not None:
        (
            bm25_candidate_ids,
            lex_rrf_raw,
            title_best_raw,
            body_best_raw,
            bm25_hits,
        ) = collect_bm25_candidates_multiquery(
            bm25_index=bm25_index,
            id_to_article=id_to_article,
            query_specs=query_specs,
            bm25_title_k=bm25_title_k,
            bm25_body_k=bm25_body_k,
            rrf_k=rrf_k,
        )

    union_ids = dense_candidate_ids | bm25_candidate_ids
    if debug:
        print(
            f"[retrieval:{label}] dense_candidates={len(dense_candidate_ids)} raw_dense_hits={dense_hits} "
            f"bm25_candidates={len(bm25_candidate_ids)} raw_bm25_hits={bm25_hits} union_before_cap={len(union_ids)}"
        )

    if not union_ids:
        return []

    rrf_scores = weighted_rrf(union_ids, dense_rrf_raw, lex_rrf_raw)
    ordered_union = sorted(union_ids, key=lambda aid: rrf_scores.get(aid, 0.0), reverse=True)
    capped_ids = ordered_union[: max(1, int(candidate_cap))]
    candidate_ids = set(capped_ids)

    if debug:
        print(f"[retrieval:{label}] union_after_cap={len(candidate_ids)}")

    candidates = {
        aid: id_to_article[aid]
        for aid in capped_ids
        if aid in id_to_article
    }
    if not candidates:
        return []

    missing_ids = [aid for aid, article in candidates.items() if article.embedding is None]
    if missing_ids:
        vecs = qdrant_index.retrieve_vectors(missing_ids, batch_size=256)
        for aid in missing_ids:
            vec = vecs.get(aid)
            if vec is not None:
                candidates[aid].embedding = vec

    candidates = {aid: article for aid, article in candidates.items() if article.embedding is not None}
    if not candidates:
        return []

    if lang_filter:
        candidates = {
            aid: article
            for aid, article in candidates.items()
            if (article.lang or "").lower() in lang_filter
        }
        if not candidates:
            return []

    surviving_ids = list(candidates.keys())
    surviving_set = set(surviving_ids)
    rrf_scores = weighted_rrf(surviving_set, dense_rrf_raw, lex_rrf_raw)

    title_best_norm = normalize_score_dict({aid: title_best_raw.get(aid, 0.0) for aid in surviving_ids if title_best_raw.get(aid, 0.0) > 0})
    body_best_norm = normalize_score_dict({aid: body_best_raw.get(aid, 0.0) for aid in surviving_ids if body_best_raw.get(aid, 0.0) > 0})
    lex_max = {aid: max(title_best_norm.get(aid, 0.0), body_best_norm.get(aid, 0.0)) for aid in surviving_ids}

    dense_max, dense_cov = compute_exact_similarity_features(candidates, query_specs)
    final_scores, feature_map = compute_multiquery_final_scores(
        candidate_ids=surviving_ids,
        s_rrf=rrf_scores,
        s_dense_max=dense_max,
        s_dense_cov=dense_cov,
        s_lex_max=lex_max,
        s_lex_title=title_best_norm,
        min_sim=min_sim,
        min_bm25=min_bm25 if not dense_only else 0.0,
    )

    ordered_ids = sorted(final_scores.keys(), key=lambda aid: final_scores[aid], reverse=True)
    final_ids = ordered_ids[:top_k]

    if debug:
        print(f"[retrieval:{label}] survivors_after_thresholds={len(final_scores)}")
        for rank, aid in enumerate(final_ids[:5], 1):
            feats = feature_map.get(aid, {})
            article = candidates[aid]
            print(
                f"[retrieval:{label}] top{rank} score={final_scores[aid]:.4f} "
                f"rrf={feats.get('s_rrf', 0.0):.4f} dense_max={feats.get('s_dense_max', 0.0):.4f} "
                f"dense_cov={feats.get('s_dense_cov', 0.0):.4f} lex_max={feats.get('s_lex_max', 0.0):.4f} "
                f"title={article.title[:120]}"
            )

    if scores_out is not None:
        scores_out.clear()
        for aid in final_ids:
            scores_out[aid] = float(final_scores.get(aid, 0.0))

    return [candidates[aid] for aid in final_ids]


def retrieve_feed(
    qdrant_index: DenseIndexQdrant,
    bm25_index: Optional[BM25Index],
    id_to_article: Dict[str, Article],
    user: UserProfile,
    dense_per_center: int = 250,
    bm25_k: int = 250,
    days: int = 14,
    top_k: int = 20,
    lang_filter: Optional[Set[str]] = None,
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
    dense_only: bool = False,
    scores_out: Optional[Dict[str, float]] = None,
) -> List[Article]:
    query_specs = merge_query_specs(
        [QuerySpec(text=text, weight=1.0, kind="anchor") for text in user.interests_text if str(text or "").strip()]
    )
    if query_specs:
        embedder = Embedder("BAAI/bge-m3", use_fp16=True)
        vecs = embedder.encode([spec.text for spec in query_specs], batch_size=min(16, len(query_specs)))
        for spec, vec in zip(query_specs, vecs):
            spec.vec = np.asarray(vec, dtype=np.float32)

    return retrieve_feed_multiquery_for_interest(
        qdrant_index=qdrant_index,
        bm25_index=bm25_index,
        id_to_article=id_to_article,
        query_specs=query_specs,
        top_k=top_k,
        days=days,
        dense_per_anchor=dense_per_center,
        dense_per_expansion=max(1, int(round(dense_per_center * 0.4))),
        bm25_title_k=bm25_k,
        bm25_body_k=bm25_k,
        lang_filter=lang_filter,
        min_sim=min_sim,
        min_bm25=min_bm25,
        dense_only=dense_only,
        rrf_k=60,
        candidate_cap=max(800, top_k),
        scores_out=scores_out,
    )


# ----------------------------
# End-to-end: load -> dedup -> embed -> index -> demo retrieval
# ----------------------------
def load_articles(path: str, limit: Optional[int] = None) -> List[Article]:
    """
    Supports:
      - JSONL (one json per line)
      - JSON (list of objects)
    """
    articles: List[Article] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for d in data:
                articles.append(Article.from_dict(d))
                if limit and len(articles) >= limit:
                    break
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                articles.append(Article.from_dict(d))
                if limit and len(articles) >= limit:
                    break
    return articles


def load_articles_cached(path: str, limit: Optional[int] = None, cache_dir: Optional[Path] = None) -> Tuple[List[Article], bool]:
    cache_dir = cache_dir or default_cache_dir()
    src = Path(path)
    try:
        stat = src.stat()
    except FileNotFoundError:
        return load_articles(path, limit=limit), False

    signature = "|".join(
        [
            f"v={CACHE_VERSION}",
            f"path={src.resolve()}",
            f"mtime_ns={stat.st_mtime_ns}",
            f"size={stat.st_size}",
            f"limit={limit if limit is not None else 'all'}",
        ]
    )
    cache_path = cache_dir / f"articles_file_{namespace_hash(str(src.resolve()))}_{limit if limit is not None else 'all'}.pkl"
    cached = load_pickle_cache(cache_path)
    if cached and cached.get("signature") == signature and isinstance(cached.get("articles"), list):
        return cached["articles"], True

    articles = load_articles(path, limit=limit)
    save_pickle_cache(
        cache_path,
        {
            "signature": signature,
            "articles": articles,
        },
    )
    return articles, False


def load_or_build_bm25(articles: List[Article], cache_key: str, cache_dir: Optional[Path] = None) -> Tuple[BM25Index, bool]:
    cache_dir = cache_dir or default_cache_dir()
    signature = articles_signature(articles)
    cache_path = cache_dir / f"bm25_{namespace_hash(cache_key)}_{signature}.pkl"
    cached = load_pickle_cache(cache_path)
    if cached and cached.get("signature") == signature and isinstance(cached.get("bm25"), BM25Index):
        bm25 = cached["bm25"]
        bm25.articles = articles
        bm25.id_to_pos = {a.id: i for i, a in enumerate(articles)}
        return bm25, True

    bm25 = BM25Index(articles)
    save_pickle_cache(
        cache_path,
        {
            "signature": signature,
            "bm25": bm25,
        },
    )
    return bm25, False

def dedup_articles(articles: List[Article]) -> List[Article]:
    seen_fp = set()
    out = []
    for a in articles:
        if a.fingerprint in seen_fp:
            continue
        seen_fp.add(a.fingerprint)
        out.append(a)
    return out

def main(
    data_path: str,
    qdrant_url: str = "http://localhost:6333",
    collection: str = "news_dense",
    max_articles: Optional[int] = None,
    user_interests: Optional[List[str]] = None,
    user_tags: Optional[List[str]] = None,
    reindex: bool = False,
    resume_index: bool = False,
    lang: Optional[str] = None,
    days: int = 100,
    tau_hours: float = 336.0,
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
    dense_only: bool = False,
    aggregate: bool = False,
    top_k_per_interest: int = 20,
    max_expansions_per_interest: int = 6,
    dense_per_anchor: int = 300,
    dense_per_expansion: int = 120,
    bm25_title_k: int = 80,
    bm25_body_k: int = 120,
    rrf_k: int = 60,
    candidate_cap: int = 800,
    debug_retrieval: bool = False,
    out_path: Optional[str] = None,
):
    # 1) Load + dedup
    raw, raw_from_cache = load_articles_cached(data_path, limit=max_articles)
    articles = dedup_articles(raw)
    print(f"Loaded: {len(raw)} | After dedup: {len(articles)}{' | cache=articles' if raw_from_cache else ''}")

    # 2) Init Qdrant + optional reindex
    qindex = DenseIndexQdrant(url=qdrant_url, collection=collection)
    collection_exists = qindex.collection_exists()
    points_count = qindex.get_points_count() if collection_exists else 0

    need_index = reindex or resume_index or (points_count == 0)

    if need_index:
        articles_to_index = articles
        if resume_index and not reindex and collection_exists:
            existing_ids = qindex.get_existing_ids()
            articles_to_index = [a for a in articles if a.id not in existing_ids]
            print(
                f"Qdrant resume mode: existing={len(existing_ids)} | missing={len(articles_to_index)}"
            )

        if not articles_to_index:
            print(f"Qdrant already contains all {len(articles)} deduplicated articles. Skipping re-embedding.")
        else:
        # 3) Embed canonical_text
            embedder = Embedder("BAAI/bge-m3", use_fp16=True)
            texts = [a.canonical_text for a in articles_to_index]

            all_vecs = []
            bs = 32
            for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
                chunk = texts[i:i+bs]
                vecs = embedder.encode(chunk, batch_size=bs)
                all_vecs.append(vecs)
            all_vecs = np.vstack(all_vecs)

            for a, v in zip(articles_to_index, all_vecs):
                a.embedding = v

            dim = int(all_vecs.shape[1])
            print("Embedding dim:", dim)

            # 4) Index in Qdrant
            if reindex:
                qindex.recreate(vector_dim=dim)
            else:
                qindex.ensure_collection(vector_dim=dim)
            qindex.upsert_articles(articles_to_index, batch_size=256)
            print(f"Qdrant upsert done. Indexed {len(articles_to_index)} articles.")
    else:
        print(f"Qdrant already has {points_count} points. Skipping re-embedding.")

    # 5) BM25 index (local)
    bm25 = None
    if not dense_only:
        bm25, bm25_from_cache = load_or_build_bm25(
            articles,
            cache_key=f"file:{Path(data_path).resolve()}:{max_articles if max_articles is not None else 'all'}",
        )
        if bm25_from_cache:
            print("Loaded BM25 from cache.")

    # 6) Local map
    id_to_article = {a.id: a for a in articles}

    # 7) Embedder for queries
    embedder = Embedder("BAAI/bge-m3", use_fp16=True)

    # 8) Build interest list
    interests = [i for i in (user_interests or []) if i and i.strip()]
    if not interests:
        interests = [
            "Guerres et conflits internationaux",
            "IA et les LLMs",
            "Politique Française",
            "SpaceX",
            "Apple",
        ]
    tags = [t for t in (user_tags or []) if t and t.strip()]

    # 9) Retrieve personalized feed
    lang_filter = None
    if lang:
        lang_filter = {l.strip().lower() for l in lang.split(",") if l.strip()}

    if dense_only:
        min_bm25 = 0.0

    export_blocks: List[Dict[str, Any]] = []
    expansions_map: Dict[str, List[str]] = {}

    def _build_specs_for_interest(interest_text: str) -> List[QuerySpec]:
        specs = build_interest_query_specs(
            interest=interest_text,
            expansions_map=expansions_map,
            embedder=embedder,
            max_expansions=max_expansions_per_interest,
        )
        if tags:
            specs = merge_query_specs(specs + build_tag_query_specs(tags, embedder))
        return specs

    # Default behavior:
    # - If multiple interests are provided, return 20 articles per interest.
    # - Use --aggregate to get a single combined feed.
    if aggregate or len(interests) <= 1:
        aggregate_specs: List[QuerySpec] = []
        for interest in interests:
            aggregate_specs.extend(_build_specs_for_interest(interest))
        aggregate_specs = merge_query_specs(aggregate_specs)

        scores_map: Dict[str, float] = {}
        feed = retrieve_feed_multiquery_for_interest(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            query_specs=aggregate_specs,
            top_k=top_k_per_interest,
            days=days,
            dense_per_anchor=dense_per_anchor,
            dense_per_expansion=dense_per_expansion,
            bm25_title_k=bm25_title_k,
            bm25_body_k=bm25_body_k,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            rrf_k=rrf_k,
            candidate_cap=candidate_cap,
            scores_out=scores_map,
            debug=debug_retrieval,
            debug_label="aggregate",
        )

        print("\n--- FEED (AGGREGATE) ---")
        for i, a in enumerate(feed, 1):
            d = a.date.isoformat() if a.date else "no-date"
            print(f"{i:02d}. [{d}] ({a.domain}) {a.title}\n    {a.url}\n")

        export_blocks.append(
            {
                "interest": "; ".join(interests) if interests else "aggregate",
                "n": len(feed),
                "hits": [
                    {
                        "rank": i,
                        "id": a.id,
                        "score": float(scores_map.get(a.id, 0.0)),
                        "payload": {
                            "article_id": a.id,
                            "title": a.title,
                            "description": a.description,
                            "url": a.url,
                            "domain": a.domain,
                            "date": a.date.isoformat() if a.date else None,
                            "lang": a.lang,
                            "fingerprint": a.fingerprint,
                            "canonical_text": a.canonical_text,
                        },
                    }
                    for i, a in enumerate(feed, 1)
                ],
            }
        )

        if out_path:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            payload = {
                "generated_at": now_utc().isoformat(),
                "data_path": data_path,
                "qdrant_url": qdrant_url,
                "collection": collection,
                "dense_model": "BAAI/bge-m3",
                "days": days,
                "topk": top_k_per_interest,
                "dense_only": dense_only,
                "aggregate": aggregate,
                "tags": tags,
                "lang": lang,
                "tau_hours": tau_hours,
                "min_sim": min_sim,
                "min_bm25": min_bm25,
                "results": export_blocks,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Saved retrieval results to: {out_path}")

        return

    print("\n--- FEED (PER INTEREST) ---")
    total = 0
    for j, interest in enumerate(interests, 1):
        query_specs = _build_specs_for_interest(interest)

        scores_map: Dict[str, float] = {}
        feed = retrieve_feed_multiquery_for_interest(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            query_specs=query_specs,
            top_k=top_k_per_interest,
            days=days,
            dense_per_anchor=dense_per_anchor,
            dense_per_expansion=dense_per_expansion,
            bm25_title_k=bm25_title_k,
            bm25_body_k=bm25_body_k,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            rrf_k=rrf_k,
            candidate_cap=candidate_cap,
            scores_out=scores_map,
            debug=debug_retrieval,
            debug_label=interest,
        )

        print(f"\n### Interest {j}/{len(interests)}: {interest} (n={len(feed)})")
        for i, a in enumerate(feed, 1):
            total += 1
            d = a.date.isoformat() if a.date else "no-date"
            print(f"{i:02d}. [{d}] ({a.domain}) {a.title}\n    {a.url}\n")

        export_blocks.append(
            {
                "interest": interest,
                "n": len(feed),
                "hits": [
                    {
                        "rank": i,
                        "id": a.id,
                        "score": float(scores_map.get(a.id, 0.0)),
                        "payload": {
                            "article_id": a.id,
                            "title": a.title,
                            "description": a.description,
                            "url": a.url,
                            "domain": a.domain,
                            "date": a.date.isoformat() if a.date else None,
                            "lang": a.lang,
                            "fingerprint": a.fingerprint,
                            "canonical_text": a.canonical_text,
                        },
                    }
                    for i, a in enumerate(feed, 1)
                ],
            }
        )

    print(f"Total articles printed: {total}")

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "generated_at": now_utc().isoformat(),
            "data_path": data_path,
            "qdrant_url": qdrant_url,
            "collection": collection,
            "dense_model": "BAAI/bge-m3",
            "days": days,
            "topk": top_k_per_interest,
            "dense_only": dense_only,
            "aggregate": aggregate,
            "tags": tags,
            "lang": lang,
            "tau_hours": tau_hours,
            "min_sim": min_sim,
            "min_bm25": min_bm25,
            "results": export_blocks,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved retrieval results to: {out_path}")

if __name__ == "__main__":
    # Example:
    # python news_reco.py merged_dataset.json --interest "football européen" --interest "LLM" --tags "football,IA"
    import argparse

    default_data_path = str((Path(__file__).resolve().parent / "merged_dataset.json"))

    parser = argparse.ArgumentParser(description="News retrieval demo")
    parser.add_argument(
        "data_path",
        nargs="?",
        default=default_data_path,
        help=f"Path to JSON or JSONL dataset (default: {default_data_path})",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="news_dense")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument(
        "--interest",
        action="append",
        default=None,
        help="User interest phrase (repeat for multiple)",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-embedding and reindexing",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Filter by language code(s), comma-separated (e.g., en,fr)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=100,
        help="Dense retrieval date window in days (Qdrant filter). Use 0 to disable.",
    )
    parser.add_argument(
        "--tau-hours",
        type=float,
        default=336.0,
        help="Recency decay time constant in hours (higher = less aggressive freshness bias)",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.15,
        help="Minimum dense similarity to keep a candidate",
    )
    parser.add_argument(
        "--min-bm25",
        type=float,
        default=0.05,
        help="Minimum BM25 normalized score to keep a candidate",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Use dense retrieval only (ignore BM25)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Return a single combined feed instead of 20 per interest",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of articles to return per interest (or total if --aggregate)",
    )
    parser.add_argument("--max-expansions-per-interest", type=int, default=6)
    parser.add_argument("--dense-per-anchor", type=int, default=300)
    parser.add_argument("--dense-per-expansion", type=int, default=120)
    parser.add_argument("--bm25-title-k", type=int, default=80)
    parser.add_argument("--bm25-body-k", type=int, default=120)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-cap", type=int, default=800)
    parser.add_argument("--debug-retrieval", action="store_true")

    parser.add_argument(
        "--out",
        default=None,
        help="Write retrieval results to JSON (grouped by interest).",
    )

    args = parser.parse_args()
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] if args.tags else None

    main(
        data_path=args.data_path,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        max_articles=args.max_articles,
        user_interests=args.interest,
        user_tags=tags,
        reindex=args.reindex,
        lang=args.lang,
        days=args.days,
        tau_hours=args.tau_hours,
        min_sim=args.min_sim,
        min_bm25=args.min_bm25,
        dense_only=args.dense_only,
        aggregate=args.aggregate,
        top_k_per_interest=args.topk,
        max_expansions_per_interest=args.max_expansions_per_interest,
        dense_per_anchor=args.dense_per_anchor,
        dense_per_expansion=args.dense_per_expansion,
        bm25_title_k=args.bm25_title_k,
        bm25_body_k=args.bm25_body_k,
        rrf_k=args.rrf_k,
        candidate_cap=args.candidate_cap,
        debug_retrieval=args.debug_retrieval,
        out_path=args.out,
    )
