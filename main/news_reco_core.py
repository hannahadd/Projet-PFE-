import os
import re
import json
import hashlib
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
                with_vectors=True,
            )
        if hasattr(self.client, "search_points"):
            return self.client.search_points(
                collection_name=self.collection,
                query_vector=query_vec.tolist(),
                limit=limit,
                query_filter=flt,
                with_payload=True,
                with_vectors=True,
            )

        # Newer client API (query_points)
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=limit,
            query_filter=flt,
            with_payload=True,
            with_vectors=True,
        )
        return list(getattr(resp, "points", []))


# ----------------------------
# BM25 index (local)
# ----------------------------
class BM25Index:
    def __init__(self, articles: List[Article]):
        self.articles = articles
        self.id_to_pos = {a.id: i for i, a in enumerate(articles)}
        corpus_tokens = [simple_tokenize(a.canonical_text) for a in articles]
        self.bm25 = BM25Okapi(corpus_tokens)

    def topk(self, query: str, k: int = 200) -> List[Tuple[str, float]]:
        qt = simple_tokenize(query)
        scores = self.bm25.get_scores(qt)  # np array aligned with self.articles
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, kth=k)[:k]
            idx = idx[np.argsort(-scores[idx])]
        out = [(self.articles[i].id, float(scores[i])) for i in idx if scores[i] > 0]
        return out


# ----------------------------
# Retrieval + scoring + MMR
# ----------------------------
def mmr_select(
    cand_ids: List[str],
    cand_scores: Dict[str, float],
    cand_vecs: Dict[str, np.ndarray],
    top_k: int = 20,
    lambda_div: float = 0.75,
    near_dup_threshold: float = 0.92
) -> List[str]:
    """
    Greedy MMR + near-dup removal (cosine threshold).
    """
    selected: List[str] = []
    remaining = list(cand_ids)

    while remaining and len(selected) < top_k:
        best_id = None
        best_val = -1e9

        for cid in remaining:
            base = cand_scores.get(cid, 0.0)

            if not selected:
                diversity_pen = 0.0
            else:
                sims = [cosine(cand_vecs[cid], cand_vecs[sid]) for sid in selected]
                max_sim = max(sims)
                if max_sim >= near_dup_threshold:
                    # treat as near-dup -> strong penalty
                    diversity_pen = 1.0
                else:
                    diversity_pen = max_sim

            val = lambda_div * base - (1.0 - lambda_div) * diversity_pen
            if val > best_val:
                best_val = val
                best_id = cid

        if best_id is None:
            break

        selected.append(best_id)
        remaining.remove(best_id)

    return selected

def compute_final_scores(
    candidates: Dict[str, Article],
    user: UserProfile,
    bm25_scores: Dict[str, float],
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
    dense_only: bool = False
) -> Dict[str, float]:
    # normalize BM25 (kept for threshold filtering)
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

    # user centers (dense)
    centers = [c.vec for c in user.centers] if user.centers else []

    out = {}
    for aid, a in candidates.items():
        assert a.embedding is not None
        # Dense score requested:
        # score = 0.7 * sim(article, center_main) + 0.3 * max(sim(article, expansions))
        if centers:
            sim_main = cosine(a.embedding, centers[0])
            if len(centers) > 1:
                sim_exp = max(cosine(a.embedding, uc) for uc in centers[1:])
            else:
                sim_exp = sim_main
            sim_vec = 0.7 * sim_main + 0.3 * sim_exp
        else:
            sim_vec = 0.0

        bm25_norm = 0.0 if dense_only else (bm25_scores.get(aid, 0.0) / (max_bm25 + 1e-9))

        if sim_vec < min_sim and bm25_norm < min_bm25:
            continue

        score = sim_vec
        out[aid] = float(score)
    return out

def retrieve_feed(
    qdrant_index: DenseIndexQdrant,
    bm25_index: BM25Index,
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
    mmr_lambda_div: float = 0.78,
    mmr_near_dup_threshold: float = 0.92,
) -> List[Article]:

    # 1) Candidate generation
    candidates: Dict[str, Article] = {}

    # Dense ANN per center
    for c in user.centers:
        hits = qdrant_index.search(c.vec, limit=dense_per_center, days=days)
        for h in hits:
            aid = str(h.id)
            if aid in id_to_article:
                # ensure we carry vector from Qdrant if needed
                a = id_to_article[aid]
                if a.embedding is None and h.vector is not None:
                    a.embedding = np.asarray(h.vector, dtype=np.float32)
                candidates[aid] = a

    # BM25 candidates
    qtext = user.query_text()
    bm25_scores: Dict[str, float] = {}
    if not dense_only:
        bm25_top = bm25_index.topk(qtext, k=bm25_k)
        bm25_scores = {aid: s for aid, s in bm25_top}
        for aid, _ in bm25_top:
            if aid in id_to_article:
                candidates[aid] = id_to_article[aid]

    if not candidates:
        return []

    # Ensure embeddings are loaded (for BM25-only candidates when we skipped re-embedding)
    missing_ids = [aid for aid, a in candidates.items() if a.embedding is None]
    if missing_ids:
        vecs = qdrant_index.retrieve_vectors(missing_ids, batch_size=256)
        for aid in missing_ids:
            v = vecs.get(aid)
            if v is not None:
                candidates[aid].embedding = v

    # Drop candidates still missing vectors
    candidates = {aid: a for aid, a in candidates.items() if a.embedding is not None}
    if not candidates:
        return []

    # 2) Optional language filter
    if lang_filter:
        candidates = {aid: a for aid, a in candidates.items() if (a.lang or "").lower() in lang_filter}
        if not candidates:
            return []

    # 3) Scoring
    scores = compute_final_scores(
        candidates,
        user,
        bm25_scores,
        min_sim=min_sim,
        min_bm25=min_bm25,
        dense_only=dense_only,
    )

    # 4) Sort by score desc for MMR seed list
    ordered_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # vectors for MMR
    cand_vecs = {aid: candidates[aid].embedding for aid in ordered_ids if candidates[aid].embedding is not None}

    # 5) MMR diversity + near-dup control
    final_ids = mmr_select(
        cand_ids=ordered_ids,
        cand_scores=scores,
        cand_vecs=cand_vecs,
        top_k=top_k,
        lambda_div=mmr_lambda_div,
        near_dup_threshold=mmr_near_dup_threshold,
    )

    if scores_out is not None:
        scores_out.clear()
        for aid in final_ids:
            scores_out[aid] = float(scores.get(aid, 0.0))

    return [candidates[aid] for aid in final_ids]


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
    lang: Optional[str] = None,
    days: int = 100,
    tau_hours: float = 336.0,
    min_sim: float = 0.0,
    min_bm25: float = 0.0,
    dense_only: bool = False,
    aggregate: bool = False,
    top_k_per_interest: int = 20,
    out_path: Optional[str] = None,
):
    # 1) Load + dedup
    raw = load_articles(data_path, limit=max_articles)
    articles = dedup_articles(raw)
    print(f"Loaded: {len(raw)} | After dedup: {len(articles)}")

    # 2) Init Qdrant + optional reindex
    qindex = DenseIndexQdrant(url=qdrant_url, collection=collection)
    points_count = qindex.get_points_count() if qindex.collection_exists() else 0

    need_index = reindex or (points_count == 0)

    if need_index:
        # 3) Embed canonical_text
        embedder = Embedder("BAAI/bge-m3", use_fp16=True)
        texts = [a.canonical_text for a in articles]

        all_vecs = []
        bs = 32
        for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
            chunk = texts[i:i+bs]
            vecs = embedder.encode(chunk, batch_size=bs)
            all_vecs.append(vecs)
        all_vecs = np.vstack(all_vecs)

        for a, v in zip(articles, all_vecs):
            a.embedding = v

        dim = int(all_vecs.shape[1])
        print("Embedding dim:", dim)

        # 4) Index in Qdrant
        if reindex:
            qindex.recreate(vector_dim=dim)
        else:
            qindex.ensure_collection(vector_dim=dim)
        qindex.upsert_articles(articles, batch_size=256)
        print("Qdrant upsert done.")
    else:
        print(f"Qdrant already has {points_count} points. Skipping re-embedding.")

    # 5) BM25 index (local)
    bm25 = BM25Index(articles)

    # 6) Local map
    id_to_article = {a.id: a for a in articles}

    # 7) Embedder for user profile
    embedder = Embedder("BAAI/bge-m3", use_fp16=True)

    # 8) Build user profile (onboarding)
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

    # Default behavior:
    # - If multiple interests are provided, return 20 articles per interest.
    # - Use --aggregate to get a single combined feed.
    if aggregate or len(interests) <= 1:
        user = UserProfile(
            interests_text=interests,
            interests_tags=tags,
        )
        user.build_from_onboarding(embedder.encode, k_max=5)

        scores_map: Dict[str, float] = {}
        feed = retrieve_feed(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            user=user,
            dense_per_center=250,
            bm25_k=250,
            days=days,
            top_k=top_k_per_interest,
            tau_hours=tau_hours,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            scores_out=scores_map,
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
        user = UserProfile(
            interests_text=[interest],
            interests_tags=tags,
        )
        user.build_from_onboarding(embedder.encode, k_max=1)

        scores_map: Dict[str, float] = {}
        feed = retrieve_feed(
            qdrant_index=qindex,
            bm25_index=bm25,
            id_to_article=id_to_article,
            user=user,
            dense_per_center=250,
            bm25_k=250,
            days=days,
            top_k=top_k_per_interest,
            tau_hours=tau_hours,
            lang_filter=lang_filter,
            min_sim=min_sim,
            min_bm25=min_bm25,
            dense_only=dense_only,
            scores_out=scores_map,
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
        out_path=args.out,
    )
