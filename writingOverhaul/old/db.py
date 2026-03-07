from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    published_date TIMESTAMPTZ,
    source TEXT,
    url TEXT UNIQUE,
    fingerprint TEXT,
    lang TEXT,
    raw JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles (published_date DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles (source);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_path TEXT,
    qdrant_url TEXT,
    collection TEXT,
    dense_model TEXT,
    days INT,
    topk INT,
    dense_only BOOLEAN,
    aggregate BOOLEAN,
    tags JSONB,
    lang TEXT,
    tau_hours DOUBLE PRECISION,
    min_sim DOUBLE PRECISION,
    min_bm25 DOUBLE PRECISION,
    interests JSONB
);

CREATE TABLE IF NOT EXISTS retrieval_hits (
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    interest TEXT NOT NULL,
    rank INT NOT NULL,
    article_id TEXT,
    score DOUBLE PRECISION,
    payload JSONB NOT NULL,
    PRIMARY KEY (retrieval_run_id, interest, rank)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_hits_run_interest ON retrieval_hits (retrieval_run_id, interest);

CREATE TABLE IF NOT EXISTS rerank_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retrieval_run_id BIGINT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    model TEXT,
    max_length INT,
    batch_size INT,
    topn INT,
    instruction TEXT
);

CREATE TABLE IF NOT EXISTS rerank_hits (
    rerank_run_id BIGINT NOT NULL REFERENCES rerank_runs(id) ON DELETE CASCADE,
    interest TEXT NOT NULL,
    rank INT NOT NULL,
    article_id TEXT,
    dense_rank INT,
    dense_score DOUBLE PRECISION,
    rerank_score DOUBLE PRECISION,
    payload JSONB NOT NULL,
    full_article JSONB,
    PRIMARY KEY (rerank_run_id, interest, rank)
);

CREATE INDEX IF NOT EXISTS idx_rerank_hits_run_interest ON rerank_hits (rerank_run_id, interest);

CREATE TABLE IF NOT EXISTS writing_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rerank_run_id BIGINT NOT NULL REFERENCES rerank_runs(id) ON DELETE CASCADE,
    model TEXT,
    ollama_url TEXT,
    interest_batch_size INT,
    offset_interests INT,
    top_n INT,
    max_chars INT,
    temperature DOUBLE PRECISION,
    num_predict INT,
    num_ctx INT
);

CREATE TABLE IF NOT EXISTS article_summaries (
    writing_run_id BIGINT NOT NULL REFERENCES writing_runs(id) ON DELETE CASCADE,
    interest TEXT NOT NULL,
    rank INT NOT NULL,
    article_id TEXT,
    title TEXT,
    url TEXT,
    source TEXT,
    published_date TEXT,
    lang TEXT,
    rerank_score DOUBLE PRECISION,
    dense_score DOUBLE PRECISION,
    summary_fr TEXT,
    points_cles JSONB,
    notes TEXT,
    raw_llm JSONB,
    PRIMARY KEY (writing_run_id, interest, rank)
);

CREATE INDEX IF NOT EXISTS idx_article_summaries_run_interest ON article_summaries (writing_run_id, interest);
"""


@dataclass
class PostgresStore:
    db_url: str

    def connect(self):
        return psycopg.connect(self.db_url)

    def init_db(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    def upsert_articles(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        sql = """
        INSERT INTO articles (id, title, content, published_date, source, url, fingerprint, lang, raw, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
        ON CONFLICT (id) DO UPDATE SET
          title = EXCLUDED.title,
          content = EXCLUDED.content,
          published_date = EXCLUDED.published_date,
          source = EXCLUDED.source,
          url = EXCLUDED.url,
          fingerprint = EXCLUDED.fingerprint,
          lang = EXCLUDED.lang,
          raw = EXCLUDED.raw,
          updated_at = NOW();
        """
        data = [
            (
                str(r.get("id") or ""),
                r.get("title"),
                r.get("content"),
                r.get("published_date"),
                r.get("source"),
                r.get("url"),
                r.get("fingerprint"),
                r.get("lang"),
                json.dumps(r.get("raw") or r, ensure_ascii=False),
            )
            for r in rows
            if str(r.get("id") or "").strip()
        ]
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, data)
            conn.commit()
        return len(data)

    def fetch_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = """
        SELECT id, title, content, published_date, source, url, fingerprint, lang, raw
        FROM articles
        ORDER BY published_date DESC NULLS LAST, id ASC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT %s"
            params = (int(limit),)

        out: List[Dict[str, Any]] = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for rec_id, title, content, published_date, source, url, fingerprint, lang, raw in cur.fetchall():
                    row: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
                    row.update(
                        {
                            "id": rec_id,
                            "title": title,
                            "content": content,
                            "published_date": published_date.isoformat() if published_date is not None else None,
                            "source": source,
                            "url": url,
                            "fingerprint": fingerprint,
                            "lang": lang,
                        }
                    )
                    out.append(row)
        return out

    def create_retrieval_run(self, payload: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO retrieval_runs (
            data_path, qdrant_url, collection, dense_model, days, topk,
            dense_only, aggregate, tags, lang, tau_hours, min_sim, min_bm25, interests
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb)
        RETURNING id;
        """
        vals = (
            payload.get("data_path"),
            payload.get("qdrant_url"),
            payload.get("collection"),
            payload.get("dense_model"),
            payload.get("days"),
            payload.get("topk"),
            payload.get("dense_only"),
            payload.get("aggregate"),
            json.dumps(payload.get("tags") or [], ensure_ascii=False),
            payload.get("lang"),
            payload.get("tau_hours"),
            payload.get("min_sim"),
            payload.get("min_bm25"),
            json.dumps(payload.get("interests") or [], ensure_ascii=False),
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
                run_id = int(cur.fetchone()[0])
            conn.commit()
        return run_id

    def insert_retrieval_hits(self, run_id: int, blocks: List[Dict[str, Any]]) -> int:
        sql = """
        INSERT INTO retrieval_hits (retrieval_run_id, interest, rank, article_id, score, payload)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (retrieval_run_id, interest, rank) DO UPDATE SET
          article_id = EXCLUDED.article_id,
          score = EXCLUDED.score,
          payload = EXCLUDED.payload;
        """
        rows = []
        for b in blocks:
            interest = str(b.get("interest") or "")
            for h in b.get("hits") or []:
                rows.append(
                    (
                        int(run_id),
                        interest,
                        int(h.get("rank") or 0),
                        h.get("id") or (h.get("payload") or {}).get("article_id"),
                        float(h.get("score") or 0.0),
                        json.dumps(h.get("payload") or {}, ensure_ascii=False),
                    )
                )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def fetch_retrieval_blocks(self, retrieval_run_id: int) -> List[Dict[str, Any]]:
        sql = """
        SELECT interest, rank, article_id, score, payload
        FROM retrieval_hits
        WHERE retrieval_run_id = %s
        ORDER BY interest ASC, rank ASC;
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(retrieval_run_id),))
                for interest, rank, article_id, score, payload in cur.fetchall():
                    grouped.setdefault(interest, []).append(
                        {
                            "rank": int(rank),
                            "id": article_id,
                            "score": float(score or 0.0),
                            "payload": payload if isinstance(payload, dict) else {},
                        }
                    )
        return [{"interest": k, "n": len(v), "hits": v} for k, v in grouped.items()]

    def create_rerank_run(self, payload: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO rerank_runs (retrieval_run_id, model, max_length, batch_size, topn, instruction)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        vals = (
            int(payload["retrieval_run_id"]),
            payload.get("model"),
            int(payload.get("max_length") or 1024),
            int(payload.get("batch_size") or 1),
            int(payload.get("topn") or 20),
            payload.get("instruction"),
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
                run_id = int(cur.fetchone()[0])
            conn.commit()
        return run_id

    def insert_rerank_hits(self, rerank_run_id: int, blocks: List[Dict[str, Any]]) -> int:
        sql = """
        INSERT INTO rerank_hits (
            rerank_run_id, interest, rank, article_id, dense_rank, dense_score, rerank_score, payload, full_article
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (rerank_run_id, interest, rank) DO UPDATE SET
          article_id = EXCLUDED.article_id,
          dense_rank = EXCLUDED.dense_rank,
          dense_score = EXCLUDED.dense_score,
          rerank_score = EXCLUDED.rerank_score,
          payload = EXCLUDED.payload,
          full_article = EXCLUDED.full_article;
        """
        rows = []
        for b in blocks:
            interest = str(b.get("interest") or "")
            for h in b.get("hits") or []:
                rows.append(
                    (
                        int(rerank_run_id),
                        interest,
                        int(h.get("rank") or 0),
                        h.get("id") or (h.get("payload") or {}).get("article_id"),
                        int(h.get("dense_rank") or h.get("rank") or 0),
                        float(h.get("score") or 0.0),
                        float(h.get("rerank_score") or 0.0),
                        json.dumps(h.get("payload") or {}, ensure_ascii=False),
                        json.dumps(h.get("full_article") or {}, ensure_ascii=False),
                    )
                )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def fetch_rerank_blocks(self, rerank_run_id: int) -> List[Dict[str, Any]]:
        sql = """
        SELECT interest, rank, article_id, dense_rank, dense_score, rerank_score, payload, full_article
        FROM rerank_hits
        WHERE rerank_run_id = %s
        ORDER BY interest ASC, rank ASC;
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(rerank_run_id),))
                for interest, rank, article_id, dense_rank, dense_score, rerank_score, payload, full_article in cur.fetchall():
                    grouped.setdefault(interest, []).append(
                        {
                            "rank": int(rank),
                            "id": article_id,
                            "dense_rank": int(dense_rank or 0),
                            "score": float(dense_score or 0.0),
                            "rerank_score": float(rerank_score or 0.0),
                            "payload": payload if isinstance(payload, dict) else {},
                            "full_article": full_article if isinstance(full_article, dict) and full_article else None,
                        }
                    )
        return [{"interest": k, "n": len(v), "hits": v} for k, v in grouped.items()]

    def find_article(self, article_id: Optional[str], url: Optional[str]) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                if article_id:
                    cur.execute(
                        "SELECT id, title, content, published_date, source, url, fingerprint, lang, raw FROM articles WHERE id = %s",
                        (str(article_id),),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._article_row_to_dict(row)
                if url:
                    cur.execute(
                        "SELECT id, title, content, published_date, source, url, fingerprint, lang, raw FROM articles WHERE url = %s",
                        (str(url),),
                    )
                    row = cur.fetchone()
                    if row:
                        return self._article_row_to_dict(row)
        return None

    def create_writing_run(self, payload: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO writing_runs (
            rerank_run_id, model, ollama_url, interest_batch_size, offset_interests,
            top_n, max_chars, temperature, num_predict, num_ctx
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        vals = (
            int(payload["rerank_run_id"]),
            payload.get("model"),
            payload.get("ollama_url"),
            int(payload.get("interest_batch_size") or 10),
            int(payload.get("offset_interests") or 0),
            int(payload.get("top_n") or 10),
            int(payload.get("max_chars") or 9000),
            float(payload.get("temperature") or 0.2),
            int(payload.get("num_predict") or 500),
            int(payload.get("num_ctx") or 4096),
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
                run_id = int(cur.fetchone()[0])
            conn.commit()
        return run_id

    def upsert_article_summary(self, writing_run_id: int, row: Dict[str, Any]) -> None:
        sql = """
        INSERT INTO article_summaries (
            writing_run_id, interest, rank, article_id, title, url, source, published_date, lang,
            rerank_score, dense_score, summary_fr, points_cles, notes, raw_llm
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
        ON CONFLICT (writing_run_id, interest, rank) DO UPDATE SET
          article_id = EXCLUDED.article_id,
          title = EXCLUDED.title,
          url = EXCLUDED.url,
          source = EXCLUDED.source,
          published_date = EXCLUDED.published_date,
          lang = EXCLUDED.lang,
          rerank_score = EXCLUDED.rerank_score,
          dense_score = EXCLUDED.dense_score,
          summary_fr = EXCLUDED.summary_fr,
          points_cles = EXCLUDED.points_cles,
          notes = EXCLUDED.notes,
          raw_llm = EXCLUDED.raw_llm;
        """
        vals = (
            int(writing_run_id),
            row.get("interest"),
            int(row.get("rank") or 0),
            row.get("article_id"),
            row.get("title"),
            row.get("url"),
            row.get("source"),
            row.get("published_date"),
            row.get("lang"),
            float(row.get("rerank_score") or 0.0),
            float(row.get("dense_score") or 0.0),
            row.get("summary_fr") or "",
            json.dumps(row.get("points_cles") or [], ensure_ascii=False),
            row.get("notes"),
            json.dumps(row.get("raw_llm") or {}, ensure_ascii=False),
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
            conn.commit()

    @staticmethod
    def _article_row_to_dict(row: Any) -> Dict[str, Any]:
        return {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "published_date": row[3].isoformat() if row[3] is not None else None,
            "source": row[4],
            "url": row[5],
            "fingerprint": row[6],
            "lang": row[7],
            "raw": row[8] if isinstance(row[8], dict) else None,
        }
