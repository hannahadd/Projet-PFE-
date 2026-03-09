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

CREATE TABLE IF NOT EXISTS dedup_runs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_table TEXT NOT NULL DEFAULT 'retrieval_hits',
    representative_retrieval_run_id BIGINT REFERENCES retrieval_runs(id) ON DELETE SET NULL,
    interests JSONB,
    source_run_ids JSONB,
    params JSONB
);

CREATE TABLE IF NOT EXISTS dedup_hits (
    dedup_run_id BIGINT NOT NULL REFERENCES dedup_runs(id) ON DELETE CASCADE,
    interest TEXT NOT NULL,
    rank INT NOT NULL,
    article_id TEXT,
    score DOUBLE PRECISION,
    payload JSONB NOT NULL,
    PRIMARY KEY (dedup_run_id, interest, rank)
);

CREATE INDEX IF NOT EXISTS idx_dedup_hits_run_interest ON dedup_hits (dedup_run_id, interest);

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

ALTER TABLE rerank_runs ADD COLUMN IF NOT EXISTS source_table TEXT DEFAULT 'retrieval_hits';
ALTER TABLE rerank_runs ADD COLUMN IF NOT EXISTS source_run_id BIGINT;
ALTER TABLE rerank_runs ADD COLUMN IF NOT EXISTS dedup_run_id BIGINT REFERENCES dedup_runs(id) ON DELETE SET NULL;
UPDATE rerank_runs SET source_table = COALESCE(source_table, 'retrieval_hits');
UPDATE rerank_runs SET source_run_id = COALESCE(source_run_id, retrieval_run_id);

CREATE INDEX IF NOT EXISTS idx_rerank_runs_source ON rerank_runs (source_table, source_run_id);

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

CREATE SEQUENCE IF NOT EXISTS article_summaries_id_seq;
ALTER TABLE article_summaries ADD COLUMN IF NOT EXISTS id BIGINT;
ALTER TABLE article_summaries ALTER COLUMN id SET DEFAULT nextval('article_summaries_id_seq');
UPDATE article_summaries SET id = nextval('article_summaries_id_seq') WHERE id IS NULL;
SELECT setval(
    'article_summaries_id_seq',
    COALESCE((SELECT MAX(id) FROM article_summaries), 0) + 1,
    false
);

CREATE INDEX IF NOT EXISTS idx_article_summaries_run_interest ON article_summaries (writing_run_id, interest);
CREATE UNIQUE INDEX IF NOT EXISTS idx_article_summaries_id ON article_summaries (id);
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

    def fetch_latest_retrieval_run_ids_by_interest(self, interests: Optional[List[str]] = None) -> Dict[str, int]:
        sql = """
        SELECT interest, MAX(retrieval_run_id) AS latest_run_id
        FROM retrieval_hits
        """
        params: tuple[Any, ...] = ()
        if interests:
            sql += " WHERE LOWER(interest) = ANY(%s)"
            params = ([str(x).strip().lower() for x in interests],)
        sql += " GROUP BY interest ORDER BY interest ASC"

        out: Dict[str, int] = {}
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for interest, run_id in cur.fetchall():
                    if interest:
                        out[str(interest)] = int(run_id)
        return out

    def create_dedup_run(self, payload: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO dedup_runs (
            source_table, representative_retrieval_run_id, interests, source_run_ids, params
        ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING id;
        """
        vals = (
            payload.get("source_table") or "retrieval_hits",
            payload.get("representative_retrieval_run_id"),
            json.dumps(payload.get("interests") or [], ensure_ascii=False),
            json.dumps(payload.get("source_run_ids") or {}, ensure_ascii=False),
            json.dumps(payload.get("params") or {}, ensure_ascii=False),
        )
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
                run_id = int(cur.fetchone()[0])
            conn.commit()
        return run_id

    def insert_dedup_hits(self, run_id: int, blocks: List[Dict[str, Any]]) -> int:
        sql = """
        INSERT INTO dedup_hits (dedup_run_id, interest, rank, article_id, score, payload)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (dedup_run_id, interest, rank) DO UPDATE SET
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

    def fetch_dedup_blocks(self, dedup_run_id: int) -> List[Dict[str, Any]]:
        sql = """
        SELECT interest, rank, article_id, score, payload
        FROM dedup_hits
        WHERE dedup_run_id = %s
        ORDER BY interest ASC, rank ASC;
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(dedup_run_id),))
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

    def fetch_dedup_run(self, dedup_run_id: int) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT id, source_table, representative_retrieval_run_id, interests, source_run_ids, params, created_at
        FROM dedup_runs
        WHERE id = %s;
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(dedup_run_id),))
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "source_table": row[1],
            "representative_retrieval_run_id": row[2],
            "interests": row[3] if isinstance(row[3], list) else [],
            "source_run_ids": row[4] if isinstance(row[4], dict) else {},
            "params": row[5] if isinstance(row[5], dict) else {},
            "created_at": row[6].isoformat() if row[6] is not None else None,
        }

    def create_rerank_run(self, payload: Dict[str, Any]) -> int:
        sql = """
        INSERT INTO rerank_runs (
            retrieval_run_id, model, max_length, batch_size, topn, instruction,
            source_table, source_run_id, dedup_run_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        vals = (
            int(payload["retrieval_run_id"]),
            payload.get("model"),
            int(payload.get("max_length") or 1024),
            int(payload.get("batch_size") or 1),
            int(payload.get("topn") or 20),
            payload.get("instruction"),
            payload.get("source_table") or "retrieval_hits",
            int(payload.get("source_run_id") or payload["retrieval_run_id"]),
            int(payload.get("dedup_run_id")) if payload.get("dedup_run_id") is not None else None,
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

    def upsert_article_summary(self, writing_run_id: int, row: Dict[str, Any]) -> int:
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
                    raw_llm = EXCLUDED.raw_llm
                RETURNING id;
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
                article_summary_id = int(cur.fetchone()[0])
            conn.commit()
        return article_summary_id

    def fetch_article_summary_repair_context(self, article_summary_id: int) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT
            s.id,
            s.writing_run_id,
            wr.rerank_run_id,
            s.interest,
            s.rank,
            s.article_id,
            s.title,
            s.url,
            s.source,
            s.published_date,
            s.lang,
            s.rerank_score,
            s.dense_score,
            s.summary_fr,
            s.points_cles,
            s.notes,
            s.raw_llm,
            h.payload,
            h.full_article
        FROM article_summaries s
        JOIN writing_runs wr ON wr.id = s.writing_run_id
        LEFT JOIN rerank_hits h
          ON h.rerank_run_id = wr.rerank_run_id
         AND h.interest = s.interest
         AND h.rank = s.rank
        WHERE s.id = %s;
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(article_summary_id),))
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "writing_run_id": row[1],
            "rerank_run_id": row[2],
            "interest": row[3],
            "rank": row[4],
            "article_id": row[5],
            "title": row[6],
            "url": row[7],
            "source": row[8],
            "published_date": row[9],
            "lang": row[10],
            "rerank_score": row[11],
            "dense_score": row[12],
            "summary_fr": row[13],
            "points_cles": row[14] if isinstance(row[14], list) else [],
            "notes": row[15],
            "raw_llm": row[16] if isinstance(row[16], dict) else {},
            "payload": row[17] if isinstance(row[17], dict) else {},
            "full_article": row[18] if isinstance(row[18], dict) else {},
        }

    def update_article_summary_by_id(self, article_summary_id: int, row: Dict[str, Any]) -> None:
        sql = """
        UPDATE article_summaries
        SET
            title = %s,
            summary_fr = %s,
            points_cles = %s::jsonb,
            notes = %s,
            raw_llm = %s::jsonb
        WHERE id = %s;
        """
        vals = (
            row.get("title") or "",
            row.get("summary_fr") or "",
            json.dumps(row.get("points_cles") or [], ensure_ascii=False),
            row.get("notes"),
            json.dumps(row.get("raw_llm") or {}, ensure_ascii=False),
            int(article_summary_id),
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
