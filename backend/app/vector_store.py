from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import settings
from .db import now_utc


def _vector_literal(vec: list[float]) -> str:
    # pgvector accepts '[1,2,3]' text input.
    inner = ",".join(f"{float(x):.6f}" for x in vec)
    return f"[{inner}]"


@dataclass(frozen=True)
class VectorHit:
    asset_id: str
    score: float
    kind: str
    text_source: str | None


def _connect() -> Any:
    try:
        import psycopg
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("psycopg is required for pgvector support") from exc
    # Avoid indefinite hangs when Postgres is down or unreachable (indexing would otherwise block here).
    return psycopg.connect(settings.pg_dsn, connect_timeout=10)


def migrate_pgvector() -> None:
    conn = _connect()
    dim = int(settings.embedding_vector_dim)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS videowala_pg_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cur.execute("SELECT value FROM videowala_pg_meta WHERE key = 'embedding_vector_dim'")
            row = cur.fetchone()
            stored = int(row[0]) if row and str(row[0]).isdigit() else None
            if stored != dim:
                cur.execute("DROP INDEX IF EXISTS idx_asset_vectors_vector")
                cur.execute("DROP TABLE IF EXISTS asset_vectors CASCADE")
            cur.execute(
                """
                INSERT INTO videowala_pg_meta (key, value)
                VALUES ('embedding_vector_dim', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (str(dim),),
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS asset_vectors (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    vector vector({dim}) NOT NULL,
                    text_source TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(tenant_id, event_id, asset_id, kind)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_asset_vectors_event ON asset_vectors(tenant_id, event_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_asset_vectors_kind ON asset_vectors(kind)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_vectors_vector ON asset_vectors USING ivfflat (vector vector_cosine_ops)"
            )
        conn.commit()
    finally:
        conn.close()


def upsert_asset_vector(
    *,
    tenant_id: str,
    event_id: str,
    asset_id: str,
    kind: str,
    vector: list[float],
    text_source: str | None,
    created_at: datetime | None = None,
) -> int:
    created_at = created_at or now_utc()
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset_vectors (tenant_id, event_id, asset_id, kind, vector, text_source, created_at)
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (tenant_id, event_id, asset_id, kind)
                DO UPDATE SET vector = EXCLUDED.vector, text_source = EXCLUDED.text_source, created_at = EXCLUDED.created_at
                RETURNING id
                """,
                (tenant_id, event_id, asset_id, kind, _vector_literal(vector), text_source, created_at),
            )
            row = cur.fetchone()
        conn.commit()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def search_vectors(
    *,
    tenant_id: str,
    event_id: str,
    query_vector: list[float],
    kind: str = "multi",
    limit: int = 20,
) -> list[VectorHit]:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT asset_id, kind, text_source,
                       1 - (vector <=> %s::vector) AS score
                FROM asset_vectors
                WHERE tenant_id = %s AND event_id = %s AND kind = %s
                ORDER BY vector <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(query_vector), tenant_id, event_id, kind, _vector_literal(query_vector), limit),
            )
            rows = cur.fetchall()
        return [VectorHit(asset_id=r[0], kind=r[1], text_source=r[2], score=float(r[3])) for r in rows]
    finally:
        conn.close()
