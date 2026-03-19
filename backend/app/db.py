from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import settings


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_parent_dir(path: str) -> None:
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    _ensure_parent_dir(settings.db_path)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def _json_default(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def migrate() -> None:
    with db_cursor() as cur:
        version = cur.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    venue TEXT,
                    date TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    media_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS asset_insights (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS planner_plans (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    output_type TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    actions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS render_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS render_specs (
                    render_job_id TEXT PRIMARY KEY,
                    input_files_json TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    scratch_dir TEXT NOT NULL,
                    FOREIGN KEY(render_job_id) REFERENCES render_jobs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS person_references (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_assets_event ON assets(event_id);
                CREATE INDEX IF NOT EXISTS idx_insights_event ON asset_insights(event_id);
                CREATE INDEX IF NOT EXISTS idx_insights_type ON asset_insights(insight_type);
                CREATE INDEX IF NOT EXISTS idx_persons_event ON persons(event_id);
                CREATE INDEX IF NOT EXISTS idx_person_refs_person ON person_references(person_id);
                """
            )
            cur.execute("PRAGMA user_version = 1")

        if version < 2:
            # Stage 2: persist render feature flags
            cur.executescript(
                """
                ALTER TABLE render_specs ADD COLUMN subtitles_enabled INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE render_specs ADD COLUMN overlays_enabled INTEGER NOT NULL DEFAULT 0;
                """
            )
            cur.execute("PRAGMA user_version = 2")

        if version < 3:
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS asset_proxies (
                    asset_id TEXT PRIMARY KEY,
                    proxy_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS asset_segments (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    keep INTEGER NOT NULL DEFAULT 1,
                    is_duplicate INTEGER NOT NULL DEFAULT 0,
                    reject_reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_percent INTEGER NOT NULL DEFAULT 0,
                    insights_generated INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_asset_segments_event ON asset_segments(event_id);
                CREATE INDEX IF NOT EXISTS idx_asset_segments_asset ON asset_segments(asset_id);
                CREATE INDEX IF NOT EXISTS idx_index_jobs_event ON index_jobs(event_id);
                CREATE INDEX IF NOT EXISTS idx_index_jobs_asset ON index_jobs(asset_id);
                """
            )
            cur.execute("PRAGMA user_version = 3")

        if version < 4:
            cur.executescript(
                """
                ALTER TABLE render_specs ADD COLUMN clip_inputs_json TEXT;
                ALTER TABLE render_jobs ADD COLUMN progress_percent INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE render_jobs ADD COLUMN error_message TEXT;
                """
            )
            cur.execute("PRAGMA user_version = 4")

        if version < 5:
            # PoC dashboard reads grouped event stats frequently.
            cur.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_render_jobs_tenant_event ON render_jobs(tenant_id, event_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_person_refs_tenant_event ON person_references(tenant_id, event_id, created_at);
                """
            )
            cur.execute("PRAGMA user_version = 5")


def reset_database_for_tests(path: str) -> None:
    target = Path(path)
    if not target.is_absolute():
        # Keep test DBs anchored at repo root to match runtime config.
        target = Path(settings.storage_root).parent / target
    if target.exists():
        target.unlink()
    settings.db_path = str(target)
    migrate()


def to_iso(dt: datetime) -> str:
    return dt.isoformat()


def from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def encode_json(payload: object) -> str:
    return _json_default(payload)


def decode_json(payload: str) -> object:
    return json.loads(payload)
