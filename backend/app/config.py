from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field


# `backend/app/config.py` → `backend/` package root and monorepo root
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _default_indexing_progress() -> bool:
    """Show tqdm bars when stderr is a TTY; override with INDEXING_PROGRESS=0|1."""
    raw = os.getenv("INDEXING_PROGRESS")
    if raw is None:
        return sys.stderr.isatty()
    return _parse_bool(raw, True)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_dotenv_if_present() -> None:
    """Load ``backend/.env`` only (shell env wins; we never override existing vars)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(dotenv_path=str(BACKEND_ROOT / ".env"), override=False)


_load_dotenv_if_present()


class Settings(BaseModel):
    app_name: str = "Videowala Backend"
    # If unset, defaults to FALSE (real model mode). If TRUE, use deterministic stubbed outputs (no GPU required).
    dev_mode: bool = Field(default_factory=lambda: _parse_bool(os.getenv("DEV_MODE"), False))
    # Paths are anchored at the project root so running from `backend/` or the repo root behaves the same.
    storage_root: str = Field(default_factory=lambda: str(PROJECT_ROOT / "storage"))
    scratch_root: str = Field(default_factory=lambda: str(PROJECT_ROOT / "tmp"))
    db_path: str = Field(default_factory=lambda: str(PROJECT_ROOT / "storage" / "videowala.db"))
    enable_real_face_recognition: bool = Field(default=False)
    # Stage 2 vector store (pgvector)
    pg_dsn: str = Field(default_factory=lambda: os.getenv("PG_DSN", "postgresql://videowala:videowala@localhost:5432/videowala"))
    stage2_stub_models: bool = Field(default_factory=lambda: _parse_bool(os.getenv("DEV_MODE"), False))
    embedding_model_id: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3"))
    # Must match the dense output dimension of `embedding_model_id` (BGE-M3 → 1024).
    embedding_vector_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_VECTOR_DIM", "1024")))
    vlm_model_id: str = Field(default_factory=lambda: os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"))
    # Lowercase tag names: if any VLM tag matches, OCR runs (after VLM). Comma-separated.
    ocr_trigger_tags: str = Field(
        default_factory=lambda: os.getenv("OCR_TRIGGER_TAGS", "text,signage,document,readable_text")
    )
    # tqdm progress for indexing (per-asset steps + batch file loop) when True.
    indexing_show_progress: bool = Field(default_factory=_default_indexing_progress)
    # Serial index jobs: 1 = strict single-flight indexing (PoC default).
    index_workers: int = Field(default_factory=lambda: max(1, int(os.getenv("INDEX_WORKERS", "1"))))
    # When ingest provides `semantic_prompt` for images, run photo culling with this keep fraction.
    image_index_semantic_cull_percent: float = Field(
        default_factory=lambda: float(os.getenv("IMAGE_INDEX_SEMANTIC_CULL_PERCENT", "0.5"))
    )


settings = Settings()


def ocr_trigger_tags_set() -> set[str]:
    return {t.strip().lower() for t in settings.ocr_trigger_tags.split(",") if t.strip()}
