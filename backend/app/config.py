from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


# `backend/app/config.py` → repo root is `.../videowala`
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    app_name: str = "Videowala Backend"
    # Paths are anchored at the project root so running from `backend/` or the repo root behaves the same.
    storage_root: str = Field(default_factory=lambda: str(PROJECT_ROOT / "storage"))
    scratch_root: str = Field(default_factory=lambda: str(PROJECT_ROOT / "tmp"))
    db_path: str = Field(default_factory=lambda: str(PROJECT_ROOT / "storage" / "videowala.db"))
    enable_real_face_recognition: bool = Field(default=False)
    # Stage 2 vector store (pgvector)
    pg_dsn: str = Field(default="postgresql://videowala:videowala@localhost:5432/videowala")
    stage2_stub_models: bool = Field(default=True)
    embedding_model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
