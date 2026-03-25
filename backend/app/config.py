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
    embedding_model_id: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID", "Alibaba-NLP/gte-Qwen2-7B-instruct")
    )
    # Must match the dense output dimension of `embedding_model_id` (gte-Qwen2-7B-instruct → 3584).
    embedding_vector_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_VECTOR_DIM", "3584")))
    vlm_model_id: str = Field(default_factory=lambda: os.getenv("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"))
    # When the primary VLM OOMs on GPU, try this smaller checkpoint (4-bit / fp16 / CPU fallbacks).
    vlm_fallback_model_id: str = Field(
        default_factory=lambda: os.getenv("VLM_FALLBACK_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
    )
    # If bitsandbytes is installed, try 4-bit NF4 on GPU before full fp16 (much lower VRAM).
    vlm_prefer_quantized: bool = Field(
        default_factory=lambda: _parse_bool(os.getenv("VLM_PREFER_QUANTIZED"), True)
    )
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
    # Plan sequencing: Qwen2.5 text LLM reorders selected segments for narrative / shot continuity (PoC).
    planner_model_enabled: bool = Field(default_factory=lambda: _parse_bool(os.getenv("PLANNER_MODEL_ENABLED"), True))
    planner_model_id: str = Field(default_factory=lambda: os.getenv("PLANNER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
    # When primary planner model OOMs on GPU, try this smaller instruct model (4-bit / fp16 / CPU).
    planner_fallback_model_id: str = Field(
        default_factory=lambda: os.getenv("PLANNER_FALLBACK_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    )
    planner_prefer_quantized: bool = Field(
        default_factory=lambda: _parse_bool(os.getenv("PLANNER_PREFER_QUANTIZED"), True)
    )
    # On ~12–16 GiB GPUs, try the smaller PLANNER_FALLBACK_MODEL_ID on CUDA before the 7B primary
    # so weights + KV for generate() fit without OOM.
    planner_prefer_small_gpu_first: bool = Field(
        default_factory=lambda: _parse_bool(os.getenv("PLANNER_PREFER_SMALL_GPU_FIRST"), True)
    )
    # If the LLM still errors (OOM, bad JSON, etc.), use deterministic continuity ordering instead of HTTP 400.
    planner_soft_fail_to_heuristic: bool = Field(
        default_factory=lambda: _parse_bool(os.getenv("PLANNER_SOFT_FAIL_TO_HEURISTIC"), True)
    )
    planner_max_segments: int = Field(default_factory=lambda: max(4, int(os.getenv("PLANNER_MAX_SEGMENTS", "80"))))
    planner_temperature: float = Field(default_factory=lambda: float(os.getenv("PLANNER_TEMPERATURE", "0.2")))
    # Floor for decode length; actual budget scales up with segment count in ``plan_sequencer``.
    planner_max_new_tokens: int = Field(default_factory=lambda: int(os.getenv("PLANNER_MAX_NEW_TOKENS", "384")))
    # Hard cap on prompt length (tokens) before left-truncation fallback — avoids multi‑GiB KV on 14 GiB GPUs.
    planner_max_input_tokens: int = Field(default_factory=lambda: int(os.getenv("PLANNER_MAX_INPUT_TOKENS", "3072")))
    # Prefer PyTorch SDPA attention (lower VRAM than eager on many setups). Set to "eager" to disable.
    planner_attn_implementation: str = Field(
        default_factory=lambda: os.getenv("PLANNER_ATTN_IMPLEMENTATION", "sdpa").strip() or "sdpa"
    )


settings = Settings()


def ocr_trigger_tags_set() -> set[str]:
    return {t.strip().lower() for t in settings.ocr_trigger_tags.split(",") if t.strip()}
