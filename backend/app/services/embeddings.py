from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory


def _deterministic_vector(seed: str, dim: int | None = None) -> list[float]:
    d = dim if dim is not None else settings.embedding_vector_dim
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(d):
        out.append((digest[i % len(digest)] / 255.0) - 0.5)
    return out


def _clip(text: str, max_len: int) -> str:
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def build_embedding_text(
    *,
    caption: str,
    tags: list[str],
    people_names: list[str],
    asr: str,
    ocr: str,
) -> str:
    """Structured text for semantic embedding: important fields first, then proportional caps."""
    cap_caption = 2000
    cap_tags = 1200
    cap_people = 800
    cap_asr = 6000
    cap_ocr = 4000
    max_total = 12000

    parts: list[str] = []
    c = _clip(caption, cap_caption)
    if c:
        parts.append(f"Caption: {c}")
    if tags:
        parts.append(f"Tags: {_clip(', '.join(tags), cap_tags)}")
    if people_names:
        parts.append(f"People: {_clip(', '.join(people_names), cap_people)}")
    if asr.strip():
        parts.append(f"ASR: {_clip(asr, cap_asr)}")
    if ocr.strip():
        parts.append(f"OCR: {_clip(ocr, cap_ocr)}")

    out = "\n".join(parts)
    if len(out) > max_total:
        out = out[: max_total - 1].rstrip() + "…"
    return out


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    vector: list[float]


class EmbeddingService:
    def __init__(self) -> None:
        self._model = None
        self._model_id = settings.embedding_model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def release(self) -> None:
        self._model = None
        reclaim_gpu_memory()

    def _ensure_model(self) -> None:
        if settings.stage2_stub_models:
            return
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is required when stage2_stub_models=False") from exc
        self._model = SentenceTransformer(self._model_id)

    def embed_text(self, text: str) -> EmbeddingResult:
        # BGE-M3: long context; avoid naive 8k tail-only truncation.
        normalized = " ".join(text.split()) if text else ""
        max_chars = 24000
        if len(normalized) > max_chars:
            normalized = normalized[: max_chars - 1].rstrip() + "…"
        if not normalized:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector("empty"))
        if settings.stage2_stub_models:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector(f"stub:{normalized}"))
        self._ensure_model()
        vector = self._model.encode([normalized], normalize_embeddings=True)[0].tolist()
        return EmbeddingResult(model=self._model_id, vector=[float(x) for x in vector])


embedding_service = EmbeddingService()
