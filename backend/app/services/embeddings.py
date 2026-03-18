from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import settings


def _deterministic_vector(seed: str, dim: int = 384) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(dim):
        out.append((digest[i % len(digest)] / 255.0) - 0.5)
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
        normalized = " ".join(text.split())[:8000]
        if not normalized:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector("empty"))
        if settings.stage2_stub_models:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector(f"stub:{normalized}"))
        self._ensure_model()
        vector = self._model.encode([normalized], normalize_embeddings=True)[0].tolist()
        return EmbeddingResult(model=self._model_id, vector=[float(x) for x in vector])


embedding_service = EmbeddingService()
