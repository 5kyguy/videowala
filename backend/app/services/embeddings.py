from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from ..config import settings
from ..gpu_memory import prepare_gpu_for_next_stage, reclaim_gpu_memory
from .ollama_client import ollama_client

logger = logging.getLogger(__name__)


def _patch_qwen2_config_rope_theta_transformers_v5() -> None:
    """
    Alibaba-NLP/gte-Qwen2-7B-instruct uses trust_remote_code modeling that reads `config.rope_theta`.
    In transformers v5+, Qwen2Config only exposes RoPE via `rope_parameters`; without this, loading fails with:
    AttributeError: 'Qwen2Config' object has no attribute 'rope_theta'
    """
    try:
        import transformers
        from packaging import version

        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            return
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    except Exception:
        return
    if getattr(Qwen2Config, "_videowala_rope_theta_compat", False):
        return

    def _get_rope_theta(self) -> float:
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            return float(rp["rope_theta"])
        return 1_000_000.0

    def _set_rope_theta(self, value: float) -> None:
        if not isinstance(getattr(self, "rope_parameters", None), dict):
            self.rope_parameters = {}
        self.rope_parameters["rope_theta"] = float(value)

    Qwen2Config.rope_theta = property(_get_rope_theta, _set_rope_theta)
    Qwen2Config._videowala_rope_theta_compat = True
    logger.info(
        "Applied Qwen2Config.rope_theta compatibility shim for transformers v5+ (gte-Qwen2 hub code)."
    )


def _patch_cache_get_usable_length_compat() -> None:
    """
    Alibaba gte-Qwen2 hub `modeling_qwen.py` calls `past_key_values.get_usable_length(seq_length)`.
    That method was removed from `transformers.cache_utils.Cache` in favor of `get_seq_length` (~4.56+).
    """
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return
    if getattr(Cache, "_videowala_get_usable_length_compat", False):
        return

    def get_usable_length(self, seq_length: int | None = None, layer_idx: int = 0) -> int:
        # Legacy call sites pass (seq_length); ignore it — length comes from cached tensors.
        return self.get_seq_length(layer_idx)

    Cache.get_usable_length = get_usable_length  # type: ignore[assignment]
    Cache._videowala_get_usable_length_compat = True
    logger.info(
        "Applied Cache.get_usable_length compatibility shim for gte-Qwen2 hub modeling (transformers 4.56+)."
    )


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


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


class EmbeddingService:
    def __init__(self) -> None:
        self._model = None
        self._model_id = (
            settings.ollama_embedding_model_id if settings.model_provider == "ollama" else settings.embedding_model_id
        )
        self._load_strategy: str | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def release(self) -> None:
        """Unload embedding weights so other stages can use the GPU."""
        if settings.model_provider == "ollama" and settings.ollama_embedding_model_id:
            try:
                ollama_client.unload(model=settings.ollama_embedding_model_id)
            except Exception:
                pass
        m = self._model
        self._model = None
        self._load_strategy = None
        del m
        reclaim_gpu_memory()

    def _ensure_model(self) -> None:
        if settings.stage2_stub_models:
            return
        if settings.model_provider == "ollama":
            return
        if self._model is not None:
            return
        _patch_qwen2_config_rope_theta_transformers_v5()
        _patch_cache_get_usable_length_compat()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is required when stage2_stub_models=False") from exc
        prepare_gpu_for_next_stage()

        model_id = self._model_id
        last_exc: BaseException | None = None

        def _cap_len() -> None:
            assert self._model is not None
            self._model.max_seq_length = min(getattr(self._model, "max_seq_length", 8192), 8192)

        def _load_cuda_4bit() -> None:
            from transformers import BitsAndBytesConfig

            self._model = SentenceTransformer(
                model_id,
                trust_remote_code=True,
                device="cuda",
                model_kwargs={
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                    "device_map": "auto",
                },
            )
            _cap_len()

        def _load_cuda_default() -> None:
            self._model = SentenceTransformer(
                model_id,
                trust_remote_code=True,
                device="cuda",
            )
            _cap_len()

        def _load_cpu() -> None:
            self._model = SentenceTransformer(
                model_id,
                trust_remote_code=True,
                device="cpu",
            )
            _cap_len()

        attempts: list[tuple[str, Any]] = []
        if torch.cuda.is_available():
            if _bitsandbytes_available():
                attempts.append(("cuda 4-bit NF4", _load_cuda_4bit))
            attempts.append(("cuda (default)", _load_cuda_default))
        attempts.append(("cpu", _load_cpu))

        for label, loader in attempts:
            try:
                loader()
                self._load_strategy = label
                logger.info(
                    "Embedding model ready: %s (%s); install bitsandbytes for 4-bit GPU to save VRAM",
                    model_id,
                    label,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                m = self._model
                self._model = None
                del m
                reclaim_gpu_memory()
                prepare_gpu_for_next_stage()
                if torch.cuda.is_available() and _is_cuda_oom(exc):
                    logger.warning("Embedding load OOM (%s): %s — trying next strategy", label, exc)
                    continue
                if label != "cpu":
                    logger.warning("Embedding load failed (%s): %s — trying next strategy", label, exc)
                    continue
                raise RuntimeError(
                    f"Could not load embedding model {model_id!r}. Last error: {last_exc!r}"
                ) from last_exc

    def embed_text(self, text: str, *, for_query: bool = False) -> EmbeddingResult:
        """
        Embed text for retrieval. Use for_query=True for user search queries (instruction-tuned query prompt);
        for_query=False for indexed document text (captions, ASR, OCR, etc.).
        """
        normalized = " ".join(text.split()) if text else ""
        max_chars = 24000
        if len(normalized) > max_chars:
            normalized = normalized[: max_chars - 1].rstrip() + "…"
        if not normalized:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector("empty"))
        if settings.stage2_stub_models:
            return EmbeddingResult(model=self._model_id, vector=_deterministic_vector(f"stub:{normalized}"))

        if settings.model_provider == "ollama":
            if not settings.ollama_embedding_model_id:
                raise RuntimeError("MODEL_PROVIDER=ollama but OLLAMA_EMBEDDING_MODEL_ID is not set.")
            embeddings = ollama_client.embed(
                model=settings.ollama_embedding_model_id,
                input_texts=[normalized],
                dimensions=settings.embedding_vector_dim,
                truncate=True,
                keep_alive=settings.ollama_keep_alive_stage,
            )
            if not embeddings:
                raise RuntimeError("Ollama embedding response was empty.")
            vec = embeddings[0]
            if settings.embedding_vector_dim and len(vec) != settings.embedding_vector_dim:
                raise RuntimeError(
                    f"Ollama embedding dimension mismatch: expected {settings.embedding_vector_dim}, got {len(vec)}."
                )
            return EmbeddingResult(model=settings.ollama_embedding_model_id, vector=[float(x) for x in vec])

        self._ensure_model()
        assert self._model is not None
        if for_query:
            encoded = self._model.encode(
                [normalized],
                prompt_name="query",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            encoded = self._model.encode(
                [normalized],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        vec = encoded[0]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return EmbeddingResult(model=self._model_id, vector=[float(x) for x in vec])


embedding_service = EmbeddingService()
