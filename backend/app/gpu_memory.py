"""Best-effort GPU memory reclaim between serial model stages (PoC)."""

from __future__ import annotations

import gc


def reclaim_gpu_memory() -> None:
    """Synchronize, drop allocator caches, and run GC (helps after releasing large models)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def prepare_gpu_for_next_stage() -> None:
    """
    Call between serial GPU-heavy stages (e.g. ASR → VLM → embeddings).
    Some backends (faster-whisper/ctranslate2) free VRAM lazily; a second pass helps.
    """
    reclaim_gpu_memory()
    reclaim_gpu_memory()
