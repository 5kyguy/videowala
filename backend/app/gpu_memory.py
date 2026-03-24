"""Best-effort GPU memory reclaim between serial model stages (PoC)."""

from __future__ import annotations

import gc


def reclaim_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
