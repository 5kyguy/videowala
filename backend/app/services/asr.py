from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import subprocess
from pathlib import Path

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory

logger = logging.getLogger(__name__)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def _stub_segments(media_path: str) -> list[dict]:
    digest = hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:8]
    return [{"start": 0.0, "end": 3.2, "text": f"stub_asr_{digest}", "confidence": 0.55}]


def _torch_gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_cuda_runtime_loaded() -> bool:
    """Best-effort loader for CUDA runtime libs required by ctranslate2/faster-whisper."""
    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        pass

    search_roots: list[str] = []
    extra = os.getenv("CUDA_LIBRARY_PATHS", "").strip()
    if extra:
        search_roots.extend([p.strip() for p in extra.split(":") if p.strip()])
    cuda_home = os.getenv("CUDA_HOME", "").strip()
    if cuda_home:
        search_roots.extend([f"{cuda_home}/lib64", f"{cuda_home}/targets/x86_64-linux/lib"])
    search_roots.extend(
        [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-12.0/lib64",
            "/usr/local/cuda-12.1/lib64",
            "/usr/local/cuda-12.2/lib64",
            "/usr/local/cuda-12.3/lib64",
            "/usr/local/cuda-12.4/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ]
    )

    seen: set[str] = set()
    for root in search_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        lib_path = Path(root) / "libcublas.so.12"
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path))
            # Preserve this path for downstream dynamic loads done by model libs.
            current = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{root}:{current}" if current else root
            logger.info("Loaded CUDA runtime library from %s", lib_path)
            return True
        except OSError:
            continue
    return False


class AsrService:
    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None
        # If we attempted CUDA but the CUDA runtime libs aren't loadable (e.g. missing libcublas.so),
        # flip this to avoid repeated failures and keep the indexing job alive.
        self._force_cpu: bool = False

    @property
    def model_name(self) -> str:
        return "faster-whisper (large-v3-turbo)"

    def release(self) -> None:
        # Drop the WhisperModel reference so ctranslate2/CUDA memory can be reclaimed before VLM.
        model = self._model
        self._model = None
        self._device = None
        del model
        reclaim_gpu_memory()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faster-whisper is required when stage2_stub_models=False") from exc
        use_gpu = (not self._force_cpu) and _torch_gpu_available() and _ensure_cuda_runtime_loaded()
        self._device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        try:
            self._model = WhisperModel("large-v3-turbo", device=self._device, compute_type=compute_type)
        except RuntimeError as exc:
            # Handle cases where torch reports CUDA availability but CUDA runtime libs
            # (like libcublas) are missing on the host/container.
            msg = str(exc)
            if self._device == "cuda" and "libcublas" in msg and ("not found" in msg or "cannot be loaded" in msg):
                logger.warning(
                    "CUDA runtime libs missing for faster-whisper (will retry on CPU). error=%s",
                    msg[:500],
                )
                self.release()
                self._force_cpu = True
                self._device = "cpu"
                self._model = WhisperModel("large-v3-turbo", device=self._device, compute_type="int8")
                compute_type = "int8"
            else:
                raise
        logger.info("Whisper model loaded (device=%s, compute_type=%s).", self._device, compute_type)

    def _extract_audio(self, video_path: Path, out_path: Path) -> bool:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "").strip()
            logger.warning(
                "ffmpeg audio extract failed for %s (exit %s): %s",
                video_path.name,
                exc.returncode,
                err[:800] if err else "(no stderr)",
            )
            return False
        return True

    def transcribe(self, media_path: str) -> list[dict]:
        if settings.stage2_stub_models:
            return _stub_segments(media_path)
        target = Path(media_path)
        if not target.exists() or target.suffix.lower() not in VIDEO_EXTS:
            return []

        self._ensure_model()
        assert self._model is not None

        scratch = Path(settings.scratch_root) / "asr"
        wav = scratch / f"{target.stem}.wav"
        if not self._extract_audio(target, wav):
            return []
        try:
            # language=None: auto-detect (English, Hindi, Gujarati, etc.) per clip.
            try:
                segments, info = self._model.transcribe(str(wav), language=None, task="transcribe")
            except RuntimeError as exc:
                # Some CUDA builds of PyTorch/faster-whisper can report CUDA availability even when
                # the CUDA runtime libraries are missing (common on CPU-only hosts).
                # Example: "Library libcublas.so.12 is not found or cannot be loaded"
                msg = str(exc)
                if self._device == "cuda" and "libcublas" in msg and ("not found" in msg or "cannot be loaded" in msg):
                    logger.warning(
                        "CUDA runtime libs missing for faster-whisper (will retry on CPU). error=%s",
                        msg[:500],
                    )
                    self.release()
                    self._force_cpu = True
                    self._ensure_model()
                    assert self._model is not None
                    segments, info = self._model.transcribe(str(wav), language=None, task="transcribe")
                else:
                    raise
            lang = getattr(info, "language", None)
            if lang:
                logger.info("Whisper detected language=%s for %s", lang, target.name)
            out = []
            for seg in segments:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "confidence": 0.6})
            return out
        finally:
            try:
                if wav.exists():
                    wav.unlink()
            except OSError:
                pass


asr_service = AsrService()
