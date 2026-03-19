from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from ..config import settings

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


class AsrService:
    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None

    @property
    def model_name(self) -> str:
        return "faster-whisper (large-v3-turbo)"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faster-whisper is required when stage2_stub_models=False") from exc
        use_gpu = _torch_gpu_available()
        self._device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if use_gpu else "int8"
        self._model = WhisperModel("large-v3-turbo", device=self._device, compute_type=compute_type)
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
            segments, _info = self._model.transcribe(str(wav))
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
