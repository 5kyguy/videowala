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


class AsrService:
    @property
    def model_name(self) -> str:
        return "faster-whisper (large-v3-turbo)"

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
        # ASR is intentionally video-only.
        if not target.exists() or target.suffix.lower() not in VIDEO_EXTS:
            return []
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faster-whisper is required when stage2_stub_models=False") from exc

        scratch = Path(settings.scratch_root) / "asr"
        wav = scratch / f"{target.stem}.wav"
        if not self._extract_audio(target, wav):
            return []
        model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(wav))
        out = []
        for seg in segments:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "confidence": 0.6})
        return out


asr_service = AsrService()
