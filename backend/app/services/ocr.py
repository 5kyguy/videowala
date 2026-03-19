from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def _stub_items(media_path: str) -> list[dict]:
    digest = hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:8]
    return [{"text": f"stub_ocr_{digest}", "bbox": [0.1, 0.1, 0.8, 0.2], "confidence": 0.55}]


def _torch_gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _paddle_pages_to_items(raw: list) -> list[dict]:
    items: list[dict] = []
    for page in raw or []:
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys") or []
        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            poly = polys[i] if i < len(polys) else []
            if hasattr(poly, "tolist"):
                bbox = poly.tolist()
            else:
                bbox = list(poly) if poly is not None else []
            items.append({"text": str(text), "bbox": bbox, "confidence": conf})
    return items


def _video_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


class OcrService:
    """Default: EasyOCR via PyTorch (GPU if `torch.cuda.is_available()` — includes ROCm). Optional: PaddleOCR."""

    _easy_reader = None
    _easy_reader_gpu: bool | None = None
    _paddle_ocr = None

    @property
    def model_name(self) -> str:
        return "EasyOCR" if settings.ocr_engine == "easyocr" else "PaddleOCR"

    def _ensure_easy_reader(self) -> None:
        import easyocr

        use_gpu = _torch_gpu_available()
        if OcrService._easy_reader is not None and OcrService._easy_reader_gpu == use_gpu:
            return
        OcrService._easy_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        OcrService._easy_reader_gpu = use_gpu
        logger.info("EasyOCR ready (gpu=%s).", use_gpu)

    def _easyocr_model_label(self) -> str:
        if OcrService._easy_reader_gpu is True:
            return "EasyOCR-GPU"
        if OcrService._easy_reader_gpu is False:
            return "EasyOCR-CPU"
        return "EasyOCR"

    def _easyocr_items(self, target: Path) -> list[dict]:
        try:
            import easyocr  # noqa: F401
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.warning("easyocr / PIL / numpy missing; OCR will be empty.")
            return []

        self._ensure_easy_reader()

        try:
            pil = Image.open(target).convert("RGB")
            w, h = pil.size
            if w < 1 or h < 1:
                return []
            img_np = np.array(pil)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR could not load image %s: %s", target.name, exc)
            return []

        items: list[dict] = []
        try:
            assert OcrService._easy_reader is not None
            for bbox, text, conf in OcrService._easy_reader.readtext(img_np):
                bb = [[float(p[0]), float(p[1])] for p in bbox] if bbox else []
                items.append({"text": str(text), "bbox": bb, "confidence": float(conf)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR failed for %s: %s", target.name, exc)
            return []
        return items

    def _sample_video_frames(self, target: Path) -> list[tuple[Path, float]]:
        duration = _video_duration_seconds(target)
        if duration <= 0.0:
            # Fallback to a single early frame if duration is unknown.
            sample_ts = [1.0]
        else:
            # Light sampling for OCR: start / middle / near-end.
            sample_ts = sorted(
                set(
                    [
                        min(1.0, max(0.0, duration * 0.05)),
                        max(0.0, duration * 0.5),
                        max(0.0, duration * 0.9),
                    ]
                )
            )

        frame_dir = Path(settings.scratch_root) / "ocr" / target.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        out: list[tuple[Path, float]] = []
        for idx, ts in enumerate(sample_ts):
            frame_path = frame_dir / f"frame_{idx:02d}.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts}",
                "-i",
                str(target),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(frame_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ffmpeg frame extraction failed for %s @ %.2fs: %s", target.name, ts, exc)
                continue
            if frame_path.exists():
                out.append((frame_path, ts))
        return out

    def _extract_image(self, target: Path) -> tuple[list[dict], str]:
        if settings.ocr_engine == "easyocr":
            return self._easyocr_items(target), self._easyocr_model_label()

        # Paddle path (optional): one cached instance — avoids reloading PP-OCR models every request.
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("paddleocr is required when OCR_ENGINE=paddle") from exc

        if OcrService._paddle_ocr is None:
            OcrService._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        ocr = OcrService._paddle_ocr
        try:
            raw = ocr.predict(
                str(target),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            return _paddle_pages_to_items(raw), "PaddleOCR"
        except RuntimeError as exc:
            logger.warning(
                "PaddleOCR inference failed for %s (%s); falling back to EasyOCR.",
                target.name,
                exc,
            )
            return self._easyocr_items(target), self._easyocr_model_label()

    def _extract_video(self, target: Path) -> tuple[list[dict], str]:
        frame_samples = self._sample_video_frames(target)
        if not frame_samples:
            return [], "none"

        merged: list[dict] = []
        model_label = "none"
        for frame, ts in frame_samples:
            items, label = self._extract_image(frame)
            model_label = label
            for item in items:
                enriched = dict(item)
                # Preserve timestamp so downstream can decide whether to surface timing.
                enriched["time_s"] = round(ts, 3)
                merged.append(enriched)
        return merged, model_label

    def extract(self, media_path: str) -> tuple[list[dict], str]:
        if settings.stage2_stub_models:
            return _stub_items(media_path), "stub-ocr"
        target = Path(media_path)
        if not target.exists():
            return [], "none"
        if target.suffix.lower() in VIDEO_EXTS:
            return self._extract_video(target)
        return self._extract_image(target)


ocr_service = OcrService()
