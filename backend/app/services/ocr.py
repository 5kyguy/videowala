from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


def _stub_items(media_path: str) -> list[dict]:
    digest = hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:8]
    return [{"text": f"stub_ocr_{digest}", "bbox": [0.1, 0.1, 0.8, 0.2], "confidence": 0.55}]


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


def _paddle_lang_code(ocr_languages: list[str] | None) -> str:
    """Map event hints to PaddleOCR `lang` codes (PP-OCR supports en, ch, hi, ta, te, etc.)."""
    for raw in ocr_languages or ["en"]:
        low = raw.strip().lower().split("-")[0]
        if low in {"en", "ch", "hi", "ta", "te", "ka", "gu", "mr", "bn", "or"}:
            return low
    return "en"


def _preprocess_image(path: Path) -> Path:
    """Light contrast boost; write next to source for Paddle predict."""
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        return path
    try:
        img = Image.open(path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.12)
        img = ImageEnhance.Sharpness(img).enhance(1.05)
        out = path.parent / f"_pre_{path.name}"
        img.save(out, quality=92)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("OCR preprocess skipped for %s: %s", path.name, exc)
        return path


class OcrService:
    """PaddleOCR (PP-OCR) only; language driven by event `ocr_languages`."""

    _paddle_by_lang: dict[str, object] = {}

    @property
    def model_name(self) -> str:
        return "PaddleOCR"

    def _ensure_paddle(self, lang: str) -> object:
        if lang in OcrService._paddle_by_lang:
            return OcrService._paddle_by_lang[lang]
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("paddleocr is required for OCR when stage2_stub_models=False") from exc
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        OcrService._paddle_by_lang[lang] = ocr
        logger.info("PaddleOCR ready (lang=%s).", lang)
        return ocr

    def _extract_image(self, target: Path, *, lang: str) -> tuple[list[dict], str]:
        pre = _preprocess_image(target)
        ocr = self._ensure_paddle(lang)
        label = f"PaddleOCR-{lang}"
        try:
            raw = ocr.predict(
                str(pre),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            items = _paddle_pages_to_items(raw)
        except RuntimeError as exc:
            logger.warning("PaddleOCR inference failed for %s: %s", target.name, exc)
            items = []
        finally:
            if pre != target and pre.exists():
                try:
                    pre.unlink()
                except OSError:
                    pass
        return items, label

    def _sample_video_frames(self, target: Path) -> list[tuple[Path, float]]:
        duration = _video_duration_seconds(target)
        if duration <= 0.0:
            sample_ts = [1.0]
        else:
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

    def _extract_video(self, target: Path, *, lang: str) -> tuple[list[dict], str]:
        frame_samples = self._sample_video_frames(target)
        if not frame_samples:
            return [], "none"

        frame_dir = frame_samples[0][0].parent if frame_samples else None
        merged: list[dict] = []
        model_label = "none"
        try:
            for frame, ts in frame_samples:
                items, label = self._extract_image(frame, lang=lang)
                model_label = label
                for item in items:
                    enriched = dict(item)
                    enriched["time_s"] = round(ts, 3)
                    merged.append(enriched)
        finally:
            if frame_dir is not None and frame_dir.exists():
                shutil.rmtree(frame_dir, ignore_errors=True)
        return merged, model_label

    def extract(
        self,
        media_path: str,
        *,
        run_ocr: bool = True,
        ocr_languages: list[str] | None = None,
    ) -> tuple[list[dict], str]:
        if not run_ocr:
            return [], "skipped-vlm-gate"
        if settings.stage2_stub_models:
            return _stub_items(media_path), "stub-ocr"
        target = Path(media_path)
        if not target.exists():
            return [], "none"
        lang = _paddle_lang_code(ocr_languages)
        if target.suffix.lower() in VIDEO_EXTS:
            return self._extract_video(target, lang=lang)
        return self._extract_image(target, lang=lang)


ocr_service = OcrService()
