from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import settings


def _stub_items(media_path: str) -> list[dict]:
    digest = hashlib.sha256(media_path.encode("utf-8")).hexdigest()[:8]
    return [{"text": f"stub_ocr_{digest}", "bbox": [0.1, 0.1, 0.8, 0.2], "confidence": 0.55}]


class OcrService:
    @property
    def model_name(self) -> str:
        return "PaddleOCR"

    def extract(self, media_path: str) -> list[dict]:
        if settings.stage2_stub_models:
            return _stub_items(media_path)
        target = Path(media_path)
        if not target.exists():
            return []
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("paddleocr is required when stage2_stub_models=False") from exc

        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        # PaddleOCR expects images; for videos, Stage2 should sample frames (not yet implemented).
        if target.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            return _stub_items(media_path)
        result = ocr.ocr(str(target), cls=True)
        items: list[dict] = []
        for line in result or []:
            for bbox, (text, conf) in line:
                items.append({"text": text, "bbox": bbox, "confidence": float(conf)})
        return items


ocr_service = OcrService()
