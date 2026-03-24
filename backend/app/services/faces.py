from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def _deterministic_vector(seed: str, size: int = 8) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = []
    for i in range(size):
        values.append(round(digest[i] / 255.0, 6))
    return values


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _first_video_frame_bgr(video_path: Path):
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class FaceService:
    def __init__(self) -> None:
        self._face_analyzer: Any | None = None
        self._model_name = "insightface"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_real_analyzer(self) -> None:
        if not settings.enable_real_face_recognition:
            return
        if self._face_analyzer is not None:
            return
        try:
            from insightface.app import FaceAnalysis

            self._face_analyzer = FaceAnalysis(name="buffalo_l")
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            self._face_analyzer = None

    def release(self) -> None:
        self._face_analyzer = None
        reclaim_gpu_memory()

    def embed_reference(self, image_path: str) -> list[float]:
        target = Path(image_path)
        if not target.exists():
            return _deterministic_vector(f"media:{image_path}")
        self._ensure_real_analyzer()
        if self._face_analyzer is None:
            return _deterministic_vector(f"media:{target.resolve()}")
        try:
            import cv2
        except ImportError:
            return _deterministic_vector(f"media:{target.resolve()}")
        img = cv2.imread(str(target))
        if img is None:
            return _deterministic_vector(f"media:{target.resolve()}")
        faces = self._face_analyzer.get(img)
        if not faces:
            return _deterministic_vector(f"media:{target.resolve()}")
        return faces[0].normed_embedding.tolist()

    def detect_faces(self, media_path: str) -> list[dict]:
        target = Path(media_path)
        if not target.exists():
            return []
        self._ensure_real_analyzer()
        if self._face_analyzer is None:
            confidence = 0.7 if target.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else 0.55
            embedding = _deterministic_vector(f"media:{target.resolve()}")
            return [
                {
                    "bbox": [0.1, 0.1, 0.6, 0.6],
                    "confidence": confidence,
                    "embedding": embedding,
                    "time_range": [0, 5],
                }
            ]

        try:
            import cv2
        except ImportError:
            embedding = _deterministic_vector(f"media:{target.resolve()}")
            return [
                {
                    "bbox": [0.1, 0.1, 0.6, 0.6],
                    "confidence": 0.55,
                    "embedding": embedding,
                    "time_range": [0, 5],
                }
            ]

        if target.suffix.lower() in VIDEO_EXTS:
            frame = _first_video_frame_bgr(target)
            if frame is None:
                return []
            img = frame
            tr = [0.0, 5.0]
        else:
            img = cv2.imread(str(target))
            if img is None:
                return []
            tr = [0.0, 0.0]

        faces = self._face_analyzer.get(img)
        out: list[dict] = []
        for det in faces:
            bbox = det.bbox.astype(float).tolist() if hasattr(det, "bbox") else [0.0, 0.0, 0.0, 0.0]
            score = float(getattr(det, "det_score", 0.0) or 0.0)
            emb = det.normed_embedding.tolist() if hasattr(det, "normed_embedding") else []
            out.append(
                {
                    "bbox": bbox,
                    "confidence": score,
                    "embedding": emb,
                    "time_range": tr,
                }
            )
        return out

    def match_faces(self, detections: list[dict], reference_embeddings: list[dict], threshold: float = 0.8) -> list[dict]:
        matches: list[dict] = []
        for detection in detections:
            emb = detection.get("embedding", [])
            best = None
            best_score = -1.0
            for ref in reference_embeddings:
                score = _cosine_similarity(emb, ref.get("embedding", []))
                if score > best_score:
                    best_score = score
                    best = ref
            if best is None:
                continue
            if best_score >= threshold:
                matches.append(
                    {
                        "person_id": best["person_id"],
                        "name": best.get("display_name", "unknown"),
                        "confidence": round(float(best_score), 4),
                        "time_range": detection.get("time_range", [0, 5]),
                    }
                )
        return matches


face_service = FaceService()
