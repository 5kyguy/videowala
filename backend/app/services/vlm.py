from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings


def _stub_caption(name: str) -> str:
    return f"Media summary for {name}. This is a DEV_MODE stub caption."


def _stub_tags(base: str, media_type: str) -> list[str]:
    tags = ["event", media_type]
    b = base.lower()
    if "dance" in b:
        tags.extend(["performance", "group"])
    if "ride" in b:
        tags.extend(["outdoor", "motion"])
    return tags


def _safe_extract_frame(video_path: Path, out_path: Path, *, timestamp_s: float = 1.0) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp_s),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path.exists()
    except Exception:
        return False


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


def _extract_representative_frames(video_path: Path, scratch_dir: Path) -> list[Path]:
    """Extract 3-5 evenly-spaced frames for better video understanding."""
    duration = _video_duration_seconds(video_path)
    if duration <= 0:
        timestamps = [1.0]
    elif duration <= 10:
        timestamps = [min(1.0, duration * 0.5)]
    else:
        timestamps = sorted(set([
            round(min(1.0, duration * 0.05), 3),
            round(duration * 0.25, 3),
            round(duration * 0.5, 3),
            round(duration * 0.75, 3),
            round(max(0.0, duration - 1.0), 3),
        ]))

    scratch_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for idx, ts in enumerate(timestamps):
        frame_path = scratch_dir / f"{video_path.stem}_frame_{idx:02d}.jpg"
        if _safe_extract_frame(video_path, frame_path, timestamp_s=ts):
            extracted.append(frame_path)
    return extracted


def _parse_json_block(text: str) -> dict | None:
    # Try to find a JSON object in a model response.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass(frozen=True)
class VlmResult:
    model: str
    caption: str
    tags: list[str]


class VlmService:
    def __init__(self) -> None:
        self._processor = None
        self._model: Any | None = None
        self._model_id = settings.vlm_model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def _ensure_loaded(self) -> None:
        if settings.stage2_stub_models:
            return
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers + torch are required for VLM inference") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        try:
            # SmolVLM support moved across Transformers releases; try both common auto-classes.
            from transformers import AutoModelForVision2Seq

            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as first_exc:  # noqa: BLE001
            try:
                from transformers import AutoModelForImageTextToText

                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_id,
                    dtype=dtype,
                    trust_remote_code=True,
                )
            except Exception as second_exc:  # noqa: BLE001
                tv = getattr(transformers, "__version__", "unknown")
                raise RuntimeError(
                    "Failed to load VLM model. For SmolVLM2 use a recent transformers version (>=4.48 recommended). "
                    f"Current transformers={tv}, model_id={self._model_id}. "
                    f"Original errors: {first_exc} | {second_exc}"
                ) from second_exc
        self._model.to(device)
        self._model.eval()

    def caption_and_tags(self, *, media_path: str, media_type: str, scratch_root: str) -> VlmResult:
        p = Path(media_path)
        base = p.stem.replace("_", " ")
        if settings.stage2_stub_models:
            return VlmResult(model=self._model_id, caption=_stub_caption(base), tags=_stub_tags(base, media_type))

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        image_paths: list[Path] = []
        _vlm_scratch_files: list[Path] = []
        if media_type == "image":
            image_paths = [p]
        elif media_type == "video":
            vlm_scratch_dir = Path(scratch_root) / "vlm"
            frames = _extract_representative_frames(p, vlm_scratch_dir)
            _vlm_scratch_files.extend(frames)
            image_paths = frames

        if not image_paths:
            for f in _vlm_scratch_files:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            return VlmResult(model=self._model_id, caption=_stub_caption(base), tags=_stub_tags(base, media_type))

        from PIL import Image
        import torch

        device = next(self._model.parameters()).device
        prompt = (
            "Return JSON only with keys:\n"
            '- "caption": one sentence describing the image\n'
            '- "tags": an array of 5-12 short tags\n'
            "No markdown.\n"
        )

        best_caption = ""
        all_tags: list[str] = []
        for image_path in image_paths:
            if not image_path.exists():
                continue
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            try:
                inputs = self._processor(images=img, text=prompt, return_tensors="pt")
            except ValueError as exc:
                if "number of images in the text" not in str(exc).lower():
                    raise
                inputs = self._processor(images=img, text=f"<image>\n{prompt}", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=160, do_sample=False)
            decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

            cleaned = decoded.strip()
            if cleaned.startswith(prompt.strip()):
                cleaned = cleaned[len(prompt.strip()) :].lstrip()
            for marker in ("Assistant:", "assistant:", "ASSISTANT:"):
                if marker in cleaned:
                    cleaned = cleaned.split(marker, 1)[1].strip()
                    break

            payload = _parse_json_block(cleaned) or {}
            frame_caption = str(payload.get("caption") or "").strip()
            tags_raw = payload.get("tags") or []
            if isinstance(tags_raw, list):
                all_tags.extend([str(t).strip() for t in tags_raw if str(t).strip()])
            if frame_caption and len(frame_caption) > len(best_caption):
                best_caption = frame_caption
            elif not best_caption and cleaned:
                best_caption = cleaned[:300]

        caption = best_caption or _stub_caption(base)
        seen: set[str] = set()
        tags: list[str] = []
        for t in all_tags:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                tags.append(t)
        if not tags:
            tags = _stub_tags(base, media_type)
        for f in _vlm_scratch_files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        return VlmResult(model=self._model_id, caption=caption, tags=tags[:16])


vlm_service = VlmService()

