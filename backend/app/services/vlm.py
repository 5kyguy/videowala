from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory


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
        timestamps = sorted(
            set(
                [
                    round(min(1.0, duration * 0.05), 3),
                    round(duration * 0.25, 3),
                    round(duration * 0.5, 3),
                    round(duration * 0.75, 3),
                    round(max(0.0, duration - 1.0), 3),
                ]
            )
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for idx, ts in enumerate(timestamps):
        frame_path = scratch_dir / f"{video_path.stem}_frame_{idx:02d}.jpg"
        if _safe_extract_frame(video_path, frame_path, timestamp_s=ts):
            extracted.append(frame_path)
    return extracted


def _parse_json_block(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _normalize_confidence(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v > 1.0:
        v = v / 100.0 if v <= 100.0 else 1.0
    return max(0.0, min(1.0, v))


def _split_predefined_tags(tags: list[str], predefined: set[str]) -> tuple[list[str], list[str]]:
    from_pre: list[str] = []
    added: list[str] = []
    seen: set[str] = set()
    pl = {p.lower() for p in predefined}
    for t in tags:
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        if low in pl:
            from_pre.append(t)
        else:
            added.append(t)
    return from_pre, added


def _build_vlm_prompt(*, predefined_tags: list[str], event_context: dict | None) -> str:
    lines = [
        "You describe event media for a private photo/video library.",
        "Return JSON only with exactly these keys:",
        '- "caption": one vivid sentence about what is happening (people, setting, mood).',
        '- "tags": array of short lowercase tags. Prefer tags from the ALLOWED list when they apply; '
        "you may add new short tags not in the list when needed. "
        "If the scene is sideways because the camera was rotated wrong, add exactly one of: "
        "`needs_rotate_ccw` (rotate 90° counter-clockwise to upright) or `needs_rotate_cw` (rotate 90° clockwise).",
        '- "caption_confidence": your estimated confidence in the caption, number between 0 and 1.',
        "No markdown or code fences.",
        "",
    ]
    if event_context:
        et = event_context.get("event_type") or ""
        title = event_context.get("title") or ""
        venue = event_context.get("venue") or ""
        date = event_context.get("date") or ""
        lines.append("Event context (use to bias wording and tags):")
        lines.append(f"- Title: {title}")
        lines.append(f"- Type: {et}")
        if venue:
            lines.append(f"- Venue: {venue}")
        if date:
            lines.append(f"- Date: {date}")
        lines.append("")
    if predefined_tags:
        lines.append("ALLOWED tags (subset to apply when relevant):")
        lines.append(", ".join(predefined_tags))
        lines.append("")
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


@dataclass(frozen=True)
class VlmResult:
    model: str
    caption: str
    tags: list[str]
    caption_confidence: float | None
    tags_from_predefined: list[str]
    tags_added: list[str]


class VlmService:
    def __init__(self) -> None:
        self._processor = None
        self._model: Any | None = None
        self._model_id = settings.vlm_model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def release(self) -> None:
        """Unload VLM weights so the next serial stage can load a different model."""
        self._model = None
        self._processor = None
        reclaim_gpu_memory()

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

    def caption_and_tags(
        self,
        *,
        media_path: str,
        media_type: str,
        scratch_root: str,
        event_context: dict | None = None,
        predefined_tags: list[str] | None = None,
    ) -> VlmResult:
        p = Path(media_path)
        base = p.stem.replace("_", " ")
        pre = [t.strip() for t in (predefined_tags or []) if t.strip()]
        pre_set = {x.lower() for x in pre}

        if settings.stage2_stub_models:
            st = _stub_tags(base, media_type)
            fp, ad = _split_predefined_tags(st, pre_set)
            return VlmResult(
                model=self._model_id,
                caption=_stub_caption(base),
                tags=st,
                caption_confidence=0.72,
                tags_from_predefined=fp,
                tags_added=ad,
            )

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
            st = _stub_tags(base, media_type)
            fp, ad = _split_predefined_tags(st, pre_set)
            return VlmResult(
                model=self._model_id,
                caption=_stub_caption(base),
                tags=st,
                caption_confidence=None,
                tags_from_predefined=fp,
                tags_added=ad,
            )

        from PIL import Image
        import torch

        device = next(self._model.parameters()).device
        prompt = _build_vlm_prompt(predefined_tags=pre, event_context=event_context)

        # Per-frame candidates: (caption, confidence, tags_list)
        candidates: list[tuple[str, float | None, list[str]]] = []
        all_tags_flat: list[str] = []

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
                generated = self._model.generate(**inputs, max_new_tokens=320, do_sample=False)
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
            conf = _normalize_confidence(payload.get("caption_confidence"))
            tags_raw = payload.get("tags") or []
            frame_tags: list[str] = []
            if isinstance(tags_raw, list):
                frame_tags = [str(t).strip() for t in tags_raw if str(t).strip()]
                all_tags_flat.extend(frame_tags)
            if not frame_caption and cleaned:
                frame_caption = cleaned[:400]
            if frame_caption:
                candidates.append((frame_caption, conf, frame_tags))

        # Pick best frame: highest caption_confidence if any; else longest caption.
        best_caption = ""
        best_conf: float | None = None
        if candidates:
            with_conf = [c for c in candidates if c[1] is not None]
            if with_conf:
                best = max(with_conf, key=lambda x: (x[1] or 0.0, len(x[0])))
            else:
                best = max(candidates, key=lambda x: len(x[0]))
            best_caption, best_conf, _best_frame_tags = best
        caption = best_caption or _stub_caption(base)

        seen: set[str] = set()
        tags: list[str] = []
        for t in all_tags_flat:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                tags.append(t)
        if not tags:
            tags = _stub_tags(base, media_type)
        fp, ad = _split_predefined_tags(tags, pre_set)

        for f in _vlm_scratch_files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        return VlmResult(
            model=self._model_id,
            caption=caption,
            tags=tags[:24],
            caption_confidence=best_conf,
            tags_from_predefined=fp,
            tags_added=ad,
        )


vlm_service = VlmService()
