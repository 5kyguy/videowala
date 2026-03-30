from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory
from .ollama_client import ollama_client

logger = logging.getLogger(__name__)


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


def _file_uri(path: Path) -> str:
    """Qwen2.5-VL expects file:// URIs for local images."""
    return path.resolve().as_uri()


@dataclass(frozen=True)
class VlmResult:
    model: str
    caption: str
    tags: list[str]
    caption_confidence: float | None
    tags_from_predefined: list[str]
    tags_added: list[str]


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


class VlmService:
    def __init__(self) -> None:
        self._processor = None
        self._model: Any | None = None
        self._model_id = settings.ollama_vlm_model_id if settings.model_provider == "ollama" else settings.vlm_model_id
        self._effective_model_id: str | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _result_model_id(self) -> str:
        return self._effective_model_id or self._model_id

    def release(self) -> None:
        """Unload VLM weights so the next serial stage can load a different model."""
        if settings.model_provider == "ollama" and settings.ollama_vlm_model_id:
            try:
                ollama_client.unload(model=settings.ollama_vlm_model_id)
            except Exception:
                # Best-effort unload; the indexing pipeline should still proceed.
                pass
        self._model = None
        self._processor = None
        self._effective_model_id = None
        reclaim_gpu_memory()

    def _drop_model(self) -> None:
        self._model = None
        self._processor = None
        self._effective_model_id = None
        reclaim_gpu_memory()

    @staticmethod
    def _load_fp16_cuda(model_id: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.to("cuda")
        model.eval()
        return processor, model

    @staticmethod
    def _load_4bit_cuda(model_id: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return processor, model

    @staticmethod
    def _load_cpu_fp32(model_id: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to("cpu")
        model.eval()
        return processor, model

    def _ensure_loaded(self) -> None:
        if settings.stage2_stub_models:
            return
        if settings.model_provider == "ollama":
            return
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "transformers with Qwen2.5-VL support is required (transformers>=4.48). "
                "Install: pip install 'transformers>=4.48' qwen-vl-utils accelerate"
            ) from exc
        try:
            from qwen_vl_utils import process_vision_info  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install qwen-vl-utils for Qwen2.5-VL: pip install qwen-vl-utils") from exc

        primary = self._model_id
        fallback = settings.vlm_fallback_model_id
        bnb = _bitsandbytes_available()
        prefer_q = bool(settings.vlm_prefer_quantized) and bnb

        attempts: list[tuple[str, str, Any]] = []

        def _add_cuda_strategies(hf_id: str, label: str) -> None:
            """Order: quantized-first (default) or fp16-first, then the other if bnb is available."""
            if not torch.cuda.is_available():
                return
            if prefer_q and bnb:
                attempts.append((hf_id, f"cuda 4-bit NF4 ({label})", self._load_4bit_cuda))
            attempts.append((hf_id, f"cuda fp16 ({label})", self._load_fp16_cuda))
            if not prefer_q and bnb:
                attempts.append((hf_id, f"cuda 4-bit NF4 ({label})", self._load_4bit_cuda))

        _add_cuda_strategies(primary, "primary")
        if fallback != primary:
            _add_cuda_strategies(fallback, "fallback")
        attempts.append((fallback, "cpu fp32 (fallback)", self._load_cpu_fp32))

        last_exc: BaseException | None = None
        for hf_id, label, loader in attempts:
            try:
                reclaim_gpu_memory()
                self._processor, self._model = loader(hf_id)
                self._effective_model_id = hf_id
                logger.info(
                    "VLM ready: model=%s (%s); install bitsandbytes for 4-bit GPU fallbacks",
                    hf_id,
                    label,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if torch.cuda.is_available() and _is_cuda_oom(exc):
                    logger.warning("VLM load OOM for %s (%s): %s", hf_id, label, exc)
                else:
                    logger.warning("VLM load failed for %s (%s): %s", hf_id, label, exc)
                self._drop_model()
                continue

        raise RuntimeError(
            f"Could not load any VLM checkpoint (primary={primary!r}, fallback={fallback!r}). "
            "Try VLM_FALLBACK_MODEL_ID, free GPU memory before VLM, or use a machine with more VRAM. "
            f"Last error: {last_exc!r}"
        ) from last_exc

    def _generate_one_image(
        self,
        *,
        image_uri: str,
        prompt: str,
        device: str,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        import torch

        assert self._model is not None and self._processor is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()

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
                model=self._result_model_id(),
                caption=_stub_caption(base),
                tags=st,
                caption_confidence=0.72,
                tags_from_predefined=fp,
                tags_added=ad,
            )

        if settings.model_provider == "ollama":
            if not settings.ollama_vlm_model_id:
                raise RuntimeError("MODEL_PROVIDER=ollama but OLLAMA_VLM_MODEL_ID is not set.")
        else:
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
                model=self._result_model_id(),
                caption=_stub_caption(base),
                tags=st,
                caption_confidence=None,
                tags_from_predefined=fp,
                tags_added=ad,
            )

        prompt = _build_vlm_prompt(predefined_tags=pre, event_context=event_context)

        candidates: list[tuple[str, float | None, list[str]]] = []
        all_tags_flat: list[str] = []

        if settings.model_provider == "ollama":
            for image_path in image_paths:
                if not image_path.exists():
                    continue
                try:
                    raw = image_path.read_bytes()
                    b64 = base64.b64encode(raw).decode("utf-8")
                    cleaned = ollama_client.generate(
                        model=settings.ollama_vlm_model_id,
                        prompt=prompt,
                        images=[b64],
                        keep_alive=settings.ollama_keep_alive_stage,
                        stream=False,
                        # Ask Ollama to emit JSON so our existing parser works.
                        format="json",
                        options={"temperature": 0.0},
                    )
                except Exception:
                    continue

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
        else:
            import torch

            device = next(self._model.parameters()).device
            for image_path in image_paths:
                if not image_path.exists():
                    continue
                try:
                    uri = _file_uri(image_path)
                    cleaned = self._generate_one_image(image_uri=uri, prompt=prompt, device=str(device))
                except Exception:
                    continue

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

        # Choose best frame caption/tags across extracted frames.
        # This mirrors the HF path logic.
        # (We keep this logic shared between providers to reduce behavioral drift.)
        # candidates is populated by the provider-specific block above.
        # If candidates is empty, we fall back to stub caption/tags as before.

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
            model=self._result_model_id(),
            caption=caption,
            tags=tags[:24],
            caption_confidence=best_conf,
            tags_from_predefined=fp,
            tags_added=ad,
        )


vlm_service = VlmService()
