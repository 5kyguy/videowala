"""Model-assisted playback order for planner-selected segments (PoC).

When ``stage2_stub_models`` is True (tests / no GPU), uses a deterministic continuity heuristic
instead of loading Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from ..config import settings
from ..gpu_memory import prepare_gpu_for_next_stage, reclaim_gpu_memory

logger = logging.getLogger(__name__)


class PlanSequencerError(Exception):
    """Raised when sequencing output is missing or invalid."""


def continuity_heuristic_order(rows: list[dict]) -> tuple[list[str], str]:
    """
    Reorder segments so all cuts from the same asset are contiguous, preserving first-seen asset order.

    Example: interleaved selection A1, B1, A2, B2 -> A1, A2, B1, B2 (within each asset, sorted by start_s).
    """
    if not rows:
        return [], "continuity_heuristic: empty"
    first_idx: dict[str, int] = {}
    for i, r in enumerate(rows):
        aid = str(r.get("asset_id", ""))
        if aid and aid not in first_idx:
            first_idx[aid] = i
    by_asset: dict[str, list[dict]] = {}
    for r in rows:
        aid = str(r.get("asset_id", ""))
        if not aid:
            continue
        by_asset.setdefault(aid, []).append(r)
    for aid in by_asset:
        by_asset[aid].sort(key=lambda x: float(x.get("start_s", 0.0)))
    ordered_assets = sorted(by_asset.keys(), key=lambda a: first_idx[a])
    out: list[str] = []
    for aid in ordered_assets:
        for r in by_asset[aid]:
            sid = r.get("segment_id")
            if sid:
                out.append(str(sid))
    if len(out) != len(rows):
        raise PlanSequencerError("continuity_heuristic: missing segment_ids")
    return out, "continuity_heuristic: group_by_asset_first_seen_order"


def _clip(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _build_user_prompt(candidates: list[dict], user_prompt: str, *, cue_max: int = 120) -> str:
    lines: list[str] = [
        "You are a video editor. Reorder the clip segments for the best narrative flow and shot continuity.",
        "Prefer grouping all segments from the same source video together (contiguous in timeline order within that video),",
        "unless the user prompt clearly requires interleaving for effect.",
        "",
        f"User prompt / brief: {_clip(user_prompt, 1200)}",
        "",
        "Segments (JSON array of objects). Each object has segment_id, asset_id, start_s, end_s, score, cue (text hints).",
        "Return ONLY a JSON object with exactly these keys:",
        '- "segment_ids": array of strings — a permutation of every segment_id below, in playback order.',
        '- "rationale": short string explaining the ordering.',
        "",
        json.dumps(
            [
                {
                    **{k: v for k, v in c.items() if k != "cue"},
                    "cue": _clip(str(c.get("cue", "")), cue_max),
                }
                for c in candidates
            ],
            ensure_ascii=False,
            indent=2,
        ),
    ]
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        parts = re.split(r"```(?:json)?", text, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p.startswith("{") and "}" in p:
                text = p
                break
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise PlanSequencerError("Model output did not contain a JSON object.")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise PlanSequencerError(f"Invalid JSON from planner model: {exc}") from exc
    if not isinstance(obj, dict):
        raise PlanSequencerError("Model JSON root must be an object.")
    return obj


def _validate_permutation(
    ordered_ids: list[str],
    allowed: set[str],
    candidates: list[dict],
) -> list[str]:
    if not ordered_ids:
        raise PlanSequencerError("segment_ids is empty.")
    seen: set[str] = set()
    out: list[str] = []
    for sid in ordered_ids:
        if sid not in allowed:
            raise PlanSequencerError(f"Unknown segment_id in model output: {sid}")
        if sid in seen:
            raise PlanSequencerError(f"Duplicate segment_id in model output: {sid}")
        seen.add(sid)
        out.append(sid)
    missing = allowed - seen
    if missing:
        logger.warning("Planner model omitted segment_ids %s; appending in candidate order.", missing)
        for c in candidates:
            sid = str(c.get("segment_id", ""))
            if sid in missing:
                out.append(sid)
    return out


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
    return "out of memory" in str(exc).lower()


def _planner_attn_kwargs() -> dict[str, Any]:
    impl = (settings.planner_attn_implementation or "").strip().lower()
    if not impl or impl in ("eager", "none"):
        return {}
    return {"attn_implementation": impl}


def _prompt_token_count(tokenizer: Any, messages: list[dict]) -> tuple[int, str]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(prompt_text)), prompt_text


class PlanSequencerService:
    """Lazy-load Qwen2.5 instruct LM for one-shot segment reordering (quantized / CPU fallbacks)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._effective_model_id: str | None = None

    def release(self) -> None:
        m, t = self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        self._effective_model_id = None
        del m, t
        reclaim_gpu_memory()

    def _drop_model(self) -> None:
        m, t = self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        self._effective_model_id = None
        del m, t
        reclaim_gpu_memory()

    def _ensure_loaded(self) -> None:
        if settings.stage2_stub_models:
            return
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise PlanSequencerError("transformers/torch are required for planner sequencing.") from exc

        prepare_gpu_for_next_stage()

        primary = settings.planner_model_id
        fallback = settings.planner_fallback_model_id
        bnb = _bitsandbytes_available()
        prefer_q = bool(settings.planner_prefer_quantized) and bnb
        attn_kw = _planner_attn_kwargs()

        attempts: list[tuple[str, str, Callable[[], None]]] = []

        def _add_cuda_strategies(hf_id: str, label: str) -> None:
            if not torch.cuda.is_available():
                return

            def load_4bit() -> None:
                from transformers import BitsAndBytesConfig

                self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                q = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    quantization_config=q,
                    device_map="auto",
                    trust_remote_code=True,
                    **attn_kw,
                )

            def load_fp16() -> None:
                self._tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    **attn_kw,
                )

            if prefer_q and bnb:
                attempts.append((hf_id, f"cuda 4-bit NF4 ({label})", load_4bit))
            attempts.append((hf_id, f"cuda fp16 ({label})", load_fp16))
            if not prefer_q and bnb:
                attempts.append((hf_id, f"cuda 4-bit NF4 ({label})", load_4bit))

        _add_cuda_strategies(primary, "primary")
        if fallback != primary:
            _add_cuda_strategies(fallback, "fallback")

        def load_cpu() -> None:
            self._tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                fallback,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                **attn_kw,
            )
            self._model.to("cpu")

        attempts.append((fallback, "cpu fp32 (fallback)", load_cpu))

        last_exc: BaseException | None = None
        for hf_id, label, loader in attempts:
            try:
                reclaim_gpu_memory()
                loader()
                self._effective_model_id = hf_id
                logger.info(
                    "Planner model ready: %s (%s); free VRAM before /requests/render if indexing ran in-process",
                    hf_id,
                    label,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if torch.cuda.is_available() and _is_cuda_oom(exc):
                    logger.warning("Planner load OOM for %s (%s): %s", hf_id, label, exc)
                else:
                    logger.warning("Planner load failed for %s (%s): %s", hf_id, label, exc)
                self._drop_model()
                prepare_gpu_for_next_stage()
                continue

        raise PlanSequencerError(
            f"Could not load any planner checkpoint (primary={primary!r}, fallback={fallback!r}). Last error: {last_exc!r}"
        ) from last_exc

    def sequence_playback_order(self, candidates: list[dict], user_prompt: str) -> tuple[list[str], str]:
        if not candidates:
            return [], "no_candidates"
        if settings.stage2_stub_models:
            return continuity_heuristic_order(candidates)

        self._ensure_loaded()

        allowed = {str(c["segment_id"]) for c in candidates if c.get("segment_id")}
        max_in = settings.planner_max_input_tokens
        max_rows = len(candidates)
        cue_max = 200
        messages: list[dict] = []
        prompt_text: str | None = None
        for _ in range(16):
            user = _build_user_prompt(candidates[:max_rows], user_prompt, cue_max=cue_max)
            messages = [
                {
                    "role": "system",
                    "content": "You reply with a single JSON object only. No markdown fences unless necessary.",
                },
                {"role": "user", "content": user},
            ]
            ntokens, prompt_text = _prompt_token_count(self._tokenizer, messages)
            if ntokens <= max_in:
                break
            if cue_max > 48:
                cue_max = max(48, cue_max - 32)
            elif max_rows > 8:
                max_rows = max(8, max_rows - 8)
            else:
                break
        if prompt_text is None or not messages:
            raise PlanSequencerError("Planner prompt could not be built.")

        try:
            import torch

            inputs = self._tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_in,
                truncation_side="left",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            gen_kw: dict[str, Any] = {
                "max_new_tokens": settings.planner_max_new_tokens,
                "pad_token_id": pad_id,
            }
            if settings.planner_temperature > 1e-6:
                gen_kw["do_sample"] = True
                gen_kw["temperature"] = settings.planner_temperature
            else:
                gen_kw["do_sample"] = False

            max_new = int(settings.planner_max_new_tokens)
            decoded: str | None = None
            last_gen_exc: BaseException | None = None
            prompt_rebuilt_once = False

            def _tokenize_current_prompt() -> Any:
                nonlocal prompt_text
                _, prompt_text = _prompt_token_count(self._tokenizer, messages)
                return self._tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_in,
                    truncation_side="left",
                )

            for gen_try in range(10):
                try:
                    reclaim_gpu_memory()
                    gen_kw["max_new_tokens"] = max_new
                    with torch.no_grad():
                        out_ids = self._model.generate(**inputs, **gen_kw)
                    gen = out_ids[0][inputs["input_ids"].shape[1] :]
                    decoded = self._tokenizer.decode(gen, skip_special_tokens=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_gen_exc = exc
                    if not _is_cuda_oom(exc):
                        raise PlanSequencerError(f"Planner model generation failed: {exc}") from exc
                    logger.warning(
                        "Planner generate CUDA OOM (attempt %s/10); max_new_tokens=%s",
                        gen_try + 1,
                        max_new,
                    )
                    if max_new > 64:
                        max_new = max(64, max_new // 2)
                        continue
                    if not prompt_rebuilt_once and max_rows > 12:
                        prompt_rebuilt_once = True
                        max_rows = max(12, max_rows // 2)
                        cue_max = max(48, cue_max - 40)
                        user = _build_user_prompt(candidates[:max_rows], user_prompt, cue_max=cue_max)
                        messages = [
                            {
                                "role": "system",
                                "content": "You reply with a single JSON object only. No markdown fences unless necessary.",
                            },
                            {"role": "user", "content": user},
                        ]
                        inputs = _tokenize_current_prompt()
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        max_new = min(256, max(96, int(settings.planner_max_new_tokens) // 2))
                        reclaim_gpu_memory()
                        continue
                    raise PlanSequencerError(
                        f"Planner model generation failed (CUDA OOM after retries): {last_gen_exc!r}"
                    ) from last_gen_exc

            if decoded is None:
                raise PlanSequencerError(
                    f"Planner model generation failed after {last_gen_exc!r}"
                ) from last_gen_exc

            model_id_for_note = self._effective_model_id or settings.planner_model_id
        except PlanSequencerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PlanSequencerError(f"Planner model generation failed: {exc}") from exc
        finally:
            self.release()

        data = _extract_json_object(decoded)
        raw_ids = data.get("segment_ids")
        rationale = str(data.get("rationale", "") or "").strip()
        if not isinstance(raw_ids, list):
            raise PlanSequencerError('Model JSON must include "segment_ids" array.')
        ordered = [str(x) for x in raw_ids]
        ordered = _validate_permutation(ordered, allowed, candidates)
        mid = model_id_for_note
        note = f"model={mid}; {rationale}" if rationale else f"model={mid}"
        return ordered, note


plan_sequencer_service = PlanSequencerService()


def sequence_playback_order(candidates: list[dict], user_prompt: str) -> tuple[list[str], str]:
    """Reorder segment dicts (segment_id, asset_id, start_s, end_s, score, cue) for playback."""
    return plan_sequencer_service.sequence_playback_order(candidates, user_prompt)
