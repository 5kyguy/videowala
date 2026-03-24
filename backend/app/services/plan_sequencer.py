"""Model-assisted playback order for planner-selected segments (PoC).

When ``stage2_stub_models`` is True (tests / no GPU), uses a deterministic continuity heuristic
instead of loading Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import settings
from ..gpu_memory import reclaim_gpu_memory

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


def _build_user_prompt(candidates: list[dict], user_prompt: str) -> str:
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
        json.dumps(candidates, ensure_ascii=False, indent=2),
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


class PlanSequencerService:
    """Lazy-load Qwen2.5-7B-Instruct for one-shot segment reordering."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
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

        model_id = settings.planner_model_id
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def sequence_playback_order(self, candidates: list[dict], user_prompt: str) -> tuple[list[str], str]:
        if not candidates:
            return [], "no_candidates"
        if settings.stage2_stub_models:
            return continuity_heuristic_order(candidates)

        self._ensure_loaded()

        allowed = {str(c["segment_id"]) for c in candidates if c.get("segment_id")}
        user = _build_user_prompt(candidates, user_prompt)
        messages = [
            {
                "role": "system",
                "content": "You reply with a single JSON object only. No markdown fences unless necessary.",
            },
            {"role": "user", "content": user},
        ]
        try:
            import torch

            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
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
            with torch.no_grad():
                out_ids = self._model.generate(**inputs, **gen_kw)
            gen = out_ids[0][inputs["input_ids"].shape[1] :]
            decoded = self._tokenizer.decode(gen, skip_special_tokens=True)
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
        note = f"model={settings.planner_model_id}; {rationale}" if rationale else f"model={settings.planner_model_id}"
        return ordered, note


plan_sequencer_service = PlanSequencerService()


def sequence_playback_order(candidates: list[dict], user_prompt: str) -> tuple[list[str], str]:
    """Reorder segment dicts (segment_id, asset_id, start_s, end_s, score, cue) for playback."""
    return plan_sequencer_service.sequence_playback_order(candidates, user_prompt)
