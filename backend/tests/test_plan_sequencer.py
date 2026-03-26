from __future__ import annotations

from app.services.plan_sequencer import (
    _enforce_asset_contiguity,
    _extract_json_object,
    _prompt_requests_interleaving,
    _validate_permutation,
    continuity_heuristic_order,
)


def test_continuity_heuristic_groups_interleaved_assets() -> None:
    rows = [
        {"segment_id": "s_a1", "asset_id": "va", "start_s": 0.0, "end_s": 1.0},
        {"segment_id": "s_b1", "asset_id": "vb", "start_s": 0.0, "end_s": 1.0},
        {"segment_id": "s_a2", "asset_id": "va", "start_s": 2.0, "end_s": 3.0},
        {"segment_id": "s_b2", "asset_id": "vb", "start_s": 2.0, "end_s": 3.0},
    ]
    ordered, note = continuity_heuristic_order(rows)
    assert ordered == ["s_a1", "s_a2", "s_b1", "s_b2"]
    assert "continuity_heuristic" in note


def test_extract_json_object_raw_decode_with_trailing_prose() -> None:
    text = '{"segment_ids": ["a", "b"], "rationale": "ok"} Here is extra commentary.'
    assert _extract_json_object(text)["segment_ids"] == ["a", "b"]


def test_extract_json_object_markdown_fence() -> None:
    text = '```json\n{"segment_ids": ["z"], "rationale": ""}\n```'
    assert _extract_json_object(text)["segment_ids"] == ["z"]


def test_extract_json_object_root_array() -> None:
    text = '["s1", "s2"]'
    out = _extract_json_object(text)
    assert out["segment_ids"] == ["s1", "s2"]
    assert out["rationale"] == ""


def test_extract_json_object_truncated_mid_array() -> None:
    """Generation can stop before closing ``]`` / ``}``; recover complete quoted ids + trailing fragment."""
    text = """{
  "segment_ids": [
    "seg_a",
    "seg_b",
    "seg_0ce185bc5133"""
    out = _extract_json_object(text)
    assert out["segment_ids"] == ["seg_a", "seg_b", "seg_0ce185bc5133"]


def test_prompt_requests_interleaving_detects_montage_terms() -> None:
    assert _prompt_requests_interleaving("Make a rapid-cut montage with intercut visuals")
    assert not _prompt_requests_interleaving("Keep the story smooth and chronological")


def test_validate_permutation_resolves_truncated_id_prefix() -> None:
    """Planner output can be cut mid-UUID; unique ``seg_…`` prefix maps to the full segment_id."""
    candidates = [
        {"segment_id": "seg_8d2a1b3c4d5e6f7", "asset_id": "a"},
        {"segment_id": "seg_9zzz", "asset_id": "b"},
    ]
    allowed = {c["segment_id"] for c in candidates}
    out = _validate_permutation(["seg_9zzz", "seg_8d2"], allowed, candidates)
    assert out == ["seg_9zzz", "seg_8d2a1b3c4d5e6f7"]


def test_validate_permutation_dedupes_prefix_then_full_same_id() -> None:
    """Truncated fragment and full id both present → one canonical id; keep first in order."""
    candidates = [
        {"segment_id": "seg_8d219907e8f7", "asset_id": "a"},
        {"segment_id": "seg_other", "asset_id": "b"},
    ]
    allowed = {c["segment_id"] for c in candidates}
    full = "seg_8d219907e8f7"
    out = _validate_permutation(["seg_other", "seg_8d2", full], allowed, candidates)
    assert out == ["seg_other", full]


def test_enforce_asset_contiguity_groups_by_first_seen_asset() -> None:
    candidates = [
        {"segment_id": "a1", "asset_id": "asset_a", "start_s": 0.0},
        {"segment_id": "b1", "asset_id": "asset_b", "start_s": 0.0},
        {"segment_id": "a2", "asset_id": "asset_a", "start_s": 5.0},
        {"segment_id": "b2", "asset_id": "asset_b", "start_s": 5.0},
    ]
    ordered = _enforce_asset_contiguity(["a1", "b1", "a2", "b2"], candidates)
    assert ordered == ["a1", "a2", "b1", "b2"]
