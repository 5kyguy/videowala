from __future__ import annotations

from app.services.plan_sequencer import continuity_heuristic_order


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
