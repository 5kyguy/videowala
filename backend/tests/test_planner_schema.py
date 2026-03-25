from __future__ import annotations

import pytest

from app.config import settings
from app.db import now_utc, reset_database_for_tests
from app.repositories import AssetRepository, EventRepository, SegmentRepository
from pydantic import ValidationError

from app.schemas import Asset, AssetSegment, ContentRequestCreate, Event, OutputType, PlannerAction, PlannerPlan
from app.services.planner import PlannerValidationError, build_plan, validate_plan


def setup_function() -> None:
    reset_database_for_tests("storage/test_planner.db")


def _ensure_event(event_id: str = "event_a", tenant_id: str = "tenant_a") -> None:
    EventRepository.create(
        Event(
            id=event_id,
            tenant_id=tenant_id,
            title="Test Event",
            event_type="test",
            predefined_tags=[],
            ocr_languages=["en"],
            created_at=now_utc(),
        )
    )


def test_planner_plan_requires_select_and_render() -> None:
    plan = PlannerPlan(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        rationale="test",
        actions=[
            PlannerAction(action="set_duration", params={"seconds": 60}),
        ],
    )
    with pytest.raises(PlannerValidationError):
        validate_plan(plan)


def test_content_request_schema_valid() -> None:
    request = ContentRequestCreate(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        prompt="Create a short preview focused on dancing moments.",
        target_duration_seconds=60,
    )
    assert request.output_type == OutputType.highlight_reel
    assert request.include_media_types == ["video"]


def test_content_request_rejects_image_only_media_types() -> None:
    with pytest.raises(ValidationError):
        ContentRequestCreate(
            tenant_id="tenant_a",
            event_id="event_a",
            output_type=OutputType.highlight_reel,
            prompt="Enough chars for prompt.",
            target_duration_seconds=60,
            include_media_types=["image"],
        )


def test_feedback_include_exclude_affects_plan_selection() -> None:
    _ensure_event()
    for aid in ("a1", "a2", "a3"):
        AssetRepository.create(
            Asset(
                id=aid,
                tenant_id="tenant_a",
                event_id="event_a",
                media_path=f"media/{aid}.mp4",
                media_type="video",
                created_at=now_utc(),
            )
        )
    request = ContentRequestCreate(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        prompt="Regenerate with selected shots.",
        target_duration_seconds=60,
        include_asset_ids=["a2"],
        excluded_asset_ids=["a1"],
    )
    context = {
        "vlm_caption": [{"asset_id": "a1"}, {"asset_id": "a2"}, {"asset_id": "a3"}],
        "vlm_tags": [],
        "face_matches": [],
    }
    plan = build_plan(request, context)
    select_action = next(a for a in plan.actions if a.action == "select_segments")
    selected = select_action.params["asset_ids"]
    assert "a1" not in selected
    assert selected[0] == "a2"


def test_video_orientation_in_render_preview_params() -> None:
    _ensure_event()
    AssetRepository.create(
        Asset(
            id="a1",
            tenant_id="tenant_a",
            event_id="event_a",
            media_path="media/a1.mp4",
            media_type="video",
            created_at=now_utc(),
        )
    )
    request = ContentRequestCreate(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        prompt="Portrait reel output.",
        target_duration_seconds=60,
        video_orientation="portrait",
    )
    context = {"vlm_caption": [{"asset_id": "a1"}], "vlm_tags": [], "face_matches": []}
    plan = build_plan(request, context)
    preview = next(a for a in plan.actions if a.action == "render_preview")
    assert preview.params.get("orientation") == "portrait"


def test_include_media_types_filters_to_video_only() -> None:
    from app.schemas import Asset

    _ensure_event()
    AssetRepository.create(
        Asset(
        id="img1",
        tenant_id="tenant_a",
        event_id="event_a",
        media_path="media/a.jpg",
        media_type="image",
        created_at=now_utc(),
    )
    )
    AssetRepository.create(
        Asset(
        id="vid1",
        tenant_id="tenant_a",
        event_id="event_a",
        media_path="media/b.mp4",
        media_type="video",
        created_at=now_utc(),
    )
    )
    request = ContentRequestCreate(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        prompt="Video-only highlight.",
        target_duration_seconds=60,
        include_media_types=["video"],
    )
    context = {
        "vlm_caption": [{"asset_id": "img1"}, {"asset_id": "vid1"}],
        "vlm_tags": [],
        "face_matches": [],
    }
    plan = build_plan(request, context)
    select_action = next(a for a in plan.actions if a.action == "select_segments")
    selected = select_action.params["asset_ids"]
    assert selected == ["vid1"]


def test_build_plan_stub_sequencer_preserves_shot_continuity() -> None:
    """With stub models, sequencing uses continuity heuristic: group clips per asset."""
    saved_stub = settings.stage2_stub_models
    saved_planner = settings.planner_model_enabled
    settings.stage2_stub_models = True
    settings.planner_model_enabled = True
    try:
        _ensure_event()
        now = now_utc()
        for aid in ("va", "vb"):
            AssetRepository.create(
                Asset(
                    id=aid,
                    tenant_id="tenant_a",
                    event_id="event_a",
                    media_path=f"media/{aid}.mp4",
                    media_type="video",
                    created_at=now,
                )
            )
        SegmentRepository.replace_for_asset(
            "va",
            "event_a",
            [
                AssetSegment(
                    id="seg_va1",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="va",
                    start_s=0.0,
                    end_s=5.0,
                    score=0.95,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
                AssetSegment(
                    id="seg_va2",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="va",
                    start_s=5.0,
                    end_s=10.0,
                    score=0.93,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
            ],
        )
        SegmentRepository.replace_for_asset(
            "vb",
            "event_a",
            [
                AssetSegment(
                    id="seg_vb1",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="vb",
                    start_s=0.0,
                    end_s=5.0,
                    score=0.94,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
                AssetSegment(
                    id="seg_vb2",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="vb",
                    start_s=5.0,
                    end_s=10.0,
                    score=0.92,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
            ],
        )
        request = ContentRequestCreate(
            tenant_id="tenant_a",
            event_id="event_a",
            output_type=OutputType.highlight_reel,
            prompt="Create a short preview focused on dancing moments.",
            target_duration_seconds=60,
        )
        context = {
            "vlm_caption": [{"asset_id": "va", "text": "dance"}, {"asset_id": "vb", "text": "dance"}],
            "vlm_tags": [],
            "face_matches": [],
        }
        plan = build_plan(request, context)
        order_action = next(a for a in plan.actions if a.action == "set_order")
        assert order_action.params.get("strategy") == "preserve_planner"
        select_action = next(a for a in plan.actions if a.action == "select_segments")
        ids = select_action.params["segment_ids"]
        assert ids == ["seg_va1", "seg_va2", "seg_vb1", "seg_vb2"]
    finally:
        settings.stage2_stub_models = saved_stub
        settings.planner_model_enabled = saved_planner


def test_build_plan_planner_disabled_uses_legacy_order_strategy() -> None:
    """When PLANNER_MODEL_ENABLED is false, planner does not reorder segments."""
    saved_stub = settings.stage2_stub_models
    saved_planner = settings.planner_model_enabled
    settings.stage2_stub_models = True
    settings.planner_model_enabled = False
    try:
        _ensure_event()
        now = now_utc()
        for aid in ("va", "vb"):
            AssetRepository.create(
                Asset(
                    id=aid,
                    tenant_id="tenant_a",
                    event_id="event_a",
                    media_path=f"media/{aid}.mp4",
                    media_type="video",
                    created_at=now,
                )
            )
        SegmentRepository.replace_for_asset(
            "va",
            "event_a",
            [
                AssetSegment(
                    id="seg_va1",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="va",
                    start_s=0.0,
                    end_s=5.0,
                    score=0.95,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
                AssetSegment(
                    id="seg_va2",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="va",
                    start_s=5.0,
                    end_s=10.0,
                    score=0.93,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
            ],
        )
        SegmentRepository.replace_for_asset(
            "vb",
            "event_a",
            [
                AssetSegment(
                    id="seg_vb1",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="vb",
                    start_s=0.0,
                    end_s=5.0,
                    score=0.94,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
                AssetSegment(
                    id="seg_vb2",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="vb",
                    start_s=5.0,
                    end_s=10.0,
                    score=0.92,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                ),
            ],
        )
        request = ContentRequestCreate(
            tenant_id="tenant_a",
            event_id="event_a",
            output_type=OutputType.highlight_reel,
            prompt="Create a short preview focused on dancing moments.",
            target_duration_seconds=60,
        )
        context = {
            "vlm_caption": [{"asset_id": "va", "text": "dance"}, {"asset_id": "vb", "text": "dance"}],
            "vlm_tags": [],
            "face_matches": [],
        }
        plan = build_plan(request, context)
        order_action = next(a for a in plan.actions if a.action == "set_order")
        assert order_action.params.get("strategy") == "highlight_diverse"
        select_action = next(a for a in plan.actions if a.action == "select_segments")
        ids = select_action.params["segment_ids"]
        assert ids == ["seg_va1", "seg_vb1", "seg_va2", "seg_vb2"]
    finally:
        settings.stage2_stub_models = saved_stub
        settings.planner_model_enabled = saved_planner


def test_build_plan_applies_duration_aware_segment_cap() -> None:
    saved_duration_cap = settings.planner_duration_aware_cap
    saved_min_clip = settings.planner_min_clip_seconds
    saved_max_seg = settings.planner_max_segments
    settings.planner_duration_aware_cap = True
    settings.planner_min_clip_seconds = 3
    settings.planner_max_segments = 80
    try:
        _ensure_event()
        now = now_utc()
        AssetRepository.create(
            Asset(
                id="asset_many",
                tenant_id="tenant_a",
                event_id="event_a",
                media_path="media/many.mp4",
                media_type="video",
                created_at=now,
            )
        )
        rows: list[AssetSegment] = []
        for i in range(30):
            rows.append(
                AssetSegment(
                    id=f"seg_{i}",
                    tenant_id="tenant_a",
                    event_id="event_a",
                    asset_id="asset_many",
                    start_s=float(i * 6),
                    end_s=float(i * 6 + 6),
                    score=0.95,
                    keep=True,
                    is_duplicate=False,
                    reject_reasons=[],
                    created_at=now,
                )
            )
        SegmentRepository.replace_for_asset("asset_many", "event_a", rows)
        request = ContentRequestCreate(
            tenant_id="tenant_a",
            event_id="event_a",
            output_type=OutputType.highlight_reel,
            prompt="Build a smooth one minute highlight.",
            target_duration_seconds=60,
        )
        context = {"vlm_caption": [{"asset_id": "asset_many", "text": "dance"}], "vlm_tags": [], "face_matches": []}
        plan = build_plan(request, context)
        select_action = next(a for a in plan.actions if a.action == "select_segments")
        segment_ids = select_action.params.get("segment_ids", [])
        assert len(segment_ids) <= 20
    finally:
        settings.planner_duration_aware_cap = saved_duration_cap
        settings.planner_min_clip_seconds = saved_min_clip
        settings.planner_max_segments = saved_max_seg
