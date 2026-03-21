from __future__ import annotations

import pytest

from app.db import now_utc, reset_database_for_tests
from app.repositories import AssetRepository, EventRepository
from app.schemas import ContentRequestCreate, Event, OutputType, PlannerAction, PlannerPlan
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


def test_feedback_include_exclude_affects_plan_selection() -> None:
    _ensure_event()
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
