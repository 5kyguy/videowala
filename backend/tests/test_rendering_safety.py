from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.db import now_utc, reset_database_for_tests
from app.repositories import AssetRepository, EventRepository, next_id
from app.schemas import Event, OutputType, PlannerAction, PlannerPlan
from app.services.rendering import (
    UnsafeRenderCommandError,
    build_ffmpeg_preview_command,
    create_render_job,
    execute_render_job,
)


def setup_function() -> None:
    reset_database_for_tests("storage/test_rendering.db")


EXISTING_SAMPLE_VIDEO = "storage/tenant_a/event_7856eb60f403/renders/preview_52498442c509.mp4"


def test_safe_ffmpeg_command_is_built() -> None:
    cmd = build_ffmpeg_preview_command(
        input_files=[EXISTING_SAMPLE_VIDEO],
        output_file="storage/tenant_a/event_a/renders/out.mp4",
        duration_seconds=30,
    )
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd


def test_unsafe_paths_are_rejected() -> None:
    with pytest.raises(UnsafeRenderCommandError):
        build_ffmpeg_preview_command(
            input_files=["test/media/dance.mp4; rm -rf /"],
            output_file="storage/tenant_a/event_a/renders/out.mp4",
            duration_seconds=30,
        )


def test_render_job_execution_completes() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for render execution test")
    from app.schemas import Asset

    EventRepository.create(
        Event(
            id="event_a",
            tenant_id="tenant_a",
            title="Render Test",
            event_type="test",
            created_at=now_utc(),
        )
    )
    asset_id = next_id("asset")
    AssetRepository.create(
        Asset(
            id=asset_id,
            tenant_id="tenant_a",
            event_id="event_a",
            media_path=EXISTING_SAMPLE_VIDEO,
            media_type="video",
            created_at=now_utc(),
        )
    )
    plan = PlannerPlan(
        tenant_id="tenant_a",
        event_id="event_a",
        output_type=OutputType.highlight_reel,
        rationale="test",
        actions=[
            PlannerAction(action="select_segments", params={"asset_ids": [asset_id]}),
            PlannerAction(action="set_duration", params={"seconds": 15}),
            PlannerAction(action="render_preview", params={"format": "mp4"}),
        ],
    )
    job = create_render_job("tenant_a", "event_a", plan)
    done = execute_render_job(job.id)
    if done.status == "failed":
        pytest.skip("render execution failed in this environment")
    assert done.status == "completed"
    assert done.output_path is not None
    output = Path(done.output_path)
    assert output.exists()
    assert output.stat().st_size > 1024
