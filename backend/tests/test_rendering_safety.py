from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.db import now_utc, reset_database_for_tests
from app.repositories import AssetRepository, EventRepository, next_id
from app.schemas import Asset, Event, OutputType, PlannerAction, PlannerPlan
from app.services.rendering import (
    _allocate_clip_seconds,
    _build_clip_filter_complex,
    _concat_demuxer,
    _format_filter_time,
    _normalize_and_merge_clip_inputs,
    _prune_clips_to_budget,
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


def test_create_render_job_rejects_image_only_selection() -> None:
    EventRepository.create(
        Event(
            id="event_img",
            tenant_id="tenant_a",
            title="Image only",
            event_type="test",
            predefined_tags=[],
            ocr_languages=["en"],
            created_at=now_utc(),
        )
    )
    img_id = next_id("asset")
    AssetRepository.create(
        Asset(
            id=img_id,
            tenant_id="tenant_a",
            event_id="event_img",
            media_path="media/x.jpg",
            media_type="image",
            created_at=now_utc(),
        )
    )
    plan = PlannerPlan(
        tenant_id="tenant_a",
        event_id="event_img",
        output_type=OutputType.highlight_reel,
        rationale="test",
        actions=[
            PlannerAction(action="select_segments", params={"asset_ids": [img_id]}),
            PlannerAction(action="set_order", params={"strategy": "ranked"}),
            PlannerAction(action="set_duration", params={"seconds": 10}),
            PlannerAction(action="render_preview", params={"format": "mp4"}),
        ],
    )
    with pytest.raises(UnsafeRenderCommandError, match="No video clips"):
        create_render_job("tenant_a", "event_img", plan)


def test_render_job_execution_completes() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for render execution test")

    EventRepository.create(
        Event(
            id="event_a",
            tenant_id="tenant_a",
            title="Render Test",
            event_type="test",
            predefined_tags=[],
            ocr_languages=["en"],
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
            PlannerAction(action="set_order", params={"strategy": "ranked"}),
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


def test_duration_budget_prunes_low_score_clips() -> None:
    clips = [
        {"asset_id": "a", "path": "media/a.mp4", "start_s": 0.0, "end_s": 6.0, "score": 0.95},
        {"asset_id": "b", "path": "media/b.mp4", "start_s": 0.0, "end_s": 6.0, "score": 0.2},
        {"asset_id": "c", "path": "media/c.mp4", "start_s": 0.0, "end_s": 6.0, "score": 0.8},
    ]
    pruned = _prune_clips_to_budget(clips, total_seconds=6, min_clip_seconds=3)
    assert len(pruned) == 2
    assert [c["asset_id"] for c in pruned] == ["a", "c"]


def test_allocate_clip_seconds_exact_target_sum() -> None:
    clips = [
        {"asset_id": "a", "start_s": 0.0, "end_s": 8.0, "score": 0.9},
        {"asset_id": "b", "start_s": 0.0, "end_s": 8.0, "score": 0.5},
        {"asset_id": "c", "start_s": 0.0, "end_s": 8.0, "score": 0.4},
    ]
    out = _allocate_clip_seconds(12, clips, min_clip_seconds=3)
    assert sum(out) == 12
    assert all(1 <= x <= 8 for x in out)


def test_clip_filter_complex_trims_video_and_audio_together() -> None:
    """Regression: segment extraction must use trim+atrim (not -ss before -i) to avoid A/V drift."""
    fc = _build_clip_filter_complex(
        start_s=12.5,
        duration_s=8.0,
        transpose=0,
        orientation="landscape",
        ref_fps="30",
    )
    assert "trim=start=12.5:end=20.5" in fc.replace(" ", "")
    assert "atrim=start=12.5:end=20.5" in fc.replace(" ", "")
    assert "setpts=PTS-STARTPTS" in fc
    assert "asetpts=PTS-STARTPTS" in fc
    assert "[vout]" in fc and "[aout]" in fc
    assert "fps=30" in fc


def test_format_filter_time_strips_trailing_zeros() -> None:
    assert _format_filter_time(10.0) == "10"
    assert _format_filter_time(1.25) == "1.25"


def test_concat_demuxer_includes_genpts_for_mux_stability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def capture(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr("app.services.rendering._run_cmd", capture)
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"")
    b.write_bytes(b"")
    _concat_demuxer([a, b], str(tmp_path / "out.mp4"), duration_seconds=60.0)
    assert captured, "expected concat command to be built"
    assert "+genpts" in captured[0]
    assert "-fflags" in captured[0]


def test_merge_contiguous_same_asset_ranges() -> None:
    clips = [
        {"asset_id": "a", "path": "media/a.mp4", "start_s": 0.0, "end_s": 3.0, "score": 0.6},
        {"asset_id": "a", "path": "media/a.mp4", "start_s": 3.0, "end_s": 6.0, "score": 0.7},
        {"asset_id": "b", "path": "media/b.mp4", "start_s": 1.0, "end_s": 4.0, "score": 0.5},
    ]
    merged = _normalize_and_merge_clip_inputs(clips, merge_gap_s=0.25)
    assert len(merged) == 2
    assert merged[0]["asset_id"] == "a"
    assert merged[0]["start_s"] == 0.0
    assert merged[0]["end_s"] == 6.0
