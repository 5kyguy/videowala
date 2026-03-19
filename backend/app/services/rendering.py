from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import settings
from ..db import now_utc
from ..repositories import AssetRepository, InsightRepository, RenderRepository, next_id
from ..schemas import InsightType, PlannerPlan, RenderJob
from .privacy import audit_action
from .subtitles import segments_to_srt


class UnsafeRenderCommandError(ValueError):
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".heif"}
TARGET_FILTER = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def validate_safe_path(path: str) -> None:
    illegal = ["..", ";", "&&", "|", "`", "$("]
    for token in illegal:
        if token in path:
            raise UnsafeRenderCommandError(f"Unsafe token in path: {token}")


def build_ffmpeg_preview_command(input_files: list[str], output_file: str, duration_seconds: int) -> list[str]:
    if not input_files:
        raise UnsafeRenderCommandError("At least one input file is required.")
    for path in input_files:
        validate_safe_path(path)
    validate_safe_path(output_file)

    # MVP command: first input clip truncated to duration.
    return [
        "ffmpeg",
        "-y",
        "-i",
        input_files[0],
        "-t",
        str(duration_seconds),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]


def _seconds_per_asset(total_seconds: int, count: int) -> int:
    if count <= 0:
        return 1
    return max(1, total_seconds // count)


def _prepare_video_clip(input_file: str, output_clip: Path, seconds: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-t",
        str(seconds),
        "-vf",
        TARGET_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_clip),
    ]


def _prepare_image_clip(input_file: str, output_clip: Path, seconds: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        input_file,
        "-t",
        str(seconds),
        "-vf",
        TARGET_FILTER,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_clip),
    ]


def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _resolve_source_path(raw_path: str) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct.resolve()
    project_relative = PROJECT_ROOT / raw_path
    if project_relative.exists():
        return project_relative.resolve()
    return direct


def _render_from_inputs(input_files: list[str], output_file: str, duration_seconds: int, scratch_dir: Path) -> None:
    if not input_files:
        raise UnsafeRenderCommandError("At least one input file is required for rendering.")
    for path in input_files:
        validate_safe_path(path)
    validate_safe_path(output_file)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    seconds_each = _seconds_per_asset(duration_seconds, len(input_files))
    prepared_clips: list[Path] = []

    for index, file_path in enumerate(input_files):
        source = _resolve_source_path(file_path)
        if not source.exists():
            continue
        clip_path = scratch_dir / f"clip_{index:04d}.mp4"
        if source.suffix.lower() in IMAGE_EXTS:
            cmd = _prepare_image_clip(str(source), clip_path, seconds_each)
        else:
            cmd = _prepare_video_clip(str(source), clip_path, seconds_each)
        _run_cmd(cmd)
        prepared_clips.append(clip_path)

    if not prepared_clips:
        raise UnsafeRenderCommandError("No valid media clips were prepared.")

    concat_list = scratch_dir / "concat.txt"
    concat_list.write_text(
        "".join([f"file '{clip.resolve()}'\n" for clip in prepared_clips]),
        encoding="utf-8",
    )
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_file,
    ]
    _run_cmd(concat_cmd)


def _burn_subtitles(input_file: Path, output_file: Path, srt_path: Path) -> None:
    # ffmpeg subtitles filter needs ':' escaped in the path; build it outside the f-string to avoid backslash issues.
    srt_escaped = str(srt_path).replace(":", r"\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vf",
        f"subtitles={srt_escaped}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    _run_cmd(cmd)


def _safe_drawtext_text(text: str) -> str:
    # Escapes for ffmpeg drawtext; keep it conservative.
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", " ")
    )[:120]


def _apply_overlays(input_file: Path, output_file: Path, items: list[dict]) -> None:
    filters: list[str] = []
    y = 40
    for item in items[:5]:
        t = _safe_drawtext_text(str(item.get("text", "")))
        if not t:
            continue
        filters.append(f"drawtext=text='{t}':x=40:y={y}:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.45")
        y += 44
    vf = ",".join(filters) if filters else "null"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    _run_cmd(cmd)


def create_render_job(tenant_id: str, event_id: str, plan: PlannerPlan) -> RenderJob:
    selected_ids: list[str] = []
    duration = 60
    want_subtitles = False
    want_overlays = False
    for action in plan.actions:
        if action.action == "select_segments":
            selected_ids = list(action.params.get("asset_ids", []))
        if action.action == "set_duration":
            duration = int(action.params.get("seconds", 60))
        if action.action == "render_subtitles":
            want_subtitles = True
        if action.action == "render_overlays":
            want_overlays = True

    input_files: list[str] = []
    for asset_id in selected_ids:
        asset = AssetRepository.get(asset_id)
        if asset is not None:
            input_files.append(asset.media_path)
    if not input_files:
        raise UnsafeRenderCommandError("No selected assets found for render.")
    output_dir = Path(settings.storage_root) / tenant_id / event_id / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / f"{next_id('preview')}.mp4")
    cmd = build_ffmpeg_preview_command(input_files=input_files, output_file=output_file, duration_seconds=duration)

    job = RenderJob(
        id=next_id("render"),
        tenant_id=tenant_id,
        event_id=event_id,
        plan_id=next_id("inline_plan"),
        status="queued",
        output_path=output_file,
        created_at=now_utc(),
    )
    scratch_dir = Path(settings.scratch_root) / tenant_id / event_id / "renders" / job.id
    RenderRepository.create(
        job,
        input_files=input_files,
        duration_seconds=duration,
        scratch_dir=str(scratch_dir),
        subtitles_enabled=want_subtitles,
        overlays_enabled=want_overlays,
    )
    audit_action(tenant_id, event_id, "render_job_created", {"render_job_id": job.id, "ffmpeg_cmd": cmd})
    if want_subtitles:
        audit_action(tenant_id, event_id, "render_job_subtitles_requested", {"render_job_id": job.id})
    if want_overlays:
        audit_action(tenant_id, event_id, "render_job_overlays_requested", {"render_job_id": job.id})
    return job


def execute_render_job(job_id: str) -> RenderJob:
    job = RenderRepository.get(job_id)
    if job is None:
        raise KeyError(f"Render job not found: {job_id}")
    job.status = "running"
    RenderRepository.update_status(job_id, "running")
    if job.output_path is None:
        job.status = "failed"
        RenderRepository.update_status(job_id, "failed")
        return job

    spec = RenderRepository.get_spec(job_id) or {}
    input_files = list(spec.get("input_files", []))
    duration_seconds = int(spec.get("duration_seconds", 60))
    scratch_dir = Path(spec.get("scratch_dir", settings.scratch_root)) / job_id
    subtitles_enabled = bool(spec.get("subtitles_enabled", False))
    overlays_enabled = bool(spec.get("overlays_enabled", False))

    try:
        output = Path(job.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        base_output = scratch_dir / "base.mp4"
        _render_from_inputs(
            input_files=input_files,
            output_file=str(base_output),
            duration_seconds=duration_seconds,
            scratch_dir=scratch_dir,
        )

        # Stage 2 post-processing: subtitles and OCR overlays are derived from insights.
        final_path = base_output
        # Stage2 flags are persisted in render_specs for restart-safety.
        insights = InsightRepository.list_for_event(job.event_id)
        asr_segments: list[dict] = []
        ocr_items: list[dict] = []
        for ins in insights:
            if ins.insight_type == InsightType.asr_transcript and isinstance(ins.payload, dict):
                asr_segments.extend(list(ins.payload.get("segments", [])))
            if ins.insight_type == InsightType.ocr_text and isinstance(ins.payload, dict):
                ocr_items.extend(list(ins.payload.get("items", [])))

        if subtitles_enabled and asr_segments:
            srt_path = scratch_dir / "subtitles.srt"
            srt_path.write_text(segments_to_srt(asr_segments), encoding="utf-8")
            subbed = scratch_dir / "subbed.mp4"
            _burn_subtitles(final_path, subbed, srt_path)
            final_path = subbed
            audit_action(job.tenant_id, job.event_id, "subtitles_rendered", {"render_job_id": job.id})

        if overlays_enabled and ocr_items:
            overlaid = scratch_dir / "overlaid.mp4"
            _apply_overlays(final_path, overlaid, ocr_items)
            final_path = overlaid
            audit_action(job.tenant_id, job.event_id, "overlays_rendered", {"render_job_id": job.id})

        shutil.copyfile(final_path, output)
        job.status = "completed"
        RenderRepository.update_status(job_id, "completed", output_path=job.output_path)
        audit_action(job.tenant_id, job.event_id, "render_job_completed", {"render_job_id": job.id, "output_path": job.output_path})
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        RenderRepository.update_status(job_id, "failed")
        audit_action(job.tenant_id, job.event_id, "render_job_failed", {"render_job_id": job.id, "error": str(exc)})
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
    return job
