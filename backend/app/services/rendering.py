from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import settings
from ..db import now_utc
from ..repositories import AssetRepository, InsightRepository, RenderRepository, SegmentRepository, next_id
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
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_clip),
    ]


def _prepare_video_clip_range(input_file: str, output_clip: Path, start_s: float, seconds: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, start_s)),
        "-i",
        input_file,
        "-t",
        str(seconds),
        "-vf",
        TARGET_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
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
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        str(seconds),
        "-vf",
        TARGET_FILTER,
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
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


def _allocate_clip_seconds(total_seconds: int, clip_inputs: list[dict]) -> list[int]:
    if not clip_inputs:
        return []
    scores = [max(0.05, float(item.get("score", 0.5))) for item in clip_inputs]
    total = sum(scores)
    raw = [(score / total) * total_seconds for score in scores]
    out = [max(1, int(x)) for x in raw]
    delta = total_seconds - sum(out)
    idx = 0
    while delta != 0 and out:
        pos = idx % len(out)
        if delta > 0:
            out[pos] += 1
            delta -= 1
        else:
            if out[pos] > 1:
                out[pos] -= 1
                delta += 1
        idx += 1
        if idx > 10000:
            break
    return out


def _continuity_order(clip_inputs: list[dict]) -> list[dict]:
    if not clip_inputs:
        return []
    ordered = sorted(clip_inputs, key=lambda x: (x.get("asset_id", ""), float(x.get("start_s", 0.0))))
    # Avoid immediate duplicates from same source asset when alternatives exist.
    for idx in range(1, len(ordered)):
        prev = ordered[idx - 1].get("asset_id")
        cur = ordered[idx].get("asset_id")
        if prev == cur:
            swap_idx = next((j for j in range(idx + 1, len(ordered)) if ordered[j].get("asset_id") != prev), None)
            if swap_idx is not None:
                ordered[idx], ordered[swap_idx] = ordered[swap_idx], ordered[idx]
    return ordered


def _render_from_inputs(clip_inputs: list[dict], output_file: str, duration_seconds: int, scratch_dir: Path) -> None:
    if not clip_inputs:
        raise UnsafeRenderCommandError("At least one input file is required for rendering.")
    for item in clip_inputs:
        path = str(item.get("path", ""))
        validate_safe_path(path)
    validate_safe_path(output_file)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    ordered_inputs = _continuity_order(clip_inputs)
    allocated_seconds = _allocate_clip_seconds(duration_seconds, ordered_inputs)
    prepared_clips: list[Path] = []

    for index, item in enumerate(ordered_inputs):
        file_path = str(item.get("path", ""))
        source = _resolve_source_path(file_path)
        if not source.exists():
            continue
        clip_path = scratch_dir / f"clip_{index:04d}.mp4"
        seconds_each = allocated_seconds[index] if index < len(allocated_seconds) else _seconds_per_asset(duration_seconds, len(ordered_inputs))
        start_s = float(item.get("start_s", 0.0) or 0.0)
        if source.suffix.lower() in IMAGE_EXTS:
            cmd = _prepare_image_clip(str(source), clip_path, seconds_each)
        else:
            cmd = _prepare_video_clip_range(str(source), clip_path, start_s, seconds_each)
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
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_file,
    ]
    _run_cmd(concat_cmd)


def _burn_subtitles(input_file: Path, output_file: Path, srt_path: Path) -> None:
    srt_escaped = str(srt_path).replace(":", r"\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vf",
        f"subtitles={srt_escaped}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
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
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    _run_cmd(cmd)


def _apply_subs_and_overlays(
    input_file: Path,
    output_file: Path,
    srt_path: Path,
    overlay_items: list[dict],
) -> None:
    """Single-pass ffmpeg invocation that combines subtitles and overlay drawtext filters."""
    srt_escaped = str(srt_path).replace(":", r"\:")
    filter_parts = [f"subtitles={srt_escaped}"]
    y = 40
    for item in overlay_items[:5]:
        t = _safe_drawtext_text(str(item.get("text", "")))
        if not t:
            continue
        filter_parts.append(f"drawtext=text='{t}':x=40:y={y}:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.45")
        y += 44
    vf = ",".join(filter_parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    _run_cmd(cmd)


def create_render_job(tenant_id: str, event_id: str, plan: PlannerPlan) -> RenderJob:
    selected_ids: list[str] = []
    selected_segment_ids: list[str] = []
    duration = 60
    want_subtitles = False
    want_overlays = False
    for action in plan.actions:
        if action.action == "select_segments":
            selected_ids = list(action.params.get("asset_ids", []))
            selected_segment_ids = list(action.params.get("segment_ids", []))
        if action.action == "set_duration":
            duration = int(action.params.get("seconds", 60))
        if action.action == "render_subtitles":
            want_subtitles = True
        if action.action == "render_overlays":
            want_overlays = True

    clip_inputs: list[dict] = []
    if selected_segment_ids:
        for seg_id in selected_segment_ids:
            seg = SegmentRepository.get(seg_id)
            if seg is None:
                continue
            asset = AssetRepository.get(seg.asset_id)
            if asset is None:
                continue
            clip_inputs.append(
                {
                    "segment_id": seg.id,
                    "asset_id": seg.asset_id,
                    "path": asset.media_path,
                    "start_s": seg.start_s,
                    "end_s": seg.end_s,
                    "score": seg.score,
                }
            )

    if not clip_inputs:
        for asset_id in selected_ids:
            asset = AssetRepository.get(asset_id)
            if asset is not None:
                clip_inputs.append(
                    {
                        "segment_id": None,
                        "asset_id": asset.id,
                        "path": asset.media_path,
                        "start_s": 0.0,
                        "end_s": float(duration),
                        "score": 0.5,
                    }
                )
    if not clip_inputs:
        raise UnsafeRenderCommandError("No selected assets found for render.")
    input_files = [str(item.get("path", "")) for item in clip_inputs]
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
        progress_percent=0,
        error_message=None,
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
        clip_inputs=clip_inputs,
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
    RenderRepository.update_status(job_id, "running", progress_percent=5)
    if job.output_path is None:
        job.status = "failed"
        RenderRepository.update_status(job_id, "failed", progress_percent=100, error_message="Missing output path")
        return job

    spec = RenderRepository.get_spec(job_id) or {}
    input_files = list(spec.get("input_files", []))
    clip_inputs = list(spec.get("clip_inputs", []))
    duration_seconds = int(spec.get("duration_seconds", 60))
    scratch_dir = Path(spec.get("scratch_dir", settings.scratch_root)) / job_id
    subtitles_enabled = bool(spec.get("subtitles_enabled", False))
    overlays_enabled = bool(spec.get("overlays_enabled", False))

    try:
        output = Path(job.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        base_output = scratch_dir / "base.mp4"
        RenderRepository.update_status(job_id, "running", progress_percent=25)
        _render_from_inputs(
            clip_inputs=clip_inputs or [{"path": p, "asset_id": "", "start_s": 0.0, "end_s": float(duration_seconds), "score": 0.5} for p in input_files],
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

        has_subs = subtitles_enabled and asr_segments
        has_overlays = overlays_enabled and ocr_items

        if has_subs and has_overlays:
            srt_path = scratch_dir / "subtitles.srt"
            srt_path.write_text(segments_to_srt(asr_segments), encoding="utf-8")
            combined = scratch_dir / "combined.mp4"
            _apply_subs_and_overlays(final_path, combined, srt_path, ocr_items)
            final_path = combined
            audit_action(job.tenant_id, job.event_id, "subtitles_rendered", {"render_job_id": job.id})
            audit_action(job.tenant_id, job.event_id, "overlays_rendered", {"render_job_id": job.id})
            RenderRepository.update_status(job_id, "running", progress_percent=85)
        elif has_subs:
            srt_path = scratch_dir / "subtitles.srt"
            srt_path.write_text(segments_to_srt(asr_segments), encoding="utf-8")
            subbed = scratch_dir / "subbed.mp4"
            _burn_subtitles(final_path, subbed, srt_path)
            final_path = subbed
            audit_action(job.tenant_id, job.event_id, "subtitles_rendered", {"render_job_id": job.id})
            RenderRepository.update_status(job_id, "running", progress_percent=85)
        elif has_overlays:
            overlaid = scratch_dir / "overlaid.mp4"
            _apply_overlays(final_path, overlaid, ocr_items)
            final_path = overlaid
            audit_action(job.tenant_id, job.event_id, "overlays_rendered", {"render_job_id": job.id})
            RenderRepository.update_status(job_id, "running", progress_percent=85)

        shutil.copyfile(final_path, output)
        job.status = "completed"
        job.progress_percent = 100
        RenderRepository.update_status(job_id, "completed", output_path=job.output_path, progress_percent=100, error_message="")
        audit_action(job.tenant_id, job.event_id, "render_job_completed", {"render_job_id": job.id, "output_path": job.output_path})
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error_message = str(exc)
        RenderRepository.update_status(job_id, "failed", progress_percent=100, error_message=str(exc))
        audit_action(job.tenant_id, job.event_id, "render_job_failed", {"render_job_id": job.id, "error": str(exc)})
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
    return job
