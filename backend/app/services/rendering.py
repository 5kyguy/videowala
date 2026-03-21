from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..config import settings
from ..db import now_utc
from ..repositories import AssetRepository, InsightRepository, RenderRepository, SegmentRepository, next_id
from ..schemas import Asset, InsightType, PlannerPlan, RenderJob
from .privacy import audit_action

logger = logging.getLogger(__name__)


class UnsafeRenderCommandError(ValueError):
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".heif"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENCODE_PRESET = "medium"
_ENCODE_CRF = "18"
_DEFAULT_IMAGE_FPS = "30"


def validate_safe_path(path: str) -> None:
    illegal = ["..", ";", "&&", "|", "`", "$("]
    for token in illegal:
        if token in path:
            raise UnsafeRenderCommandError(f"Unsafe token in path: {token}")


def build_ffmpeg_preview_command(input_files: list[str], output_file: str, duration_seconds: int) -> list[str]:
    """Minimal ffmpeg command for tests; full renders use the orientation crop pipeline."""
    if not input_files:
        raise UnsafeRenderCommandError("At least one input file is required.")
    for path in input_files:
        validate_safe_path(path)
    validate_safe_path(output_file)

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


def _seconds_per_asset(total_seconds: int, count: int) -> int:
    if count <= 0:
        return 1
    return max(1, total_seconds // count)


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
    for idx in range(1, len(ordered)):
        prev = ordered[idx - 1].get("asset_id")
        cur = ordered[idx].get("asset_id")
        if prev == cur:
            swap_idx = next((j for j in range(idx + 1, len(ordered)) if ordered[j].get("asset_id") != prev), None)
            if swap_idx is not None:
                ordered[idx], ordered[swap_idx] = ordered[swap_idx], ordered[idx]
    return ordered


def _order_clip_inputs(clip_inputs: list[dict], strategy: str, asset_by_id: dict[str, Asset]) -> list[dict]:
    if strategy in (
        "chronological_capture",
        "highlight_diverse",
        "person_focus",
        "ranked",
        "preserve_planner",
    ):
        return list(clip_inputs)
    if strategy == "chronological":
        return _continuity_order(clip_inputs)
    return sorted(
        clip_inputs,
        key=lambda x: (
            asset_by_id[x["asset_id"]].created_at.isoformat() if x.get("asset_id") in asset_by_id else "",
            float(x.get("start_s", 0.0)),
        ),
    )


def _ffprobe_json(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout)


def _first_video_stream(data: dict) -> dict | None:
    for s in data.get("streams") or []:
        if s.get("codec_type") == "video":
            return s
    return None


def _video_avg_frame_rate(path: Path) -> str | None:
    try:
        data = _ffprobe_json(path)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None
    vs = _first_video_stream(data)
    if not vs:
        return None
    rate = str(vs.get("avg_frame_rate") or "")
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            v = float(a) / float(b)
            if v <= 0:
                return None
            s = f"{v:.6f}".rstrip("0").rstrip(".")
            return s if s else None
        except (ValueError, ZeroDivisionError):
            return None
    return rate


def _video_dimensions(path: Path) -> tuple[int, int]:
    try:
        data = _ffprobe_json(path)
        vs = _first_video_stream(data)
        if not vs:
            return 0, 0
        return int(vs.get("width", 0)), int(vs.get("height", 0))
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, ValueError, TypeError):
        return 0, 0


def _transpose_from_tags(tags: list[str]) -> int:
    """Return ffmpeg transpose argument: 0=none, 1=90° clockwise, 2=90° counter-clockwise."""
    for raw in tags:
        t = raw.lower().replace(" ", "_").replace("-", "_")
        if any(
            x in t
            for x in (
                "needs_rotate_ccw",
                "rotate_left",
                "rotation_90",
                "sideways_left",
                "counter_clockwise",
                "counterclockwise",
            )
        ):
            return 2
        if any(
            x in t
            for x in (
                "needs_rotate_cw",
                "rotate_right",
                "rotation_270",
                "sideways_right",
            )
        ):
            return 1
        if "clockwise" in t and "counter" not in t:
            return 1
    return 0


def _latest_vlm_tags_by_asset(event_id: str) -> dict[str, list[str]]:
    insights = InsightRepository.list_for_event(event_id, insight_type=InsightType.vlm_tags.value)
    ordered = sorted(insights, key=lambda x: x.created_at, reverse=True)
    out: dict[str, list[str]] = {}
    for ins in ordered:
        if ins.asset_id in out:
            continue
        pl = ins.payload if isinstance(ins.payload, dict) else {}
        tags = pl.get("tags") or []
        out[ins.asset_id] = [str(x) for x in tags] if isinstance(tags, list) else []
    return out


def _crop_vf_for_orientation(orientation: str) -> str:
    o = orientation.lower()
    if o in ("portrait", "standing", "vertical", "reel", "reels"):
        return (
            "crop=trunc(min(iw\\,ih*9/16)/2)*2:trunc(min(ih\\,iw*16/9)/2)*2:"
            "(iw-trunc(min(iw\\,ih*9/16)/2)*2)/2:(ih-trunc(min(ih\\,iw*16/9)/2)*2)/2"
        )
    return (
        "crop=trunc(min(iw\\,ih*16/9)/2)*2:trunc(min(ih\\,iw*9/16)/2)*2:"
        "(iw-trunc(min(iw\\,ih*16/9)/2)*2)/2:(ih-trunc(min(ih\\,iw*9/16)/2)*2)/2"
    )


def _build_transpose_and_crop_vf(transpose: int, orientation: str) -> str:
    parts: list[str] = []
    if transpose == 1:
        parts.append("transpose=1")
    elif transpose == 2:
        parts.append("transpose=2")
    elif transpose not in (0, None):
        logger.warning("Unknown display_transpose %s; ignoring.", transpose)
    parts.append(_crop_vf_for_orientation(orientation))
    return ",".join(parts)


def _encode_video_audio_args() -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        _ENCODE_PRESET,
        "-crf",
        _ENCODE_CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ]


def _prepare_video_clip_range(
    input_file: str,
    output_clip: Path,
    start_s: float,
    seconds: int,
    *,
    vf: str,
) -> list[str]:
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
        vf,
        *_encode_video_audio_args(),
        str(output_clip),
    ]


def _prepare_image_clip(
    input_file: str,
    output_clip: Path,
    seconds: int,
    *,
    vf: str,
    framerate: str,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        framerate,
        "-i",
        input_file,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t",
        str(seconds),
        "-vf",
        vf,
        "-shortest",
        *_encode_video_audio_args(),
        str(output_clip),
    ]


def _pad_to_canvas(input_path: Path, output_path: Path, target_w: int, target_h: int) -> list[str]:
    vf = f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        *_encode_video_audio_args(),
        "-c:a",
        "copy",
        str(output_path),
    ]


def _concat_demuxer(prepared_clips: list[Path], output_file: str) -> None:
    scratch_dir = prepared_clips[0].parent
    concat_list = scratch_dir / "concat.txt"
    concat_list.write_text(
        "".join([f"file '{clip.resolve()}'\n" for clip in prepared_clips]),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        *_encode_video_audio_args(),
        "-movflags",
        "+faststart",
        output_file,
    ]
    _run_cmd(cmd)


def _reference_video_fps(ordered_inputs: list[dict]) -> str:
    for item in ordered_inputs:
        p = _resolve_source_path(str(item.get("path", "")))
        if p.suffix.lower() in IMAGE_EXTS or not p.exists():
            continue
        fps = _video_avg_frame_rate(p)
        if fps:
            return fps
    return _DEFAULT_IMAGE_FPS


def _orientation_from_plan(plan: PlannerPlan) -> str:
    for action in plan.actions:
        if action.action != "render_preview":
            continue
        o = str(action.params.get("orientation", "landscape")).lower()
        if o in ("portrait", "standing", "vertical", "reel", "reels"):
            return "portrait"
        return "landscape"
    return "landscape"


def _render_from_inputs(
    clip_inputs: list[dict],
    output_file: str,
    duration_seconds: int,
    scratch_dir: Path,
    *,
    job_id: str | None = None,
    render_options: dict | None = None,
) -> None:
    if not clip_inputs:
        raise UnsafeRenderCommandError("At least one input file is required for rendering.")
    for item in clip_inputs:
        path = str(item.get("path", ""))
        validate_safe_path(path)
    validate_safe_path(output_file)

    ro = render_options or {}
    strategy = str(ro.get("order_strategy", "chronological"))
    orientation = str(ro.get("orientation", "landscape")).lower()
    if orientation in ("standing", "vertical", "reel", "reels"):
        orientation = "portrait"

    scratch_dir.mkdir(parents=True, exist_ok=True)

    asset_by_id: dict[str, Asset] = {}
    for c in clip_inputs:
        aid = c.get("asset_id")
        if aid and aid not in asset_by_id:
            a = AssetRepository.get(str(aid))
            if a:
                asset_by_id[str(aid)] = a

    ordered_inputs = _order_clip_inputs(clip_inputs, strategy, asset_by_id)
    allocated_seconds = _allocate_clip_seconds(duration_seconds, ordered_inputs)
    ref_fps = _reference_video_fps(ordered_inputs)

    prepared_raw: list[Path] = []
    n_clips = len(ordered_inputs)
    for index, item in enumerate(ordered_inputs):
        file_path = str(item.get("path", ""))
        source = _resolve_source_path(file_path)
        if not source.exists():
            continue
        clip_path = scratch_dir / f"clip_{index:04d}_raw.mp4"
        seconds_each = (
            allocated_seconds[index]
            if index < len(allocated_seconds)
            else _seconds_per_asset(duration_seconds, len(ordered_inputs))
        )
        start_s = float(item.get("start_s", 0.0) or 0.0)
        transpose = int(item.get("display_transpose", 0) or 0)
        vf = _build_transpose_and_crop_vf(transpose, orientation)
        if source.suffix.lower() in IMAGE_EXTS:
            cmd = _prepare_image_clip(str(source), clip_path, seconds_each, vf=vf, framerate=ref_fps)
        else:
            cmd = _prepare_video_clip_range(str(source), clip_path, start_s, seconds_each, vf=vf)
        logger.info("Render %s: encoding clip %d/%d from %s", job_id or "?", index + 1, n_clips, source.name)
        _run_cmd(cmd)
        prepared_raw.append(clip_path)
        if job_id and n_clips > 0:
            pct = 25 + int(((index + 1) / n_clips) * 40)
            RenderRepository.update_status(job_id, "running", progress_percent=min(65, pct))

    if not prepared_raw:
        raise UnsafeRenderCommandError("No valid media clips were prepared.")

    if job_id:
        RenderRepository.update_status(job_id, "running", progress_percent=70)

    dims = [_video_dimensions(p) for p in prepared_raw]
    if any(w * h == 0 for w, h in dims):
        raise UnsafeRenderCommandError("Failed to read dimensions from prepared clips.")
    max_w = max(w for w, _ in dims)
    max_h = max(h for _, h in dims)
    max_w += max_w % 2
    max_h += max_h % 2

    need_pad = any(w != max_w or h != max_h for w, h in dims)
    prepared_final: list[Path] = []
    if not need_pad:
        prepared_final = prepared_raw
    else:
        for i, p in enumerate(prepared_raw):
            w, h = dims[i]
            if w == max_w and h == max_h:
                prepared_final.append(p)
                continue
            outp = scratch_dir / f"clip_{i:04d}_pad.mp4"
            logger.info("Render %s: padding clip %d to %dx%d", job_id or "?", i, max_w, max_h)
            _run_cmd(_pad_to_canvas(p, outp, max_w, max_h))
            prepared_final.append(outp)

    logger.info("Render %s: concatenating %d clip(s) (orientation=%s)", job_id or "?", len(prepared_final), orientation)
    _concat_demuxer(prepared_final, output_file)


def create_render_job(
    tenant_id: str,
    event_id: str,
    plan: PlannerPlan,
    *,
    planner_prompt: str | None = None,
) -> RenderJob:
    selected_ids: list[str] = []
    selected_segment_ids: list[str] = []
    duration = 60
    order_strategy = "chronological"
    for action in plan.actions:
        if action.action == "select_segments":
            selected_ids = list(action.params.get("asset_ids", []))
            selected_segment_ids = list(action.params.get("segment_ids", []))
        if action.action == "set_duration":
            duration = int(action.params.get("seconds", 60))
        if action.action == "set_order":
            order_strategy = str(action.params.get("strategy", "chronological"))

    orientation = _orientation_from_plan(plan)

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

    tag_by_asset = _latest_vlm_tags_by_asset(event_id)
    for c in clip_inputs:
        aid = str(c.get("asset_id", ""))
        c["display_transpose"] = _transpose_from_tags(tag_by_asset.get(aid, []))

    input_files = [str(item.get("path", "")) for item in clip_inputs]
    output_dir = Path(settings.storage_root) / tenant_id / event_id / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / f"{next_id('preview')}.mp4")

    p_prompt = planner_prompt.strip() if planner_prompt and planner_prompt.strip() else None
    render_options: dict = {
        "order_strategy": order_strategy,
        "orientation": orientation,
    }
    if p_prompt:
        render_options["planner_prompt"] = p_prompt

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
        planner_prompt=p_prompt,
    )
    scratch_dir = Path(settings.scratch_root) / tenant_id / event_id / "renders" / job.id
    RenderRepository.create(
        job,
        input_files=input_files,
        duration_seconds=duration,
        scratch_dir=str(scratch_dir),
        subtitles_enabled=False,
        overlays_enabled=False,
        clip_inputs=clip_inputs,
        render_options=render_options,
    )
    audit_action(
        tenant_id,
        event_id,
        "render_job_created",
        {
            "render_job_id": job.id,
            "orientation": orientation,
            "clip_count": len(clip_inputs),
        },
    )
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
    raw_scratch = spec.get("scratch_dir")
    scratch_dir = Path(raw_scratch) if raw_scratch else Path(settings.scratch_root) / job_id
    render_options = spec.get("render_options") if isinstance(spec.get("render_options"), dict) else {}

    logger.info("Render job %s: starting (duration=%ss, clips=%d)", job_id, duration_seconds, len(clip_inputs or input_files))

    try:
        output = Path(job.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        RenderRepository.update_status(job_id, "running", progress_percent=25)
        ci = clip_inputs or [
            {"path": p, "asset_id": "", "start_s": 0.0, "end_s": float(duration_seconds), "score": 0.5} for p in input_files
        ]
        _render_from_inputs(
            clip_inputs=ci,
            output_file=str(output),
            duration_seconds=duration_seconds,
            scratch_dir=scratch_dir,
            job_id=job_id,
            render_options=render_options,
        )

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
