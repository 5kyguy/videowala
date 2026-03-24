from __future__ import annotations

import logging
import mimetypes
import sys
import zipfile
from io import BytesIO
from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from ..config import PROJECT_ROOT, settings
from ..db import now_utc
from ..repositories import (
    AssetRepository,
    EventRepository,
    IndexJobRepository,
    InsightRepository,
    PersonReferenceRepository,
    PersonRepository,
    RenderRepository,
    next_id,
)
from ..schemas import (
    AssetIngestBody,
    AssetRegister,
    ContentRequestCreate,
    Event,
    EventCreate,
    EventSummary,
    EventSummaryStats,
    EventUpdate,
    MediaExtensionCount,
    FeedbackUpdate,
    Person,
    PersonCreate,
    PhotoCurationListResponse,
    PhotoCurationRequest,
    PersonReference,
    PersonReferenceCreate,
    RenderJobList,
)
from ..services.event_stats import media_footprint, renders_output_bytes, top_extensions_by_count
from ..services.faces import face_service
from ..services.ingest import create_asset_record, discover_media_files, purge_event_proxies, register_asset, resolve_ingest_path
from ..services.indexing import get_event_context, get_event_context_filtered, reindex_face_insights_for_asset
from ..services.search import semantic_search
from ..services.planner import PlannerValidationError, build_plan
from ..services.privacy import assert_tenant_scope, audit_action, cleanup_tenant_scratch
from ..services.photo_curation import kept_photo_items_for_export, list_photo_curation_items, score_photo_segments
from ..services.rendering import UnsafeRenderCommandError, create_render_job, validate_safe_path
from ..workers.index_worker import submit_index_job, submit_render_job, submit_staged_index_job

logger = logging.getLogger(__name__)

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ALLOWED_EVENT_ASSET_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def _resolve_repo_media_path(raw: str) -> Path:
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()
    alt = _REPO_ROOT / raw
    if alt.is_file():
        return alt.resolve()
    return direct


def _allowed_media_file_roots() -> tuple[Path, Path]:
    """Ingest may register repo-relative paths or files under tenant storage; block arbitrary filesystem reads."""
    return (Path(settings.storage_root).resolve(), PROJECT_ROOT.resolve())


def _media_path_is_allowed(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in _allowed_media_file_roots())


def _assert_media_path_allowed(path: Path) -> None:
    if not _media_path_is_allowed(path):
        raise HTTPException(
            status_code=400,
            detail="Media file path is outside allowed directories (project root or configured storage).",
        )


def _image_bytes_scaled_max_edge(path: Path, max_edge: int) -> tuple[bytes, str]:
    """Resize so longest edge is at most `max_edge` (aspect preserved). JPEG for opaque, PNG if alpha."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buf = BytesIO()
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
            has_alpha = True
        if has_alpha:
            im.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), "image/png"
        rgb = im.convert("RGB")
        rgb.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"


def _assert_path_within_root(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved == resolved:
        return
    if root_resolved not in resolved.parents:
        raise HTTPException(status_code=400, detail="Render output path is outside storage root.")


_ALLOWED_FACE_REF_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_MAX_FACE_UPLOAD_BYTES = 15 * 1024 * 1024


def _face_refs_dir(tenant_id: str, event_id: str) -> Path:
    base = Path(settings.storage_root) / tenant_id / event_id / "face_refs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_face_filename(original: str) -> str:
    p = Path(original)
    ext = p.suffix.lower()
    if ext not in _ALLOWED_FACE_REF_SUFFIXES:
        ext = ".jpg"
    stem = "".join(c for c in p.stem if c.isalnum() or c in "._-")[:64]
    return f"{stem or 'photo'}{ext}"


@router.post("/events", response_model=Event)
def create_event(payload: EventCreate) -> Event:
    event = Event(
        id=next_id("event"),
        tenant_id=payload.tenant_id,
        title=payload.title,
        event_type=payload.event_type,
        venue=payload.venue,
        date=payload.date,
        predefined_tags=list(payload.predefined_tags),
        ocr_languages=list(payload.ocr_languages) if payload.ocr_languages else ["en"],
        created_at=now_utc(),
    )
    EventRepository.create(event)
    audit_action(event.tenant_id, event.id, "event_created", {"title": event.title})
    return event


@router.patch("/events/{event_id}", response_model=Event)
def patch_event(event_id: str, payload: EventUpdate, tenant_id: str) -> Event:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    patch = payload.model_dump(exclude_unset=True)
    if not patch:
        return event
    updated = EventRepository.update(event_id, patch)
    if updated is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    audit_action(tenant_id, event_id, "event_updated", patch)
    return updated


@router.get("/events")
def list_events(tenant_id: str) -> dict:
    events = EventRepository.list_for_tenant(tenant_id=tenant_id)
    return {"events": [event.model_dump() for event in events]}


@router.delete("/events/{event_id}")
def delete_event(event_id: str, tenant_id: str) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    storage_dir = Path(settings.storage_root) / tenant_id / event_id
    scratch_dir = Path(settings.scratch_root) / tenant_id / event_id

    deleted = EventRepository.delete(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Best-effort cleanup for event-scoped files; DB delete already succeeded.
    if storage_dir.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)

    audit_action(tenant_id, event_id, "event_deleted", {})
    return {"status": "deleted", "event_id": event_id}


@router.post("/assets")
def ingest_asset(payload: AssetIngestBody) -> dict:
    event = EventRepository.get(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if payload.path:
        try:
            root = resolve_ingest_path(payload.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        pairs = discover_media_files(root, recursive=payload.recursive)
        if not pairs:
            raise HTTPException(
                status_code=400,
                detail="No supported image/video files found. "
                "Supported: common image extensions (jpg, png, …) and video (mp4, mov, mkv, …).",
            )
        assets_out: list[dict] = []
        file_iter = pairs
        if settings.indexing_show_progress:
            try:
                from tqdm import tqdm

                file_iter = tqdm(
                    pairs,
                    desc="Batch ingest",
                    unit="file",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                pass
        failed = 0
        staged_ids: list[str] = []
        for abs_path, mtype in file_iter:
            try:
                asset = create_asset_record(payload.tenant_id, payload.event_id, str(abs_path), mtype)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.exception("Asset create failed for %s", abs_path)
                assets_out.append(
                    {
                        "media_path": str(abs_path),
                        "media_type": mtype,
                        "insights_generated": 0,
                        "error": str(exc),
                    }
                )
                continue
            staged_ids.append(asset.id)
            assets_out.append(
                {
                    "asset_id": asset.id,
                    "media_path": str(abs_path),
                    "media_type": mtype,
                    "insights_generated": 0,
                    "index_job_id": None,
                    "error": None,
                }
            )
        batch_job = None
        if staged_ids:
            try:
                batch_job = submit_staged_index_job(
                    payload.tenant_id,
                    payload.event_id,
                    staged_ids,
                    semantic_prompt=payload.semantic_prompt,
                )
            except Exception as exc:  # noqa: BLE001
                failed += len(staged_ids)
                logger.exception("Staged index job failed for event %s", payload.event_id)
                for row in assets_out:
                    if row.get("error") is None:
                        row["error"] = str(exc)
            else:
                jid = batch_job.id
                for row in assets_out:
                    if row.get("error") is None:
                        row["index_job_id"] = jid
        audit_action(
            payload.tenant_id,
            payload.event_id,
            "batch_ingest_completed",
            {"count": len(assets_out), "failed": failed, "root": payload.path},
        )
        return {"batch": True, "count": len(assets_out), "failed": failed, "assets": assets_out}

    assert payload.media_path is not None and payload.media_type is not None
    asset = register_asset(
        AssetRegister(
            tenant_id=payload.tenant_id,
            event_id=payload.event_id,
            media_path=payload.media_path,
            media_type=payload.media_type,
        )
    )
    job = submit_index_job(asset.id, semantic_prompt=payload.semantic_prompt)
    return {"asset_id": asset.id, "insights_generated": 0, "index_job_id": job.id, "index_status": job.status}


@router.get("/index-jobs/{index_job_id}")
def get_index_job(index_job_id: str, tenant_id: str) -> dict:
    job = IndexJobRepository.get(index_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Index job not found.")
    try:
        assert_tenant_scope(tenant_id, job.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"index_job": job.model_dump()}


@router.get("/events/{event_id}/context")
def event_context(event_id: str, tenant_id: str, insight_type: str | None = None, person_id: str | None = None) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    context = get_event_context_filtered(event_id, insight_type=insight_type, person_id=person_id)
    return {"event_id": event_id, "context": context}


@router.get("/events/{event_id}/summary", response_model=EventSummary)
def get_event_summary(event_id: str, tenant_id: str) -> EventSummary:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    assets = AssetRepository.list_for_event(event_id)
    persons = PersonRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    references = PersonReferenceRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    face_match_insights = InsightRepository.list_for_event(event_id=event_id, insight_type="face_matches")
    renders = RenderRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)

    images_total = sum(1 for asset in assets if asset.media_type == "image")
    videos_total = sum(1 for asset in assets if asset.media_type == "video")
    renders_by_status = {
        "queued": sum(1 for job in renders if job.status == "queued"),
        "running": sum(1 for job in renders if job.status == "running"),
        "completed": sum(1 for job in renders if job.status == "completed"),
        "failed": sum(1 for job in renders if job.status == "failed"),
    }

    index_by_status = IndexJobRepository.count_by_status_for_event(event_id)
    index_jobs_total = sum(index_by_status.values())

    active_index = IndexJobRepository.get_active_index_job_for_event(event_id)
    index_current_stage: str | None = None
    index_current_progress_percent: int | None = None
    if active_index is not None:
        index_current_progress_percent = active_index.progress_percent
        index_current_stage = active_index.index_stage
        if active_index.status == "queued" and not index_current_stage:
            index_current_stage = "Queued"

    mf = media_footprint(assets)
    idx_sec, idx_n = IndexJobRepository.sum_index_job_duration_seconds(event_id)
    ext_top = [
        MediaExtensionCount(extension=ext, count=cnt)
        for ext, cnt in top_extensions_by_count(mf["extension_counts"], limit=12)
    ]

    return EventSummary(
        event=event,
        stats=EventSummaryStats(
            assets_total=len(assets),
            images_total=images_total,
            videos_total=videos_total,
            has_media=bool(assets),
            persons_total=len(persons),
            face_references_total=len(references),
            faces_saved=bool(references),
            face_match_insights_total=len(face_match_insights),
            has_face_matches=bool(face_match_insights),
            renders_total=len(renders),
            renders_queued=renders_by_status["queued"],
            renders_running=renders_by_status["running"],
            renders_completed=renders_by_status["completed"],
            renders_failed=renders_by_status["failed"],
            index_jobs_total=index_jobs_total,
            index_jobs_queued=index_by_status["queued"],
            index_jobs_running=index_by_status["running"],
            index_jobs_completed=index_by_status["completed"],
            index_jobs_failed=index_by_status["failed"],
            media_storage_bytes=int(mf["total_bytes"]),
            media_storage_files_found=int(mf["files_found"]),
            media_storage_files_missing=int(mf["files_missing"]),
            media_bytes_images=int(mf["bytes_images"]),
            media_bytes_videos=int(mf["bytes_videos"]),
            renders_storage_bytes=renders_output_bytes(renders),
            index_duration_seconds_total=float(idx_sec),
            index_duration_job_count=int(idx_n),
            media_extension_top=ext_top,
            index_current_stage=index_current_stage,
            index_current_progress_percent=index_current_progress_percent,
        ),
    )


@router.get("/events/{event_id}/search")
def search_event(event_id: str, tenant_id: str, q: str, limit: int = 20) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    hits = semantic_search(tenant_id=tenant_id, event_id=event_id, query=q, limit=limit)
    audit_action(tenant_id, event_id, "semantic_search", {"q": q, "limit": limit})
    return {"event_id": event_id, "hits": hits}


@router.post("/requests/plan")
def create_plan(payload: ContentRequestCreate) -> dict:
    event = EventRepository.get(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    context = get_event_context(payload.event_id)
    try:
        plan = build_plan(payload, context)
    except PlannerValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"plan": plan.model_dump()}


@router.post("/requests/render")
def create_render(payload: ContentRequestCreate) -> dict:
    event = EventRepository.get(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    context = get_event_context(payload.event_id)
    try:
        plan = build_plan(payload, context)
    except PlannerValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        job = create_render_job(
            tenant_id=payload.tenant_id,
            event_id=payload.event_id,
            plan=plan,
            planner_prompt=payload.prompt,
        )
        submit_render_job(job.id)
    except UnsafeRenderCommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"render_job": job.model_dump(), "plan": plan.model_dump(), "queued": True}


@router.get("/renders/{render_job_id}/video")
def get_render_video(render_job_id: str, tenant_id: str) -> FileResponse:
    job = RenderRepository.get(render_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found.")
    try:
        assert_tenant_scope(tenant_id, job.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=409, detail=f"Render job is not completed (status={job.status}).")

    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Render output file missing on disk.")
    _assert_path_within_root(path, Path(settings.storage_root))

    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/renders/{render_job_id}")
def get_render_job(render_job_id: str, tenant_id: str) -> dict:
    job = RenderRepository.get(render_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found.")
    try:
        assert_tenant_scope(tenant_id, job.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"render_job": job.model_dump()}


@router.delete("/renders/{render_job_id}")
def delete_render_job(render_job_id: str, tenant_id: str) -> dict:
    job = RenderRepository.get(render_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found.")
    try:
        assert_tenant_scope(tenant_id, job.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    output_path = Path(job.output_path) if job.output_path else None
    spec = RenderRepository.get_spec(render_job_id)
    scratch_dir = Path(spec["scratch_dir"]) if spec and spec.get("scratch_dir") else None

    deleted = RenderRepository.delete(render_job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Render job not found.")

    # Best-effort cleanup for generated render files.
    if output_path and output_path.exists():
        output_path.unlink(missing_ok=True)
    if scratch_dir and scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)

    audit_action(tenant_id, job.event_id, "render_deleted", {"render_job_id": render_job_id})
    return {"status": "deleted", "render_job_id": render_job_id}


@router.get("/events/{event_id}/renders", response_model=RenderJobList)
def list_event_renders(event_id: str, tenant_id: str) -> RenderJobList:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    renders = RenderRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    return RenderJobList(event_id=event_id, renders=renders)


@router.get("/events/{event_id}/photos/curation", response_model=PhotoCurationListResponse)
def get_photo_curation(event_id: str, tenant_id: str) -> PhotoCurationListResponse:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    items = list_photo_curation_items(event_id)
    audit_action(tenant_id, event_id, "photo_curation_viewed", {"count": len(items)})
    return PhotoCurationListResponse(event_id=event_id, items=items)


@router.post("/events/{event_id}/photos/curation", response_model=PhotoCurationListResponse)
def run_photo_curation(event_id: str, payload: PhotoCurationRequest) -> PhotoCurationListResponse:
    if payload.event_id != event_id:
        raise HTTPException(status_code=400, detail="event_id in path must match body.")
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    items = score_photo_segments(payload)
    audit_action(
        payload.tenant_id,
        event_id,
        "photo_curation_run",
        {"prompt": payload.prompt, "cull_percent": payload.cull_percent, "count": len(items)},
    )
    return PhotoCurationListResponse(event_id=event_id, items=items)


@router.get("/events/{event_id}/assets/{asset_id}/media", response_model=None)
def get_event_asset_media(
    event_id: str,
    asset_id: str,
    tenant_id: str,
    max_edge: int | None = Query(
        None,
        ge=64,
        le=4096,
        description="If set, return a scaled preview (longest edge ≤ this value). Omit for full-resolution file.",
    ),
) -> FileResponse | Response:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    asset = AssetRepository.get(asset_id)
    if asset is None or asset.event_id != event_id:
        raise HTTPException(status_code=404, detail="Asset not found.")
    if asset.media_type != "image":
        raise HTTPException(status_code=400, detail="Only image assets can be served from this endpoint.")
    try:
        validate_safe_path(asset.media_path)
    except UnsafeRenderCommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = _resolve_repo_media_path(asset.media_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file missing on disk.")
    _assert_media_path_allowed(path)
    ext = path.suffix.lower()
    if ext not in _ALLOWED_EVENT_ASSET_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image extension for browser delivery.")
    if max_edge is not None:
        try:
            body, media_type = _image_bytes_scaled_max_edge(path, max_edge)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Could not build image preview: {exc}") from exc
        return Response(content=body, media_type=media_type)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/events/{event_id}/photos/export-kept")
def export_kept_photos_zip(event_id: str, tenant_id: str) -> FileResponse:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    kept = kept_photo_items_for_export(event_id)
    if not kept:
        raise HTTPException(
            status_code=400,
            detail="No kept photos to export (need indexed images with keep=true and not duplicate).",
        )
    export_dir = Path(settings.scratch_root) / tenant_id / event_id / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"kept_photos_{next_id('zip')}.zip"
    added = 0
    used_names: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for it in kept:
            asset = AssetRepository.get(it.asset_id)
            if asset is None or asset.media_type != "image":
                continue
            try:
                validate_safe_path(asset.media_path)
            except UnsafeRenderCommandError:
                continue
            src = _resolve_repo_media_path(asset.media_path)
            if not src.is_file():
                continue
            if not _media_path_is_allowed(src):
                continue
            base = f"{it.asset_id}{src.suffix.lower() or '.jpg'}"
            arcname = base
            n = 0
            while arcname in used_names:
                n += 1
                arcname = f"{it.asset_id}_{n}{src.suffix.lower() or '.jpg'}"
            used_names.add(arcname)
            zf.write(src, arcname=f"kept/{arcname}")
            added += 1
    if added == 0:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not add any photo files to the archive.")
    audit_action(tenant_id, event_id, "photo_export_kept", {"count": added})
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="kept_photos.zip",
    )


@router.post("/requests/feedback/regenerate")
def regenerate_with_feedback(payload: FeedbackUpdate) -> dict:
    event = EventRepository.get(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    audit_action(
        payload.tenant_id,
        payload.event_id,
        "feedback_updated",
        {"include_asset_ids": payload.include_asset_ids, "exclude_asset_ids": payload.exclude_asset_ids},
    )
    request = ContentRequestCreate(
        tenant_id=payload.tenant_id,
        event_id=payload.event_id,
        output_type=payload.output_type,
        prompt=payload.prompt,
        target_duration_seconds=payload.target_duration_seconds,
        include_faces=[],
        include_asset_ids=payload.include_asset_ids,
        excluded_asset_ids=payload.exclude_asset_ids,
        include_media_types=payload.include_media_types,
        video_orientation=payload.video_orientation,
    )
    context = get_event_context(payload.event_id)
    try:
        plan = build_plan(request, context)
    except PlannerValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        job = create_render_job(
            tenant_id=payload.tenant_id,
            event_id=payload.event_id,
            plan=plan,
            planner_prompt=payload.prompt,
        )
        submit_render_job(job.id)
    except UnsafeRenderCommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "queued",
        "feedback": payload.model_dump(),
        "plan": plan.model_dump(),
        "render_job": job.model_dump(),
    }


@router.post("/events/{event_id}/cleanup")
def cleanup_event_scratch(event_id: str, tenant_id: str) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    cleanup_tenant_scratch(tenant_id=tenant_id, event_id=event_id)
    audit_action(tenant_id, event_id, "scratch_cleanup", {})
    return {"status": "cleaned"}


@router.post("/events/{event_id}/proxies/purge")
def purge_proxies(event_id: str, tenant_id: str) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    result = purge_event_proxies(tenant_id=tenant_id, event_id=event_id)
    return {"status": "purged", **result}


@router.post("/persons", response_model=Person)
def create_person(payload: PersonCreate) -> Person:
    event = EventRepository.get(payload.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(payload.tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    person = Person(
        id=next_id("person"),
        tenant_id=payload.tenant_id,
        event_id=payload.event_id,
        display_name=payload.display_name,
        created_at=now_utc(),
    )
    PersonRepository.create(person)
    audit_action(payload.tenant_id, payload.event_id, "person_created", {"person_id": person.id})
    return person


@router.get("/persons")
def list_persons(tenant_id: str, event_id: str) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    persons = PersonRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    return {"persons": [person.model_dump() for person in persons]}


@router.post("/persons/{person_id}/references", response_model=PersonReference)
def add_person_reference(person_id: str, payload: PersonReferenceCreate) -> PersonReference:
    person = PersonRepository.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found.")
    try:
        assert_tenant_scope(payload.tenant_id, person.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if payload.event_id != person.event_id:
        raise HTTPException(status_code=400, detail="event_id must match person event.")
    embedding = face_service.embed_reference(payload.image_path)
    reference = PersonReference(
        id=next_id("pref"),
        person_id=person_id,
        tenant_id=payload.tenant_id,
        event_id=payload.event_id,
        image_path=payload.image_path,
        embedding=embedding,
        created_at=now_utc(),
    )
    PersonReferenceRepository.create(reference)
    audit_action(payload.tenant_id, payload.event_id, "person_reference_added", {"person_id": person_id, "reference_id": reference.id})
    return reference


@router.post("/events/{event_id}/persons/{person_id}/face-reference")
async def upload_face_reference(
    event_id: str,
    person_id: str,
    tenant_id: str,
    file: UploadFile = File(...),
) -> dict:
    """Store a reference face image under event-scoped storage and register its embedding."""
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    person = PersonRepository.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found.")
    if person.event_id != event_id:
        raise HTTPException(status_code=400, detail="Person does not belong to this event.")

    raw = await file.read()
    if len(raw) > _MAX_FACE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15MB).")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    filename = _safe_face_filename(file.filename or "photo.jpg")
    dest = _face_refs_dir(tenant_id, event_id) / f"{next_id('fimg')}_{filename}"
    dest.write_bytes(raw)
    image_path = str(dest.resolve())

    embedding = face_service.embed_reference(image_path)
    reference = PersonReference(
        id=next_id("pref"),
        person_id=person_id,
        tenant_id=tenant_id,
        event_id=event_id,
        image_path=image_path,
        embedding=embedding,
        created_at=now_utc(),
    )
    PersonReferenceRepository.create(reference)
    audit_action(tenant_id, event_id, "person_reference_added", {"person_id": person_id, "reference_id": reference.id})
    return {
        "reference": {
            "id": reference.id,
            "person_id": reference.person_id,
            "tenant_id": reference.tenant_id,
            "event_id": reference.event_id,
            "image_path": reference.image_path,
            "created_at": reference.created_at.isoformat(),
        }
    }


@router.get("/events/{event_id}/person-references")
def list_event_person_references(event_id: str, tenant_id: str) -> dict:
    """Face reference metadata for the event (no embedding vectors)."""
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    persons = PersonRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    name_by_id = {p.id: p.display_name for p in persons}
    refs = PersonReferenceRepository.list_for_event(tenant_id=tenant_id, event_id=event_id)
    items = [
        {
            "id": r.id,
            "person_id": r.person_id,
            "event_id": r.event_id,
            "display_name": name_by_id.get(r.person_id, "unknown"),
            "image_path": r.image_path,
            "created_at": r.created_at.isoformat(),
        }
        for r in refs
    ]
    return {"event_id": event_id, "references": items}


@router.get("/events/{event_id}/person-references/{reference_id}/image")
def get_face_reference_image(event_id: str, reference_id: str, tenant_id: str) -> FileResponse:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    ref = PersonReferenceRepository.get(reference_id)
    if ref is None or ref.event_id != event_id:
        raise HTTPException(status_code=404, detail="Reference not found.")
    path = Path(ref.image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing on disk.")
    _assert_path_within_root(path, Path(settings.storage_root))
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/events/{event_id}/faces/reindex")
def reindex_event_faces(event_id: str, tenant_id: str) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    assets = AssetRepository.list_for_event(event_id)
    for i, asset in enumerate(assets, start=1):
        logger.info("Face reindex: processing asset %s (%d/%d)", asset.id, i, len(assets))
        reindex_face_insights_for_asset(asset.id)
    audit_action(tenant_id, event_id, "face_reindex_triggered", {"asset_count": len(assets)})
    return {"status": "reindexed", "asset_count": len(assets), "mode": "face_insights_only"}


@router.get("/events/{event_id}/faces/matches")
def list_face_matches(event_id: str, tenant_id: str, person_id: str | None = None) -> dict:
    event = EventRepository.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    try:
        assert_tenant_scope(tenant_id, event.tenant_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    context = get_event_context_filtered(event_id=event_id, insight_type="face_matches", person_id=person_id)
    audit_action(tenant_id, event_id, "face_matches_viewed", {"person_id": person_id})
    return {"event_id": event_id, "matches": context.get("face_matches", [])}
