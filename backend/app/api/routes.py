from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..db import now_utc
from ..repositories import AssetRepository, EventRepository, PersonReferenceRepository, PersonRepository, RenderRepository, next_id
from ..schemas import (
    AssetIngestBody,
    AssetRegister,
    ContentRequestCreate,
    Event,
    EventCreate,
    FeedbackUpdate,
    Person,
    PersonCreate,
    PersonReference,
    PersonReferenceCreate,
)
from ..services.faces import face_service
from ..services.ingest import create_asset_record, discover_media_files, register_asset, resolve_ingest_path
from ..services.indexing import get_event_context, get_event_context_filtered
from ..services.search import semantic_search
from ..services.planner import build_plan
from ..services.privacy import assert_tenant_scope, audit_action, cleanup_tenant_scratch
from ..services.rendering import UnsafeRenderCommandError, create_render_job, execute_render_job
from ..workers.index_worker import run_index_job

logger = logging.getLogger(__name__)

router = APIRouter()


def _assert_path_within_root(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved == resolved:
        return
    if root_resolved not in resolved.parents:
        raise HTTPException(status_code=400, detail="Render output path is outside storage root.")


@router.post("/events", response_model=Event)
def create_event(payload: EventCreate) -> Event:
    event = Event(
        id=next_id("event"),
        tenant_id=payload.tenant_id,
        title=payload.title,
        event_type=payload.event_type,
        venue=payload.venue,
        date=payload.date,
        created_at=now_utc(),
    )
    EventRepository.create(event)
    audit_action(event.tenant_id, event.id, "event_created", {"title": event.title})
    return event


@router.get("/events")
def list_events(tenant_id: str) -> dict:
    events = EventRepository.list_for_tenant(tenant_id=tenant_id)
    return {"events": [event.model_dump() for event in events]}


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
        for abs_path, mtype in file_iter:
            asset = create_asset_record(payload.tenant_id, payload.event_id, str(abs_path), mtype)
            try:
                n = run_index_job(asset.id)
                assets_out.append(
                    {
                        "asset_id": asset.id,
                        "media_path": str(abs_path),
                        "media_type": mtype,
                        "insights_generated": n,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.exception("Indexing failed for %s", abs_path)
                assets_out.append(
                    {
                        "asset_id": asset.id,
                        "media_path": str(abs_path),
                        "media_type": mtype,
                        "insights_generated": 0,
                        "error": str(exc),
                    }
                )
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
    insight_count = run_index_job(asset.id)
    return {"asset_id": asset.id, "insights_generated": insight_count}


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
    plan = build_plan(payload, context)
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
    plan = build_plan(payload, context)
    try:
        job = create_render_job(tenant_id=payload.tenant_id, event_id=payload.event_id, plan=plan)
        executed = execute_render_job(job.id)
    except UnsafeRenderCommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if executed.status != "completed":
        raise HTTPException(status_code=500, detail="Rendering failed. Check server logs/audit.")
    return {"render_job": executed.model_dump(), "plan": plan.model_dump()}


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
        render_subtitles=payload.render_subtitles,
        render_overlays=payload.render_overlays,
    )
    context = get_event_context(payload.event_id)
    plan = build_plan(request, context)
    try:
        job = create_render_job(tenant_id=payload.tenant_id, event_id=payload.event_id, plan=plan)
        executed = execute_render_job(job.id)
    except UnsafeRenderCommandError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if executed.status != "completed":
        raise HTTPException(status_code=500, detail="Rendering failed. Check server logs/audit.")
    return {
        "status": "regenerated",
        "feedback": payload.model_dump(),
        "plan": plan.model_dump(),
        "render_job": executed.model_dump(),
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
    for asset in assets:
        run_index_job(asset.id)
    audit_action(tenant_id, event_id, "face_reindex_triggered", {"asset_count": len(assets)})
    return {"status": "reindexed", "asset_count": len(assets)}


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
