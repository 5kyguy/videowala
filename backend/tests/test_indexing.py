from __future__ import annotations

from pathlib import Path

from app.db import now_utc, reset_database_for_tests
from app.config import settings
from app.repositories import EventRepository, next_id
from app.schemas import AssetRegister, Event, EventCreate
from app.services.ingest import register_asset
from app.services.indexing import get_event_context, index_asset, index_image_asset, index_video_asset
from app.workers import index_worker


def setup_function() -> None:
    reset_database_for_tests("storage/test_indexing.db")
    settings.stage2_stub_models = True


def _create_event() -> Event:
    payload = EventCreate(tenant_id="tenant_a", title="Wedding A", event_type="wedding")
    event = Event(
        id=next_id("event"),
        tenant_id=payload.tenant_id,
        title=payload.title,
        event_type=payload.event_type,
        predefined_tags=[],
        ocr_languages=["en"],
        created_at=now_utc(),
    )
    EventRepository.create(event)
    return event


def test_indexing_generates_vlm_and_face_context() -> None:
    Path(settings.storage_root).mkdir(parents=True, exist_ok=True)
    event = _create_event()
    asset = register_asset(
        AssetRegister(
            tenant_id=event.tenant_id,
            event_id=event.id,
            media_path="test/media/dance.mp4",
            media_type="video",
        )
    )
    insights = index_asset(asset.id)
    kinds = {item.insight_type.value for item in insights}
    assert "vlm_caption" in kinds
    assert "vlm_tags" in kinds
    assert "face_matches" in kinds

    context = get_event_context(event.id)
    assert "vlm_caption" in context
    assert "face_matches" in context


def test_index_video_asset_explicit_pipeline() -> None:
    Path(settings.storage_root).mkdir(parents=True, exist_ok=True)
    settings.stage2_stub_models = True
    event = _create_event()
    asset = register_asset(
        AssetRegister(
            tenant_id=event.tenant_id,
            event_id=event.id,
            media_path="test/media/dance.mp4",
            media_type="video",
        )
    )
    insights = index_video_asset(asset.id)
    assert any(i.insight_type.value == "asr_transcript" for i in insights)


def test_index_image_asset_pipeline_and_semantic_prompt() -> None:
    Path(settings.storage_root).mkdir(parents=True, exist_ok=True)
    settings.stage2_stub_models = True
    event = _create_event()
    asset = register_asset(
        AssetRegister(
            tenant_id=event.tenant_id,
            event_id=event.id,
            media_path="test/media/dance.mp4",
            media_type="image",
        )
    )
    insights = index_image_asset(asset.id, semantic_prompt="outdoor celebration theme")
    asr_ins = next(i for i in insights if i.insight_type.value == "asr_transcript")
    assert asr_ins.payload.get("segments") == []
    kinds = {i.insight_type.value for i in insights}
    assert "vlm_caption" in kinds


def test_index_worker_uses_serial_default() -> None:
    assert index_worker._EXECUTOR._max_workers == settings.index_workers  # noqa: SLF001
    assert settings.index_workers >= 1
