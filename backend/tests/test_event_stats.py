from __future__ import annotations

from pathlib import Path

from app.db import now_utc
from app.schemas import Asset
from app.services.event_stats import media_footprint, top_extensions_by_count


def test_media_footprint_sums_absolute_paths(tmp_path: Path) -> None:
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x" * 500)
    vid = tmp_path / "y.mp4"
    vid.write_bytes(b"y" * 1500)
    now = now_utc()
    assets = [
        Asset(
            id="a1",
            tenant_id="t",
            event_id="e",
            media_path=str(img),
            media_type="image",
            created_at=now,
        ),
        Asset(
            id="a2",
            tenant_id="t",
            event_id="e",
            media_path=str(vid),
            media_type="video",
            created_at=now,
        ),
    ]
    mf = media_footprint(assets)
    assert mf["files_found"] == 2
    assert mf["files_missing"] == 0
    assert mf["total_bytes"] == 2000
    assert mf["bytes_images"] == 500
    assert mf["bytes_videos"] == 1500
    assert mf["extension_counts"][".jpg"] == 1
    assert mf["extension_counts"][".mp4"] == 1


def test_top_extensions_orders_by_count() -> None:
    ext = {".jpg": 3, ".mp4": 10, ".png": 10}
    top = top_extensions_by_count(ext, limit=2)
    assert top == [(".mp4", 10), (".png", 10)]


def test_sum_index_job_duration_seconds() -> None:
    from app.repositories import AssetRepository, EventRepository, IndexJobRepository, next_id
    from app.schemas import Asset, Event, IndexJob

    from app.db import reset_database_for_tests

    reset_database_for_tests("storage/test_evstat_idx.db")
    event = Event(
        id=next_id("event"),
        tenant_id="tenant_a",
        title="E",
        event_type="test",
        predefined_tags=[],
        ocr_languages=["en"],
        created_at=now_utc(),
    )
    EventRepository.create(event)
    asset = Asset(
        id=next_id("asset"),
        tenant_id="tenant_a",
        event_id=event.id,
        media_path="test/media/dance.mp4",
        media_type="video",
        created_at=now_utc(),
    )
    AssetRepository.create(asset)
    job = IndexJob(
        id=next_id("idxjob"),
        tenant_id="tenant_a",
        event_id=event.id,
        asset_id=asset.id,
        status="queued",
        progress_percent=0,
        insights_generated=0,
        error_message=None,
        created_at=now_utc(),
        semantic_prompt=None,
    )
    IndexJobRepository.create(job)
    IndexJobRepository.mark_running(job.id)
    IndexJobRepository.mark_completed(job.id, 2)
    total, n = IndexJobRepository.sum_index_job_duration_seconds(event.id)
    assert n == 1
    assert total >= 0.0
