from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.db import reset_database_for_tests
from app.main import app


def setup_function() -> None:
    reset_database_for_tests("storage/test_batch_ingest.db")
    settings.stage2_stub_models = True
    settings.db_path = "storage/test_batch_ingest.db"


def test_batch_ingest_folder_detects_media() -> None:
    client = TestClient(app)
    event = client.post(
        "/events",
        json={"tenant_id": "tenant_a", "title": "Batch", "event_type": "party"},
    ).json()
    media_root = Path(__file__).resolve().parents[1] / "test" / "media"
    if not media_root.is_dir():
        return
    resp = client.post(
        "/assets",
        json={
            "tenant_id": "tenant_a",
            "event_id": event["id"],
            "path": str(media_root),
            "recursive": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch"] is True
    assert data["count"] >= 1
    assert len(data["assets"]) == data["count"]
    for row in data["assets"]:
        assert "asset_id" in row
        assert row["media_type"] in ("image", "video")


def test_ingest_single_file_via_path() -> None:
    client = TestClient(app)
    event = client.post(
        "/events",
        json={"tenant_id": "tenant_a", "title": "One", "event_type": "party"},
    ).json()
    mp4 = Path(__file__).resolve().parents[1] / "test" / "media" / "dance.mp4"
    if not mp4.is_file():
        return
    resp = client.post(
        "/assets",
        json={
            "tenant_id": "tenant_a",
            "event_id": event["id"],
            "path": str(mp4),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch"] is True
    assert data["count"] == 1


def test_legacy_single_file_body_still_works() -> None:
    client = TestClient(app)
    event = client.post(
        "/events",
        json={"tenant_id": "tenant_a", "title": "Legacy", "event_type": "party"},
    ).json()
    resp = client.post(
        "/assets",
        json={
            "tenant_id": "tenant_a",
            "event_id": event["id"],
            "media_path": "test/media/dance.mp4",
            "media_type": "video",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("batch") is not True
    assert "asset_id" in data
