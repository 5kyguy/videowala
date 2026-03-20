from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.db import reset_database_for_tests
from app.main import app


def setup_function() -> None:
    reset_database_for_tests("storage/test_api.db")
    settings.stage2_stub_models = True


def test_events_are_persisted_across_clients() -> None:
    client_a = TestClient(app)
    payload = {"tenant_id": "tenant_a", "title": "Persisted Event", "event_type": "wedding"}
    created = client_a.post("/events", json=payload)
    assert created.status_code == 200
    event_id = created.json()["id"]

    client_b = TestClient(app)
    listed = client_b.get("/events", params={"tenant_id": "tenant_a"})
    assert listed.status_code == 200
    ids = [item["id"] for item in listed.json()["events"]]
    assert event_id in ids


def test_face_enrollment_and_match_routes_work() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Event", "event_type": "party"}).json()
    event_id = event["id"]

    person_resp = client.post(
        "/persons",
        json={"tenant_id": "tenant_a", "event_id": event_id, "display_name": "Alice"},
    )
    assert person_resp.status_code == 200
    person_id = person_resp.json()["id"]

    ref_resp = client.post(
        f"/persons/{person_id}/references",
        json={"tenant_id": "tenant_a", "event_id": event_id, "image_path": "test/media/dance.mp4"},
    )
    assert ref_resp.status_code == 200

    ingest_resp = client.post(
        "/assets",
        json={"tenant_id": "tenant_a", "event_id": event_id, "media_path": "test/media/dance.mp4", "media_type": "video"},
    )
    assert ingest_resp.status_code == 200

    reindex_resp = client.post(f"/events/{event_id}/faces/reindex", params={"tenant_id": "tenant_a"})
    assert reindex_resp.status_code == 200

    matches_resp = client.get(
        f"/events/{event_id}/faces/matches",
        params={"tenant_id": "tenant_a", "person_id": person_id},
    )
    assert matches_resp.status_code == 200
    assert "matches" in matches_resp.json()


def test_face_endpoints_reject_cross_tenant_access() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Event", "event_type": "party"}).json()
    event_id = event["id"]
    person = client.post(
        "/persons",
        json={"tenant_id": "tenant_a", "event_id": event_id, "display_name": "Bob"},
    ).json()

    bad_ref = client.post(
        f"/persons/{person['id']}/references",
        json={"tenant_id": "tenant_b", "event_id": event_id, "image_path": "test/media/dance.mp4"},
    )
    assert bad_ref.status_code == 403


def test_event_summary_reports_dashboard_stats() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Stats Event", "event_type": "party"}).json()
    event_id = event["id"]

    ingest_resp = client.post(
        "/assets",
        json={"tenant_id": "tenant_a", "event_id": event_id, "media_path": "test/media/dance.mp4", "media_type": "video"},
    )
    assert ingest_resp.status_code == 200

    person = client.post(
        "/persons",
        json={"tenant_id": "tenant_a", "event_id": event_id, "display_name": "Alice"},
    ).json()
    ref_resp = client.post(
        f"/persons/{person['id']}/references",
        json={"tenant_id": "tenant_a", "event_id": event_id, "image_path": "test/media/dance.mp4"},
    )
    assert ref_resp.status_code == 200

    summary_resp = client.get(f"/events/{event_id}/summary", params={"tenant_id": "tenant_a"})
    assert summary_resp.status_code == 200
    body = summary_resp.json()
    assert body["event"]["id"] == event_id
    assert body["stats"]["assets_total"] >= 1
    assert body["stats"]["videos_total"] >= 1
    assert body["stats"]["has_media"] is True
    assert body["stats"]["persons_total"] == 1
    assert body["stats"]["faces_saved"] is True
    assert body["stats"]["renders_total"] == 0
    assert body["stats"]["index_jobs_total"] >= 1
    assert body["stats"]["index_jobs_queued"] + body["stats"]["index_jobs_running"] + body["stats"]["index_jobs_completed"] + body["stats"]["index_jobs_failed"] == body["stats"]["index_jobs_total"]


def test_event_summary_and_render_list_reject_cross_tenant_access() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Scoped", "event_type": "wedding"}).json()
    event_id = event["id"]

    summary_resp = client.get(f"/events/{event_id}/summary", params={"tenant_id": "tenant_b"})
    assert summary_resp.status_code == 403

    renders_resp = client.get(f"/events/{event_id}/renders", params={"tenant_id": "tenant_b"})
    assert renders_resp.status_code == 403
