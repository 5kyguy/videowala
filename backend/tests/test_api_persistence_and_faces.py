from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.db import reset_database_for_tests
from app.main import app
from app.db import now_utc
from app.repositories import RenderRepository
from app.schemas import RenderJob


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


def test_face_reference_multipart_upload_list_and_image() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Face upload", "event_type": "party"}).json()
    event_id = event["id"]
    person = client.post(
        "/persons",
        json={"tenant_id": "tenant_a", "event_id": event_id, "display_name": "Jordan"},
    ).json()
    person_id = person["id"]
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    up = client.post(
        f"/events/{event_id}/persons/{person_id}/face-reference",
        params={"tenant_id": "tenant_a"},
        files={"file": ("ref.png", png_bytes, "image/png")},
    )
    assert up.status_code == 200
    ref_id = up.json()["reference"]["id"]
    listed = client.get(f"/events/{event_id}/person-references", params={"tenant_id": "tenant_a"})
    assert listed.status_code == 200
    refs = listed.json()["references"]
    assert len(refs) == 1
    assert refs[0]["display_name"] == "Jordan"
    assert refs[0]["id"] == ref_id
    img = client.get(f"/events/{event_id}/person-references/{ref_id}/image", params={"tenant_id": "tenant_a"})
    assert img.status_code == 200
    assert len(img.content) == len(png_bytes)


def test_event_summary_and_render_list_reject_cross_tenant_access() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Scoped", "event_type": "wedding"}).json()
    event_id = event["id"]

    summary_resp = client.get(f"/events/{event_id}/summary", params={"tenant_id": "tenant_b"})
    assert summary_resp.status_code == 403

    renders_resp = client.get(f"/events/{event_id}/renders", params={"tenant_id": "tenant_b"})
    assert renders_resp.status_code == 403


def test_delete_event_endpoint_removes_event() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Delete me", "event_type": "party"}).json()
    event_id = event["id"]

    deleted = client.delete(f"/events/{event_id}", params={"tenant_id": "tenant_a"})
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    listed = client.get("/events", params={"tenant_id": "tenant_a"})
    assert listed.status_code == 200
    ids = [item["id"] for item in listed.json()["events"]]
    assert event_id not in ids


def test_delete_render_endpoint_removes_render_job() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Render delete", "event_type": "party"}).json()
    event_id = event["id"]
    render_job = RenderJob(
        id="render_test_delete_1",
        tenant_id="tenant_a",
        event_id=event_id,
        plan_id="plan_test",
        status="queued",
        output_path=None,
        progress_percent=0,
        error_message=None,
        created_at=now_utc(),
    )
    RenderRepository.create(
        render_job,
        input_files=[],
        duration_seconds=30,
        scratch_dir="tmp/test-render-delete",
    )
    render_job_id = render_job.id

    deleted = client.delete(f"/renders/{render_job_id}", params={"tenant_id": "tenant_a"})
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    fetch = client.get(f"/renders/{render_job_id}", params={"tenant_id": "tenant_a"})
    assert fetch.status_code == 404
