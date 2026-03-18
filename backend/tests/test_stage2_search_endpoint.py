from __future__ import annotations

from fastapi.testclient import TestClient

from app.db import reset_database_for_tests
from app.main import app


def setup_function() -> None:
    reset_database_for_tests("storage/test_stage2_search.db")


def test_search_endpoint_returns_empty_without_pgvector_data() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "Event", "event_type": "party"}).json()
    resp = client.get(
        f"/events/{event['id']}/search",
        params={"tenant_id": "tenant_a", "q": "dancing", "limit": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["hits"] == []

