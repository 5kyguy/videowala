from __future__ import annotations

import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT
from app.db import now_utc, reset_database_for_tests
from app.main import app
from app.repositories import AssetRepository, EventRepository, SegmentRepository, next_id
from app.schemas import Asset, AssetSegment, Event
from app.services.photo_curation import kept_photo_items_for_export, list_photo_curation_items

# Minimal valid PNG (1×1) for on-disk media tests.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def setup_function() -> None:
    reset_database_for_tests("storage/test_photo_curation.db")


def test_list_photo_curation_items_only_images() -> None:
    EventRepository.create(
        Event(
            id="ev1",
            tenant_id="t1",
            title="E",
            event_type="test",
            predefined_tags=[],
            ocr_languages=["en"],
            created_at=now_utc(),
        )
    )
    AssetRepository.create(
        Asset(
            id="img1",
            tenant_id="t1",
            event_id="ev1",
            media_path="m/a.jpg",
            media_type="image",
            created_at=now_utc(),
        )
    )
    AssetRepository.create(
        Asset(
            id="vid1",
            tenant_id="t1",
            event_id="ev1",
            media_path="m/b.mp4",
            media_type="video",
            created_at=now_utc(),
        )
    )
    seg_img = AssetSegment(
        id=next_id("seg"),
        tenant_id="t1",
        event_id="ev1",
        asset_id="img1",
        start_s=0.0,
        end_s=3.0,
        score=0.8,
        keep=True,
        is_duplicate=False,
        reject_reasons=[],
        created_at=now_utc(),
    )
    seg_vid = AssetSegment(
        id=next_id("seg"),
        tenant_id="t1",
        event_id="ev1",
        asset_id="vid1",
        start_s=0.0,
        end_s=6.0,
        score=0.9,
        keep=True,
        is_duplicate=False,
        reject_reasons=[],
        created_at=now_utc(),
    )
    SegmentRepository.replace_for_asset("img1", "ev1", [seg_img])
    SegmentRepository.replace_for_asset("vid1", "ev1", [seg_vid])

    items = list_photo_curation_items("ev1")
    assert len(items) == 1
    assert items[0].asset_id == "img1"


def test_kept_photo_export_filters_duplicates() -> None:
    EventRepository.create(
        Event(
            id="ev2",
            tenant_id="t1",
            title="E2",
            event_type="test",
            predefined_tags=[],
            ocr_languages=["en"],
            created_at=now_utc(),
        )
    )
    for aid, keep, dup in (("i1", True, False), ("i2", True, True), ("i3", False, False)):
        AssetRepository.create(
            Asset(
                id=aid,
                tenant_id="t1",
                event_id="ev2",
                media_path=f"m/{aid}.jpg",
                media_type="image",
                created_at=now_utc(),
            )
        )
        seg = AssetSegment(
            id=next_id("seg"),
            tenant_id="t1",
            event_id="ev2",
            asset_id=aid,
            start_s=0.0,
            end_s=3.0,
            score=0.5,
            keep=keep,
            is_duplicate=dup,
            reject_reasons=[],
            created_at=now_utc(),
        )
        SegmentRepository.replace_for_asset(aid, "ev2", [seg])

    kept = kept_photo_items_for_export("ev2")
    assert [x.asset_id for x in kept] == ["i1"]


def test_photo_curation_api_requires_tenant() -> None:
    client = TestClient(app)
    event = client.post("/events", json={"tenant_id": "tenant_a", "title": "E", "event_type": "t"}).json()
    event_id = event["id"]
    r = client.get(f"/events/{event_id}/photos/curation", params={"tenant_id": "tenant_a"})
    assert r.status_code == 200
    assert r.json()["event_id"] == event_id
    assert r.json()["items"] == []


def _seg(
    event_id: str,
    asset_id: str,
    *,
    tenant_id: str = "t1",
    keep: bool = True,
    is_duplicate: bool = False,
) -> AssetSegment:
    return AssetSegment(
        id=next_id("seg"),
        tenant_id=tenant_id,
        event_id=event_id,
        asset_id=asset_id,
        start_s=0.0,
        end_s=3.0,
        score=0.8,
        keep=keep,
        is_duplicate=is_duplicate,
        reject_reasons=[],
        created_at=now_utc(),
    )


def test_asset_media_allows_file_under_project_root() -> None:
    media_path = PROJECT_ROOT / "storage" / "_pytest_asset_media.png"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(_TINY_PNG)
    try:
        EventRepository.create(
            Event(
                id="ev_media_ok",
                tenant_id="t1",
                title="M",
                event_type="test",
                predefined_tags=[],
                ocr_languages=["en"],
                created_at=now_utc(),
            )
        )
        AssetRepository.create(
            Asset(
                id="img_ok",
                tenant_id="t1",
                event_id="ev_media_ok",
                media_path=str(media_path),
                media_type="image",
                created_at=now_utc(),
            )
        )
        SegmentRepository.replace_for_asset("img_ok", "ev_media_ok", [_seg("ev_media_ok", "img_ok")])

        client = TestClient(app)
        r = client.get(
            "/events/ev_media_ok/assets/img_ok/media",
            params={"tenant_id": "t1"},
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("image/")
        r_preview = client.get(
            "/events/ev_media_ok/assets/img_ok/media",
            params={"tenant_id": "t1", "max_edge": 64},
        )
        assert r_preview.status_code == 200
        assert r_preview.headers.get("content-type", "").startswith("image/")
    finally:
        media_path.unlink(missing_ok=True)


def test_asset_media_rejects_path_outside_allowed_roots() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_PNG)
        tmp_path = Path(f.name)
    try:
        EventRepository.create(
            Event(
                id="ev_media_bad",
                tenant_id="t1",
                title="M2",
                event_type="test",
                predefined_tags=[],
                ocr_languages=["en"],
                created_at=now_utc(),
            )
        )
        AssetRepository.create(
            Asset(
                id="img_bad",
                tenant_id="t1",
                event_id="ev_media_bad",
                media_path=str(tmp_path),
                media_type="image",
                created_at=now_utc(),
            )
        )
        SegmentRepository.replace_for_asset("img_bad", "ev_media_bad", [_seg("ev_media_bad", "img_bad")])

        client = TestClient(app)
        r = client.get(
            "/events/ev_media_bad/assets/img_bad/media",
            params={"tenant_id": "t1"},
        )
        assert r.status_code == 400
        assert "outside allowed" in (r.json().get("detail") or "").lower()
    finally:
        tmp_path.unlink(missing_ok=True)


def test_asset_media_cross_tenant_forbidden() -> None:
    media_path = PROJECT_ROOT / "storage" / "_pytest_asset_media_tenant.png"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(_TINY_PNG)
    try:
        EventRepository.create(
            Event(
                id="ev_media_t",
                tenant_id="tenant_a",
                title="M",
                event_type="test",
                predefined_tags=[],
                ocr_languages=["en"],
                created_at=now_utc(),
            )
        )
        AssetRepository.create(
            Asset(
                id="img_t",
                tenant_id="tenant_a",
                event_id="ev_media_t",
                media_path=str(media_path),
                media_type="image",
                created_at=now_utc(),
            )
        )
        SegmentRepository.replace_for_asset(
            "img_t", "ev_media_t", [_seg("ev_media_t", "img_t", tenant_id="tenant_a")]
        )

        client = TestClient(app)
        r = client.get(
            "/events/ev_media_t/assets/img_t/media",
            params={"tenant_id": "tenant_b"},
        )
        assert r.status_code == 403
    finally:
        media_path.unlink(missing_ok=True)


def test_export_kept_skips_disallowed_paths_but_keeps_allowed() -> None:
    good = PROJECT_ROOT / "storage" / "_pytest_export_good.png"
    good.parent.mkdir(parents=True, exist_ok=True)
    good.write_bytes(_TINY_PNG)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_PNG)
        bad = Path(f.name)
    try:
        EventRepository.create(
            Event(
                id="ev_zip_mix",
                tenant_id="t1",
                title="Z",
                event_type="test",
                predefined_tags=[],
                ocr_languages=["en"],
                created_at=now_utc(),
            )
        )
        for aid, p in (("img_g", str(good)), ("img_b", str(bad))):
            AssetRepository.create(
                Asset(
                    id=aid,
                    tenant_id="t1",
                    event_id="ev_zip_mix",
                    media_path=p,
                    media_type="image",
                    created_at=now_utc(),
                )
            )
            SegmentRepository.replace_for_asset(aid, "ev_zip_mix", [_seg("ev_zip_mix", aid)])

        client = TestClient(app)
        r = client.get("/events/ev_zip_mix/photos/export-kept", params={"tenant_id": "t1"})
        assert r.status_code == 200
        assert r.headers.get("content-type") == "application/zip"

        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            names = zf.namelist()
        assert any(n.endswith("img_g.png") for n in names)
        assert not any("img_b" in n for n in names)
    finally:
        good.unlink(missing_ok=True)
        bad.unlink(missing_ok=True)


def test_export_kept_fails_when_no_allowed_files() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_PNG)
        tmp_path = Path(f.name)
    try:
        EventRepository.create(
            Event(
                id="ev_zip_bad",
                tenant_id="t1",
                title="Z2",
                event_type="test",
                predefined_tags=[],
                ocr_languages=["en"],
                created_at=now_utc(),
            )
        )
        AssetRepository.create(
            Asset(
                id="img_only_tmp",
                tenant_id="t1",
                event_id="ev_zip_bad",
                media_path=str(tmp_path),
                media_type="image",
                created_at=now_utc(),
            )
        )
        SegmentRepository.replace_for_asset(
            "img_only_tmp", "ev_zip_bad", [_seg("ev_zip_bad", "img_only_tmp")]
        )

        client = TestClient(app)
        r = client.get("/events/ev_zip_bad/photos/export-kept", params={"tenant_id": "t1"})
        assert r.status_code == 400
        assert "archive" in (r.json().get("detail") or "").lower()
    finally:
        tmp_path.unlink(missing_ok=True)
