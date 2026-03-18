from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from ..db import now_utc
from ..repositories import AssetRepository, next_id
from ..schemas import Asset, AssetRegister
from .privacy import audit_action

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wmv", ".mpeg", ".mpg", ".3gp"}


def _validate_path_token(raw: str) -> None:
    illegal = ["..", ";", "&&", "|", "`", "$("]
    for token in illegal:
        if token in raw:
            raise ValueError(f"Unsafe path token: {token!r}")


def resolve_ingest_path(raw: str) -> Path:
    _validate_path_token(raw)
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()
    alt = PROJECT_ROOT / raw
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Path not found: {raw!r}")


def media_type_for_suffix(suffix: str) -> Literal["image", "video"] | None:
    s = suffix.lower()
    if s in IMAGE_EXTS:
        return "image"
    if s in VIDEO_EXTS:
        return "video"
    return None


def discover_media_files(root: Path, *, recursive: bool) -> list[tuple[Path, Literal["image", "video"]]]:
    """List (absolute path, media_type) for supported files under root (file or directory)."""
    out: list[tuple[Path, Literal["image", "video"]]] = []
    if root.is_file():
        mt = media_type_for_suffix(root.suffix)
        if mt:
            out.append((root.resolve(), mt))
        return out
    if not root.is_dir():
        return out
    iterator = root.rglob("*") if recursive else root.glob("*")
    for f in sorted(iterator):
        if not f.is_file():
            continue
        mt = media_type_for_suffix(f.suffix)
        if mt:
            out.append((f.resolve(), mt))
    return out


def create_asset_record(
    tenant_id: str,
    event_id: str,
    media_path: str,
    media_type: Literal["image", "video"],
) -> Asset:
    asset = Asset(
        id=next_id("asset"),
        tenant_id=tenant_id,
        event_id=event_id,
        media_path=media_path,
        media_type=media_type,
        created_at=now_utc(),
    )
    AssetRepository.create(asset)
    audit_action(tenant_id, event_id, "asset_registered", {"asset_id": asset.id, "media_path": media_path})
    return asset


def register_asset(payload: AssetRegister) -> Asset:
    return create_asset_record(payload.tenant_id, payload.event_id, payload.media_path, payload.media_type)


def extract_media_metadata(media_path: str) -> dict:
    path = Path(media_path)
    if not path.exists():
        return {"exists": False, "path": media_path}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"exists": True, "ffprobe": json.loads(result.stdout)}
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "ffprobe_error": str(exc)}


def build_asset_manifest(asset: Asset) -> dict:
    metadata = extract_media_metadata(asset.media_path)
    return {
        "asset_id": asset.id,
        "tenant_id": asset.tenant_id,
        "event_id": asset.event_id,
        "media_type": asset.media_type,
        "media_path": asset.media_path,
        "metadata": metadata,
    }
