from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from ..config import settings
from ..db import now_utc
from ..repositories import AssetProxyRepository, AssetRepository, next_id
from ..schemas import Asset, AssetProxy, AssetRegister
from .privacy import audit_action

logger = logging.getLogger(__name__)

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


def _duration_from_metadata(metadata: dict) -> float:
    try:
        ffprobe = metadata.get("ffprobe", {})
        duration = ffprobe.get("format", {}).get("duration")
        if duration is None:
            return 0.0
        return float(duration)
    except Exception:
        return 0.0


def _video_stream_fields(metadata: dict) -> dict:
    ffprobe = metadata.get("ffprobe", {})
    streams = ffprobe.get("streams", []) if isinstance(ffprobe, dict) else []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "fps_raw": video.get("avg_frame_rate") or video.get("r_frame_rate"),
        "has_audio": bool(audio),
    }


def _fps_from_ratio(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            num = float(left)
            den = float(right)
            return round(num / den, 3) if den else 0.0
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _proxy_output_path(asset: Asset) -> Path:
    return Path(settings.storage_root) / asset.tenant_id / asset.event_id / "proxies" / f"{asset.id}.mp4"


def _build_video_proxy(source: Path, output: Path) -> tuple[bool, str | None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        "scale='min(960,iw)':-2,fps=12",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "27",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, (exc.stderr or exc.stdout or str(exc))[:500]


def _build_frame_manifest(duration_s: float) -> dict:
    if duration_s <= 0:
        return {"sample_timestamps_s": [0.0]}
    points = sorted(
        set(
            [
                round(min(1.0, duration_s * 0.05), 3),
                round(duration_s * 0.25, 3),
                round(duration_s * 0.5, 3),
                round(duration_s * 0.75, 3),
                round(max(0.0, duration_s - 1.0), 3),
            ]
        )
    )
    return {"sample_timestamps_s": points}


PROXY_MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB minimum free space


def _has_disk_budget(target_dir: Path) -> bool:
    try:
        stat = shutil.disk_usage(str(target_dir.parent if not target_dir.exists() else target_dir))
        return stat.free >= PROXY_MIN_FREE_BYTES
    except OSError:
        return True


def ensure_asset_proxy(asset: Asset) -> AssetProxy:
    metadata = extract_media_metadata(asset.media_path)
    duration_s = _duration_from_metadata(metadata)
    stream = _video_stream_fields(metadata)
    proxy_path = asset.media_path
    manifest = {
        "duration_s": round(duration_s, 3),
        "fps": _fps_from_ratio(stream.get("fps_raw")),
        "width": stream.get("width", 0),
        "height": stream.get("height", 0),
        "has_audio": bool(stream.get("has_audio", False)),
    }
    manifest.update(_build_frame_manifest(duration_s))

    source = Path(asset.media_path)
    if asset.media_type == "video" and source.exists():
        out = _proxy_output_path(asset)
        if not _has_disk_budget(out):
            logger.warning("Skipping proxy for %s — less than 2 GB free disk space.", asset.id)
            manifest["proxy_error"] = "insufficient_disk_space"
        else:
            ok, err = _build_video_proxy(source, out)
            if ok and out.exists():
                proxy_path = str(out.resolve())
            else:
                manifest["proxy_error"] = err or "proxy_generation_failed"

    proxy = AssetProxy(
        asset_id=asset.id,
        proxy_path=proxy_path,
        metadata=metadata,
        manifest=manifest,
        created_at=now_utc(),
    )
    AssetProxyRepository.upsert(proxy)
    return proxy


def purge_event_proxies(tenant_id: str, event_id: str) -> dict:
    """Delete all proxy files for an event. Safe to call after indexing is complete."""
    proxy_dir = Path(settings.storage_root) / tenant_id / event_id / "proxies"
    removed = 0
    freed_bytes = 0
    if proxy_dir.is_dir():
        for f in proxy_dir.iterdir():
            if f.is_file():
                try:
                    freed_bytes += f.stat().st_size
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        try:
            proxy_dir.rmdir()
        except OSError:
            pass
    audit_action(tenant_id, event_id, "proxies_purged", {"removed": removed, "freed_bytes": freed_bytes})
    return {"removed": removed, "freed_bytes": freed_bytes}


def build_asset_manifest(asset: Asset) -> dict:
    proxy = ensure_asset_proxy(asset)
    return {
        "asset_id": asset.id,
        "tenant_id": asset.tenant_id,
        "event_id": asset.event_id,
        "media_type": asset.media_type,
        "media_path": asset.media_path,
        "proxy_path": proxy.proxy_path,
        "metadata": proxy.metadata,
        "manifest": proxy.manifest,
    }
