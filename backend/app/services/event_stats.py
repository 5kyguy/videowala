"""Aggregate storage and composition stats for event summary (PoC)."""

from __future__ import annotations

from pathlib import Path

from ..config import PROJECT_ROOT
from ..schemas import Asset, RenderJob


def resolve_asset_media_path(media_path: str) -> Path | None:
    """Match ingest/API path resolution: absolute file or repo-relative."""
    direct = Path(media_path)
    if direct.is_file():
        return direct.resolve()
    rel = PROJECT_ROOT / media_path
    if rel.is_file():
        return rel.resolve()
    return None


def media_footprint(assets: list[Asset]) -> dict:
    """
    Sum on-disk sizes for registered media paths and count by extension / media_type.

    Returns keys: total_bytes, bytes_images, bytes_videos, files_found, files_missing,
    extension_counts (suffix lower, e.g. ".jpg" -> count).
    """
    total_bytes = 0
    bytes_images = 0
    bytes_videos = 0
    files_found = 0
    files_missing = 0
    extension_counts: dict[str, int] = {}

    for a in assets:
        raw = str(a.media_path or "")
        suffix = Path(raw).suffix.lower() or "(no extension)"
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        p = resolve_asset_media_path(raw)
        if p is None:
            files_missing += 1
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            files_missing += 1
            continue
        files_found += 1
        total_bytes += sz
        if a.media_type == "image":
            bytes_images += sz
        else:
            bytes_videos += sz

    return {
        "total_bytes": total_bytes,
        "bytes_images": bytes_images,
        "bytes_videos": bytes_videos,
        "files_found": files_found,
        "files_missing": files_missing,
        "extension_counts": extension_counts,
    }


def renders_output_bytes(renders: list[RenderJob]) -> int:
    """Sum file sizes for completed renders whose output file exists."""
    total = 0
    for job in renders:
        if job.status != "completed" or not job.output_path:
            continue
        p = Path(job.output_path)
        if not p.is_file():
            continue
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def top_extensions_by_count(extension_counts: dict[str, int], *, limit: int = 10) -> list[tuple[str, int]]:
    items = sorted(extension_counts.items(), key=lambda x: (-x[1], x[0]))
    return items[:limit]
