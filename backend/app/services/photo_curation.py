from __future__ import annotations

from ..repositories import AssetRepository, SegmentRepository
from ..schemas import PhotoCurationItem


def list_photo_curation_items(event_id: str) -> list[PhotoCurationItem]:
    """Per-image rows from segment culling for image assets only (one segment per photo today)."""
    segments = SegmentRepository.list_for_event(event_id, keep_only=False)
    assets = {a.id: a for a in AssetRepository.list_for_event(event_id)}
    items: list[PhotoCurationItem] = []
    for seg in segments:
        asset = assets.get(seg.asset_id)
        if asset is None or asset.media_type != "image":
            continue
        items.append(
            PhotoCurationItem(
                asset_id=seg.asset_id,
                segment_id=seg.id,
                score=float(seg.score),
                keep=bool(seg.keep),
                is_duplicate=bool(seg.is_duplicate),
                reject_reasons=list(seg.reject_reasons),
            )
        )
    items.sort(key=lambda it: (-float(it.score), it.asset_id))
    return items


def kept_photo_items_for_export(event_id: str) -> list[PhotoCurationItem]:
    """Photos to ship in export: kept and not marked duplicate."""
    return [it for it in list_photo_curation_items(event_id) if it.keep and not it.is_duplicate]
