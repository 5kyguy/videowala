from __future__ import annotations

from ..repositories import AssetRepository, SegmentRepository
from ..schemas import PhotoCurationItem, PhotoCurationRequest
from .search import semantic_search


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


def score_photo_segments(request: PhotoCurationRequest) -> list[PhotoCurationItem]:
    """Score and cull image segments using user prompt context.

    Blends semantic similarity with base cull score, then keeps top cull_percent of images.
    Persists score/keep/duplicate/reasons back to segment records.
    """
    segments = SegmentRepository.list_for_event(request.event_id, keep_only=False)
    assets = {a.id: a for a in AssetRepository.list_for_event(request.event_id)}

    image_segments = [seg for seg in segments if assets.get(seg.asset_id, {}).media_type == "image"]
    if not image_segments:
        return []

    semantic_scores: dict[str, float] = {}
    if request.prompt and request.prompt.strip():
        try:
            hits = semantic_search(
                tenant_id=request.tenant_id,
                event_id=request.event_id,
                query=request.prompt,
                limit=100,
            )
            for h in hits:
                aid = h.get("asset_id")
                if aid:
                    sc = float(h.get("score", 0.0) or 0.0)
                    semantic_scores[str(aid)] = max(semantic_scores.get(str(aid), 0.0), sc)
        except Exception:
            semantic_scores = {}

    scored: list[dict] = []

    face_set: set[str] = set(request.include_faces)

    for seg in image_segments:
        asset = assets.get(seg.asset_id)
        if asset is None:
            continue

        base_score = float(seg.score)
        sem = semantic_scores.get(seg.asset_id, 0.0)

        relevance = min(1.0, 0.38 * sem)
        face_bonus = 0.0
        if face_set:
            pass

        final_score = max(0.0, min(1.0, round(base_score + relevance + face_bonus, 4)))

        scored.append({
            "segment_id": seg.id,
            "asset_id": seg.asset_id,
            "score": final_score,
        })

    keep_map: dict[str, tuple[float, bool, list[str]]] = {}
    if scored:
        scored.sort(key=lambda x: -x["score"])
        keep_count = max(1, int(len(scored) * request.cull_percent))

        final_updates: list[tuple[str, float, bool, bool, list[str]]] = []
        for i, row in enumerate(scored):
            keep = i < keep_count
            reasons = [] if keep else ["culled_by_user_request"]
            final_updates.append((row["segment_id"], row["score"], keep, False, reasons))
            keep_map[row["segment_id"]] = (row["score"], keep, reasons)

        SegmentRepository.batch_update_culling(final_updates)

    items: list[PhotoCurationItem] = []
    for seg in image_segments:
        asset = assets.get(seg.asset_id)
        if asset is None:
            continue
        score, keep, reasons = keep_map.get(seg.id, (float(seg.score), True, []))
        items.append(
            PhotoCurationItem(
                asset_id=seg.asset_id,
                segment_id=seg.id,
                score=score,
                keep=keep,
                is_duplicate=False,
                reject_reasons=reasons,
            )
        )
    items.sort(key=lambda it: (-float(it.score), it.asset_id))
    return items
