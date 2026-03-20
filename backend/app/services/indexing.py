from __future__ import annotations

import hashlib

from ..db import now_utc
from ..repositories import (
    AssetRepository,
    InsightRepository,
    PersonReferenceRepository,
    PersonRepository,
    SegmentRepository,
    next_id,
)
from ..config import settings
from ..schemas import Asset, AssetInsight, AssetSegment, InsightType
from ..vector_store import upsert_asset_vector
from .asr import asr_service
from .embeddings import embedding_service
from .faces import face_service
from .ingest import ensure_asset_proxy
from .ocr import ocr_service
from .privacy import audit_action
from .vlm import vlm_service

def _face_detections_and_matches(asset: Asset, analysis_media_path: str) -> tuple[list[dict], list[dict]]:
    """Single detection pass on the same path used for VLM/OCR (proxy for video), then match against event references."""
    detections = face_service.detect_faces(analysis_media_path)
    person_lookup = {person.id: person.display_name for person in PersonRepository.list_for_event(asset.tenant_id, asset.event_id)}
    references = []
    for ref in PersonReferenceRepository.list_for_event(asset.tenant_id, asset.event_id):
        references.append(
            {
                "person_id": ref.person_id,
                "display_name": person_lookup.get(ref.person_id, "unknown"),
                "embedding": ref.embedding,
            }
        )
    matches = face_service.match_faces(detections, references)
    return detections, matches


def _segment_for_asset(asset: Asset, duration_s: float) -> list[tuple[float, float]]:
    if asset.media_type == "image":
        return [(0.0, 3.0)]
    if duration_s <= 0:
        return [(0.0, 4.0)]
    chunk = 6.0
    starts: list[float] = []
    cur = 0.0
    while cur < duration_s:
        starts.append(cur)
        cur += chunk
    out: list[tuple[float, float]] = []
    for start in starts:
        end = min(duration_s, start + chunk)
        if end - start >= 1.0:
            out.append((round(start, 3), round(end, 3)))
    if not out:
        out = [(0.0, min(6.0, round(duration_s, 3)))]
    return out[:40]


def _base_cull_score(asset: Asset, manifest: dict, detections: list[dict], asr_segments: list[dict], ocr_items: list[dict]) -> float:
    score = 0.45
    width = int(manifest.get("width", 0) or 0)
    height = int(manifest.get("height", 0) or 0)
    if width >= 960 or height >= 720:
        score += 0.1
    if bool(manifest.get("has_audio")):
        score += 0.05
    if detections:
        score += 0.1
    if asr_segments:
        score += 0.1
    if ocr_items:
        score += 0.05
    if asset.media_type == "image":
        score -= 0.03
    return max(0.0, min(1.0, round(score, 4)))


def _segment_signature(asset: Asset, start_s: float, end_s: float) -> str:
    seed = f"{asset.media_path}|{asset.media_type}|{start_s:.2f}|{end_s:.2f}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def index_asset(asset_id: str) -> list[AssetInsight]:
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    # Reindex clears only mutable machine-generated fields for this asset.
    InsightRepository.delete_for_asset(
        event_id=asset.event_id,
        asset_id=asset.id,
        insight_types=[
            InsightType.vlm_caption.value,
            InsightType.vlm_tags.value,
            InsightType.face_detections.value,
            InsightType.face_matches.value,
            InsightType.cull_metrics.value,
        ],
    )

    proxy = ensure_asset_proxy(asset)
    analysis_media_path = proxy.proxy_path if asset.media_type == "video" else asset.media_path

    detections, matches = _face_detections_and_matches(asset, analysis_media_path)

    ocr_items, ocr_model = ocr_service.extract(analysis_media_path)

    asr_segments = asr_service.transcribe(analysis_media_path) if asset.media_type == "video" else []

    ocr_text_joined = " ".join([item.get("text", "") for item in ocr_items]).strip()
    asr_text_joined = " ".join([seg.get("text", "") for seg in asr_segments]).strip()

    vlm = vlm_service.caption_and_tags(
        media_path=analysis_media_path,
        media_type=asset.media_type,
        scratch_root=settings.scratch_root,
    )

    duration_s = float(proxy.manifest.get("duration_s", 0.0) or 0.0)
    base_score = _base_cull_score(asset, proxy.manifest, detections, asr_segments, ocr_items)
    segments = [
        AssetSegment(
            id=next_id("seg"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            start_s=start_s,
            end_s=end_s,
            score=base_score,
            keep=True,
            is_duplicate=False,
            reject_reasons=[],
            created_at=now_utc(),
        )
        for (start_s, end_s) in _segment_for_asset(asset, duration_s)
    ]
    SegmentRepository.replace_for_asset(asset.id, asset.event_id, segments)

    insights = [
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.vlm_caption,
            payload={"text": vlm.caption, "model": vlm.model},
            confidence=0.65,
            created_at=now_utc(),
        ),
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.vlm_tags,
            payload={"tags": vlm.tags, "model": vlm.model},
            confidence=0.62,
            created_at=now_utc(),
        ),
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.face_detections,
            payload={"detections": detections, "model": face_service.model_name},
            confidence=0.58 if detections else 0.0,
            created_at=now_utc(),
        ),
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.face_matches,
            payload={"matches": matches, "model": face_service.model_name},
            confidence=0.51,
            created_at=now_utc(),
        ),
    ]

    insights.append(
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.cull_metrics,
            payload={
                "base_score": base_score,
                "proxy_path": proxy.proxy_path,
                "segment_count": len(segments),
                "segment_signatures": [
                    _segment_signature(asset, seg.start_s, seg.end_s)
                    for seg in segments
                ],
            },
            confidence=0.55,
            created_at=now_utc(),
        )
    )

    insights.append(
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.ocr_text,
            payload={"items": ocr_items, "model": ocr_model},
            confidence=0.6 if ocr_items else 0.0,
            created_at=now_utc(),
        )
    )

    insights.append(
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.asr_transcript,
            payload={"segments": asr_segments, "model": asr_service.model_name},
            confidence=0.65 if asr_segments else 0.0,
            created_at=now_utc(),
        )
    )

    combined = "\n".join([vlm.caption, asr_text_joined, ocr_text_joined]).strip()
    embed = embedding_service.embed_text(combined)
    try:
        row_id = upsert_asset_vector(
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            kind="multi",
            vector=embed.vector,
            text_source=combined[:2000] if combined else None,
        )
        enabled = True
    except Exception:
        row_id = 0
        enabled = False
    insights.append(
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.semantic_embedding,
            payload={
                "kind": "multi",
                "vector_ref": {"store": "pgvector", "row_id": row_id},
                "model": embed.model,
                "enabled": enabled,
            },
            confidence=0.5 if combined else 0.0,
            created_at=now_utc(),
        )
    )

    InsightRepository.create_many(insights)
    audit_action(asset.tenant_id, asset.event_id, "asset_indexed", {"asset_id": asset.id, "insight_count": len(insights)})
    return insights


def reindex_face_insights_for_asset(asset_id: str) -> None:
    """Replace only face detection/match insights; leaves VLM, OCR, ASR, embeddings unchanged. Fast path for UI reindex."""
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    InsightRepository.delete_for_asset(
        event_id=asset.event_id,
        asset_id=asset.id,
        insight_types=[
            InsightType.face_detections.value,
            InsightType.face_matches.value,
        ],
    )
    proxy = ensure_asset_proxy(asset)
    analysis_media_path = proxy.proxy_path if asset.media_type == "video" else asset.media_path
    detections, matches = _face_detections_and_matches(asset, analysis_media_path)
    insights = [
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.face_detections,
            payload={"detections": detections, "model": face_service.model_name},
            confidence=0.58 if detections else 0.0,
            created_at=now_utc(),
        ),
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.face_matches,
            payload={"matches": matches, "model": face_service.model_name},
            confidence=0.51,
            created_at=now_utc(),
        ),
    ]
    InsightRepository.create_many(insights)
    audit_action(asset.tenant_id, asset.event_id, "face_insights_reindexed", {"asset_id": asset.id})


def get_event_context(event_id: str) -> dict:
    insights = InsightRepository.list_for_event(event_id)
    grouped: dict[str, list[dict]] = {}
    for insight in insights:
        key = insight.insight_type.value
        grouped.setdefault(key, []).append({"asset_id": insight.asset_id, **insight.payload})
    return grouped


def get_event_context_filtered(event_id: str, insight_type: str | None = None, person_id: str | None = None) -> dict:
    insights = InsightRepository.list_for_event(event_id, insight_type=insight_type)
    grouped: dict[str, list[dict]] = {}
    for insight in insights:
        payload = {"asset_id": insight.asset_id, **insight.payload}
        if person_id and insight.insight_type == InsightType.face_matches:
            matches = payload.get("matches", [])
            matches = [item for item in matches if item.get("person_id") == person_id]
            if not matches:
                continue
            payload["matches"] = matches
        key = insight.insight_type.value
        grouped.setdefault(key, []).append(payload)
    return grouped
