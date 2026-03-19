from __future__ import annotations

from ..db import now_utc
from ..repositories import (
    AssetRepository,
    InsightRepository,
    PersonReferenceRepository,
    PersonRepository,
    next_id,
)
from ..config import settings
from ..schemas import Asset, AssetInsight, InsightType
from ..vector_store import upsert_asset_vector
from .asr import asr_service
from .embeddings import embedding_service
from .faces import face_service
from .ocr import ocr_service
from .privacy import audit_action
from .vlm import vlm_service

def _stub_face_matches(asset: Asset) -> list[dict]:
    detections = face_service.detect_faces(asset.media_path)
    references = []
    person_lookup = {person.id: person.display_name for person in PersonRepository.list_for_event(asset.tenant_id, asset.event_id)}
    for ref in PersonReferenceRepository.list_for_event(asset.tenant_id, asset.event_id):
        references.append(
            {
                "person_id": ref.person_id,
                "display_name": person_lookup.get(ref.person_id, "unknown"),
                "embedding": ref.embedding,
            }
        )
    return face_service.match_faces(detections, references)


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
        ],
    )

    detections = face_service.detect_faces(asset.media_path)
    matches = _stub_face_matches(asset)

    ocr_items, ocr_model = ocr_service.extract(asset.media_path)

    asr_segments = asr_service.transcribe(asset.media_path) if asset.media_type == "video" else []

    ocr_text_joined = " ".join([item.get("text", "") for item in ocr_items]).strip()
    asr_text_joined = " ".join([seg.get("text", "") for seg in asr_segments]).strip()

    vlm = vlm_service.caption_and_tags(
        media_path=asset.media_path,
        media_type=asset.media_type,
        scratch_root=settings.scratch_root,
    )

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
