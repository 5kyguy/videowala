from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from ..db import now_utc
from ..repositories import (
    AssetRepository,
    EventRepository,
    IndexJobRepository,
    InsightRepository,
    PersonReferenceRepository,
    PersonRepository,
    SegmentRepository,
    next_id,
)

from ..config import settings, ocr_trigger_tags_set
from ..schemas import Asset, AssetInsight, AssetSegment, InsightType
from ..vector_store import upsert_asset_vector
from .asr import asr_service
from .embeddings import build_embedding_text, embedding_service
from .faces import face_service
from .ingest import ensure_asset_proxy
from .ocr import ocr_service
from .photo_curation import apply_photo_semantic_cull_for_event
from .privacy import audit_action
from .vlm import VlmResult, vlm_service

logger = logging.getLogger(__name__)


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


def _base_cull_score(
    asset: Asset,
    manifest: dict,
    asr_segments: list[dict],
    ocr_items: list[dict],
    *,
    caption_confidence: float | None,
    vlm_tags: list[str],
) -> float:
    """
    PoC cull score: resolution, audio, ASR/OCR signals, VLM confidence and negative tags.
    No bonus for face detections or person matches (per product spec).
    """
    score = 0.45
    width = int(manifest.get("width", 0) or 0)
    height = int(manifest.get("height", 0) or 0)
    if width >= 960 or height >= 720:
        score += 0.1
    if bool(manifest.get("has_audio")):
        score += 0.05
    if asr_segments:
        score += 0.1
    if ocr_items:
        score += 0.05
    if asset.media_type == "image":
        score -= 0.10
    if caption_confidence is not None:
        score += 0.05 * max(0.0, min(1.0, caption_confidence))
    neg = {"blur", "blurry", "dark", "underexposed", "noisy", "noise"}
    tag_low = {t.lower() for t in vlm_tags}
    if neg & tag_low:
        score -= 0.08
    return max(0.0, min(1.0, round(score, 4)))


def _segment_signature(asset: Asset, start_s: float, end_s: float) -> str:
    seed = f"{asset.media_path}|{asset.media_type}|{start_s:.2f}|{end_s:.2f}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


def _should_run_ocr(vlm_tags: list[str]) -> bool:
    triggers = ocr_trigger_tags_set()
    lowered = {t.strip().lower() for t in vlm_tags if t.strip()}
    return bool(lowered & triggers)


def _people_names_from_matches(matches: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for m in matches:
        n = str(m.get("name") or "").strip()
        if n and n.lower() not in seen and n != "unknown":
            seen.add(n.lower())
            names.append(n)
    return names


def _delete_index_insights(asset: Asset) -> None:
    InsightRepository.delete_for_asset(
        event_id=asset.event_id,
        asset_id=asset.id,
        insight_types=[
            InsightType.vlm_caption.value,
            InsightType.vlm_tags.value,
            InsightType.face_detections.value,
            InsightType.face_matches.value,
            InsightType.cull_metrics.value,
            InsightType.ocr_text.value,
            InsightType.asr_transcript.value,
            InsightType.semantic_embedding.value,
        ],
    )


def _event_context_fields(event) -> tuple[dict | None, list[str], list[str]]:
    if event is None:
        return None, [], ["en"]
    event_ctx = {
        "title": event.title,
        "event_type": event.event_type,
        "venue": event.venue,
        "date": event.date,
    }
    predefined = list(event.predefined_tags)
    ocr_langs = list(event.ocr_languages) if event.ocr_languages else ["en"]
    return event_ctx, predefined, ocr_langs


@dataclass
class _AssetIndexState:
    asset: Asset
    proxy: Any
    analysis_media_path: str
    detections: list[dict] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    asr_segments: list[dict] = field(default_factory=list)
    vlm: VlmResult | None = None
    ocr_items: list[dict] = field(default_factory=list)
    ocr_model: str = ""
    run_ocr: bool = False


def _insights_vlm_rows(asset: Asset, vlm: VlmResult) -> list[AssetInsight]:
    return [
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.vlm_caption,
            payload={
                "text": vlm.caption,
                "model": vlm.model,
                "caption_confidence": vlm.caption_confidence,
            },
            confidence=float(vlm.caption_confidence) if vlm.caption_confidence is not None else 0.65,
            created_at=now_utc(),
        ),
        AssetInsight(
            id=next_id("insight"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            insight_type=InsightType.vlm_tags,
            payload={
                "tags": vlm.tags,
                "tags_from_predefined": vlm.tags_from_predefined,
                "tags_added": vlm.tags_added,
                "model": vlm.model,
            },
            confidence=0.62,
            created_at=now_utc(),
        ),
    ]


def _insights_face_rows(asset: Asset, detections: list[dict], matches: list[dict]) -> list[AssetInsight]:
    return [
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


def _insights_cull_row(
    asset: Asset,
    proxy,
    segments: list[AssetSegment],
    base_score: float,
) -> AssetInsight:
    return AssetInsight(
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


def _insights_ocr_row(asset: Asset, ocr_items: list[dict], ocr_model: str, run_ocr: bool) -> AssetInsight:
    return AssetInsight(
        id=next_id("insight"),
        tenant_id=asset.tenant_id,
        event_id=asset.event_id,
        asset_id=asset.id,
        insight_type=InsightType.ocr_text,
        payload={"items": ocr_items, "model": ocr_model, "skipped": not run_ocr},
        confidence=0.6 if ocr_items else 0.0,
        created_at=now_utc(),
    )


def _index_job_progress(job_id: str | None, pct: int, stage: str) -> None:
    if not job_id:
        return
    pct = max(0, min(100, int(pct)))
    IndexJobRepository.set_progress(job_id, pct, index_stage=stage)


def _stage_pct_linear(lo: int, hi: int, i: int, n: int) -> int:
    """Map step i in [0..n-1] to [lo..hi] without long int() plateaus (use rounding)."""
    if n <= 0:
        return lo
    v = lo + (hi - lo) * (i + 1) / n
    return max(lo, min(hi, int(round(v))))


def _insights_asr_row(asset: Asset, asr_segments: list[dict]) -> AssetInsight:
    return AssetInsight(
        id=next_id("insight"),
        tenant_id=asset.tenant_id,
        event_id=asset.event_id,
        asset_id=asset.id,
        insight_type=InsightType.asr_transcript,
        payload={"segments": asr_segments, "model": asr_service.model_name},
        confidence=0.65 if asr_segments else 0.0,
        created_at=now_utc(),
    )


def index_event_by_model_stages(
    asset_ids: list[str],
    *,
    semantic_prompt: str | None = None,
    index_job_id: str | None = None,
) -> list[AssetInsight]:
    """
    One GPU-heavy model at a time: run that stage over all assets, persist insights, unload, then next stage.
    Order: faces → ASR (video) → VLM → OCR → (segments + cull) → embeddings.
    """
    if not asset_ids:
        return []
    assets: list[Asset] = []
    for aid in asset_ids:
        a = AssetRepository.get(aid)
        if a is None:
            raise KeyError(f"Asset not found: {aid}")
        assets.append(a)
    event_id = assets[0].event_id
    for a in assets:
        if a.event_id != event_id:
            raise ValueError("All assets must belong to the same event")
    event = EventRepository.get(event_id)
    if event is None:
        raise KeyError(f"Event not found: {event_id}")

    event_ctx, predefined, ocr_langs = _event_context_fields(event)

    n_assets = len(assets)
    logger.info(
        "Index pipeline: event_id=%s assets=%d — prefetch (clear insights + proxies/metadata; can take minutes on huge folders)",
        event_id,
        n_assets,
    )
    _index_job_progress(
        index_job_id,
        8,
        f"Prefetch (metadata & proxies) · 0/{n_assets}",
    )

    states: list[_AssetIndexState] = []
    log_every = max(25, min(200, n_assets // 40 or 25))
    for i, asset in enumerate(assets):
        _delete_index_insights(asset)
        proxy = ensure_asset_proxy(asset)
        analysis_media_path = proxy.proxy_path if asset.media_type == "video" else asset.media_path
        states.append(_AssetIndexState(asset=asset, proxy=proxy, analysis_media_path=analysis_media_path))
        if n_assets and (i % 25 == 0 or i == n_assets - 1):
            pct_pf = _stage_pct_linear(8, 12, i, n_assets)
            _index_job_progress(
                index_job_id,
                pct_pf,
                f"Prefetch (metadata & proxies) · {i + 1}/{n_assets}",
            )
            if i % log_every == 0 or i == n_assets - 1:
                logger.info(
                    "Index prefetch %d/%d assets (~%d%% in this stage)",
                    i + 1,
                    n_assets,
                    pct_pf,
                )

    logger.info("Index prefetch done; phase 1/6: face model over %d asset(s)", n_assets)
    _index_job_progress(index_job_id, 12, f"Face detection · 0/{n_assets}")

    def _iter_states():
        if settings.indexing_show_progress and len(states) > 1:
            try:
                from tqdm import tqdm

                return tqdm(states, desc="Indexing assets", unit="file", file=sys.stderr, leave=False)
            except ImportError:
                pass
        return states

    all_insights: list[AssetInsight] = []

    # Phase 1: face model — all assets
    batch_face: list[AssetInsight] = []
    for fi, st in enumerate(_iter_states()):
        st.detections, st.matches = _face_detections_and_matches(st.asset, st.analysis_media_path)
        batch_face.extend(_insights_face_rows(st.asset, st.detections, st.matches))
        if n_assets and (fi % 25 == 0 or fi == n_assets - 1):
            _index_job_progress(
                index_job_id,
                _stage_pct_linear(12, 28, fi, n_assets),
                f"Face detection · {fi + 1}/{n_assets}",
            )
    if batch_face:
        InsightRepository.create_many(batch_face)
        all_insights.extend(batch_face)
    face_service.release()

    n_videos = sum(1 for s in states if s.asset.media_type == "video")
    logger.info("Index phase 2/6: ASR (%d video(s))", n_videos)
    if n_videos:
        _index_job_progress(index_job_id, 28, f"Speech (ASR) · 0/{n_videos}")
    else:
        _index_job_progress(index_job_id, 28, "Speech (ASR) (no videos)")

    # Phase 2: ASR — transcribe videos; images get an empty ASR insight (same shape as before)
    batch_asr: list[AssetInsight] = []
    v_i = 0
    for st in states:
        if st.asset.media_type != "video":
            st.asr_segments = []
        else:
            st.asr_segments = asr_service.transcribe(st.analysis_media_path)
            if n_videos and (v_i % 5 == 0 or v_i == n_videos - 1):
                _index_job_progress(
                    index_job_id,
                    _stage_pct_linear(28, 38, v_i, n_videos),
                    f"Speech (ASR) · {v_i + 1}/{n_videos}",
                )
            v_i += 1
        batch_asr.append(_insights_asr_row(st.asset, st.asr_segments))
    if batch_asr:
        InsightRepository.create_many(batch_asr)
        all_insights.extend(batch_asr)
    asr_service.release()

    logger.info("Index phase 3/6: VLM over %d asset(s)", n_assets)
    _index_job_progress(index_job_id, 38, f"Vision-language (VLM) · 0/{n_assets}")

    # Phase 3: VLM — all assets
    batch_vlm: list[AssetInsight] = []
    for vi, st in enumerate(states):
        st.vlm = vlm_service.caption_and_tags(
            media_path=st.analysis_media_path,
            media_type=st.asset.media_type,
            scratch_root=settings.scratch_root,
            event_context=event_ctx,
            predefined_tags=predefined,
        )
        assert st.vlm is not None
        batch_vlm.extend(_insights_vlm_rows(st.asset, st.vlm))
        if n_assets and (vi % 25 == 0 or vi == n_assets - 1):
            _index_job_progress(
                index_job_id,
                _stage_pct_linear(38, 58, vi, n_assets),
                f"Vision-language (VLM) · {vi + 1}/{n_assets}",
            )
    if batch_vlm:
        InsightRepository.create_many(batch_vlm)
        all_insights.extend(batch_vlm)
    vlm_service.release()

    logger.info("Index phase 4/6: OCR over %d asset(s)", n_assets)
    _index_job_progress(index_job_id, 58, f"OCR · 0/{n_assets}")

    # Phase 4: OCR — all assets
    batch_ocr: list[AssetInsight] = []
    for oi, st in enumerate(states):
        assert st.vlm is not None
        st.run_ocr = _should_run_ocr(st.vlm.tags)
        st.ocr_items, st.ocr_model = ocr_service.extract(
            st.analysis_media_path,
            run_ocr=st.run_ocr,
            ocr_languages=ocr_langs,
        )
        batch_ocr.append(_insights_ocr_row(st.asset, st.ocr_items, st.ocr_model, st.run_ocr))
        if n_assets and (oi % 25 == 0 or oi == n_assets - 1):
            _index_job_progress(
                index_job_id,
                _stage_pct_linear(58, 72, oi, n_assets),
                f"OCR · {oi + 1}/{n_assets}",
            )
    if batch_ocr:
        InsightRepository.create_many(batch_ocr)
        all_insights.extend(batch_ocr)
    ocr_service.release()

    logger.info("Index phase 5–6/6: segments, cull, embeddings for %d asset(s)", n_assets)
    _index_job_progress(index_job_id, 72, f"Embeddings & segments · 0/{n_assets}")

    # Phase 5–6: segments + cull (CPU), then embedding model once for all assets
    text_source_limit = 8000
    for ei, st in enumerate(states):
        assert st.vlm is not None
        asr_text_joined = " ".join([seg.get("text", "") for seg in st.asr_segments]).strip()
        ocr_text_joined = " ".join([item.get("text", "") for item in st.ocr_items]).strip()
        duration_s = float(st.proxy.manifest.get("duration_s", 0.0) or 0.0)
        base_score = _base_cull_score(
            st.asset,
            st.proxy.manifest,
            st.asr_segments,
            st.ocr_items,
            caption_confidence=st.vlm.caption_confidence,
            vlm_tags=st.vlm.tags,
        )
        segments = [
            AssetSegment(
                id=next_id("seg"),
                tenant_id=st.asset.tenant_id,
                event_id=st.asset.event_id,
                asset_id=st.asset.id,
                start_s=start_s,
                end_s=end_s,
                score=base_score,
                keep=True,
                is_duplicate=False,
                reject_reasons=[],
                created_at=now_utc(),
            )
            for (start_s, end_s) in _segment_for_asset(st.asset, duration_s)
        ]
        SegmentRepository.replace_for_asset(st.asset.id, st.asset.event_id, segments)
        cull_row = _insights_cull_row(st.asset, st.proxy, segments, base_score)
        InsightRepository.create_many([cull_row])
        all_insights.append(cull_row)

        people_names = _people_names_from_matches(st.matches)
        combined = build_embedding_text(
            caption=st.vlm.caption,
            tags=st.vlm.tags,
            people_names=people_names,
            asr=asr_text_joined,
            ocr=ocr_text_joined,
        )
        embed = embedding_service.embed_text(combined)
        try:
            row_id = upsert_asset_vector(
                tenant_id=st.asset.tenant_id,
                event_id=st.asset.event_id,
                asset_id=st.asset.id,
                kind="multi",
                vector=embed.vector,
                text_source=combined[:text_source_limit] if combined else None,
            )
            enabled = True
        except Exception:
            row_id = 0
            enabled = False
        sem_insight = AssetInsight(
            id=next_id("insight"),
            tenant_id=st.asset.tenant_id,
            event_id=st.asset.event_id,
            asset_id=st.asset.id,
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
        InsightRepository.create_many([sem_insight])
        all_insights.append(sem_insight)
        if n_assets and (ei % 25 == 0 or ei == n_assets - 1):
            _index_job_progress(
                index_job_id,
                _stage_pct_linear(72, 95, ei, n_assets),
                f"Embeddings & segments · {ei + 1}/{n_assets}",
            )
        pipeline = "image" if st.asset.media_type == "image" else "video"
        insight_count = 8
        audit_action(
            st.asset.tenant_id,
            st.asset.event_id,
            "asset_indexed",
            {"asset_id": st.asset.id, "insight_count": insight_count, "pipeline": pipeline},
        )

    embedding_service.release()
    logger.info("Index pipeline finished: event_id=%s assets=%d insights=%d", event_id, n_assets, len(all_insights))

    if semantic_prompt and semantic_prompt.strip() and any(a.media_type == "image" for a in assets):
        apply_photo_semantic_cull_for_event(
            tenant_id=assets[0].tenant_id,
            event_id=event_id,
            prompt=semantic_prompt.strip(),
            cull_percent=settings.image_index_semantic_cull_percent,
        )

    return all_insights


def index_image_asset(asset_id: str, *, semantic_prompt: str | None = None) -> list[AssetInsight]:
    """Image-only indexing: delegates to event-wide staged pipeline (batch of one)."""
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    if asset.media_type != "image":
        raise ValueError(f"Expected image asset, got media_type={asset.media_type!r}")
    return index_event_by_model_stages([asset_id], semantic_prompt=semantic_prompt)


def index_video_asset(asset_id: str) -> list[AssetInsight]:
    """Video-only indexing: delegates to event-wide staged pipeline (batch of one)."""
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    if asset.media_type != "video":
        raise ValueError(f"Expected video asset, got media_type={asset.media_type!r}")
    return index_event_by_model_stages([asset_id], semantic_prompt=None)


def index_asset(asset_id: str, *, semantic_prompt: str | None = None) -> list[AssetInsight]:
    """Index one asset via the staged pipeline (same path as batch: one model at a time, batch of one)."""
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    return index_event_by_model_stages([asset_id], semantic_prompt=semantic_prompt)


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
    face_service.release()
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
