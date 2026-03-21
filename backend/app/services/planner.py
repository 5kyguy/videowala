from __future__ import annotations

from ..repositories import AssetRepository, PlanRepository, SegmentRepository
from ..schemas import Asset, ContentRequestCreate, OutputType, PlannerAction, PlannerPlan
from .search import semantic_search
from .privacy import audit_action


class PlannerValidationError(ValueError):
    pass


def _tokenize(text: str) -> set[str]:
    return {tok.strip().lower() for tok in text.replace(",", " ").replace(".", " ").split() if tok.strip()}


def _asset_text(item: dict) -> str:
    tags = item.get("tags", [])
    tags_text = " ".join([str(t) for t in tags]) if isinstance(tags, list) else ""
    return f"{item.get('text', '')} {tags_text}".strip()


def _build_asset_context(event_context: dict) -> dict[str, dict]:
    by_asset: dict[str, dict] = {}
    for bucket in ("vlm_caption", "vlm_tags", "face_matches", "ocr_text", "asr_transcript"):
        for item in event_context.get(bucket, []):
            aid = item.get("asset_id")
            if not aid:
                continue
            current = by_asset.setdefault(aid, {"text": [], "faces": set()})
            if bucket in ("vlm_caption", "vlm_tags"):
                current["text"].append(_asset_text(item))
            if bucket == "ocr_text":
                current["text"].extend([str(x.get("text", "")) for x in item.get("items", [])][:8])
            if bucket == "asr_transcript":
                current["text"].extend([str(x.get("text", "")) for x in item.get("segments", [])][:8])
            if bucket == "face_matches":
                for match in item.get("matches", []):
                    pid = match.get("person_id")
                    if pid:
                        current["faces"].add(pid)
    return by_asset


def _semantic_asset_scores(request: ContentRequestCreate) -> dict[str, float]:
    if not (request.prompt and request.prompt.strip()):
        return {}
    try:
        hits = semantic_search(
            tenant_id=request.tenant_id,
            event_id=request.event_id,
            query=request.prompt,
            limit=50,
        )
    except Exception:
        return {}
    out: dict[str, float] = {}
    for h in hits:
        aid = h.get("asset_id")
        if not aid:
            continue
        sc = float(h.get("score", 0.0) or 0.0)
        out[aid] = max(out.get(aid, 0.0), sc)
    return out


def _diversify_by_asset(rows: list[dict], max_per_asset: int = 4) -> list[dict]:
    """Keep score ordering but cap how many segments we take per asset (highlight reel)."""
    sorted_rows = sorted(rows, key=lambda x: -float(x.get("score", 0.0)))
    counts: dict[str, int] = {}
    out: list[dict] = []
    for r in sorted_rows:
        aid = r["asset_id"]
        if counts.get(aid, 0) >= max_per_asset:
            continue
        out.append(r)
        counts[aid] = counts.get(aid, 0) + 1
    return out


def _order_kept_segments(
    kept: list[dict],
    output_type: OutputType,
    asset_map: dict[str, Asset],
    asset_ctx: dict[str, dict],
    request: ContentRequestCreate,
) -> list[dict]:
    if output_type == OutputType.chronological_film:
        return sorted(
            kept,
            key=lambda r: (
                asset_map[r["asset_id"]].created_at.isoformat() if r["asset_id"] in asset_map else "",
                float(r.get("start_s", 0.0)),
            ),
        )
    if output_type == OutputType.person_focus_reel:

        def pkey(r: dict) -> tuple:
            faces = asset_ctx.get(r["asset_id"], {}).get("faces", set())
            hit = bool(request.include_faces and any(f in faces for f in request.include_faces))
            return (0 if hit else 1, -float(r.get("score", 0.0)))

        return sorted(kept, key=pkey)
    if output_type == OutputType.highlight_reel:
        return _diversify_by_asset(kept, max_per_asset=4)
    return sorted(kept, key=lambda x: -float(x.get("score", 0.0)))


def _filter_person_focus_pool(
    kept: list[dict],
    asset_ctx: dict[str, dict],
    request: ContentRequestCreate,
) -> list[dict]:
    if not request.include_faces:
        return kept
    matched = [
        r
        for r in kept
        if any(
            f in asset_ctx.get(r["asset_id"], {}).get("faces", set())
            for f in request.include_faces
        )
    ]
    if len(matched) >= max(4, min(len(kept), len(kept) // 3 + 1)):
        return matched
    return kept


def _set_order_strategy(output_type: OutputType) -> str:
    if output_type == OutputType.chronological_film:
        return "chronological_capture"
    if output_type == OutputType.person_focus_reel:
        return "person_focus"
    if output_type == OutputType.highlight_reel:
        return "highlight_diverse"
    return "ranked"


def _score_segments(
    request: ContentRequestCreate,
    event_context: dict,
    output_type: OutputType,
    asset_ctx: dict[str, dict],
    semantic_by_asset: dict[str, float],
) -> list[dict]:
    segments = SegmentRepository.list_for_event(request.event_id, keep_only=False)
    if not segments:
        return []

    assets = AssetRepository.list_for_event(request.event_id)
    asset_map = {a.id: a for a in assets}

    prompt_tokens = _tokenize(request.prompt)

    signature_top: dict[str, tuple[str, float]] = {}
    scored: list[dict] = []
    culling_updates: list[tuple[str, float, bool, bool, list[str]]] = []

    for seg in segments:
        asset = asset_map.get(seg.asset_id)
        if asset is None:
            continue
        if request.include_media_types and asset.media_type not in request.include_media_types:
            continue
        if seg.asset_id in request.excluded_asset_ids:
            continue

        ctx = asset_ctx.get(seg.asset_id, {"text": [], "faces": set()})
        text_tokens = _tokenize(" ".join(ctx["text"]))
        overlap = len(prompt_tokens.intersection(text_tokens)) / max(1, len(prompt_tokens))
        sem = float(semantic_by_asset.get(seg.asset_id, 0.0))

        face_bonus = 0.0
        if request.include_faces:
            hit = any(face_id in ctx["faces"] for face_id in request.include_faces)
            if output_type == OutputType.person_focus_reel:
                face_bonus = 0.22 if hit else -0.08
            else:
                face_bonus = 0.12 if hit else -0.05
        include_bonus = 0.15 if seg.asset_id in request.include_asset_ids else 0.0
        # Blend lexical overlap with semantic retrieval score (both in ~0..1).
        relevance = min(1.0, 0.32 * overlap + 0.38 * sem)
        final_score = max(
            0.0,
            min(1.0, round(float(seg.score) + relevance + face_bonus + include_bonus, 4)),
        )

        signature = f"{asset.media_type}:{asset.media_path}:{round(seg.start_s,1)}:{round(seg.end_s,1)}"
        prev = signature_top.get(signature)
        is_dup = prev is not None and prev[1] >= final_score
        if prev is None or final_score > prev[1]:
            signature_top[signature] = (seg.id, final_score)

        keep = final_score >= 0.35 and not is_dup
        reasons: list[str] = []
        if is_dup:
            reasons.append("duplicate_candidate")
        if final_score < 0.35:
            reasons.append("low_cull_score")
        culling_updates.append((seg.id, final_score, keep, is_dup, reasons))
        scored.append(
            {
                "segment_id": seg.id,
                "asset_id": seg.asset_id,
                "score": final_score,
                "start_s": seg.start_s,
                "end_s": seg.end_s,
                "keep": keep,
            }
        )

    SegmentRepository.batch_update_culling(culling_updates)

    kept = [row for row in scored if row["keep"]]
    if len(kept) < 4:
        scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
        fallback_updates: list[tuple[str, float, bool, bool, list[str]]] = []
        for row in scored_sorted[: max(4, len(kept))]:
            if not row["keep"]:
                fallback_updates.append((row["segment_id"], row["score"], True, False, ["fallback_sparse_pool"]))
                row["keep"] = True
        SegmentRepository.batch_update_culling(fallback_updates)
        kept = [row for row in scored if row["keep"]]

    if output_type == OutputType.person_focus_reel:
        kept = _filter_person_focus_pool(kept, asset_ctx, request)

    kept = _order_kept_segments(kept, output_type, asset_map, asset_ctx, request)
    return kept


def build_plan(request: ContentRequestCreate, event_context: dict) -> PlannerPlan:
    allowed_media = set(request.include_media_types)
    _asset_cache: dict[str, Asset | None] = {}

    def _get_asset(asset_id: str) -> Asset | None:
        if asset_id not in _asset_cache:
            _asset_cache[asset_id] = AssetRepository.get(asset_id)
        return _asset_cache[asset_id]

    def media_type_allowed(asset_id: str) -> bool:
        if not allowed_media:
            return True
        asset = _get_asset(asset_id)
        if asset is None:
            return False
        return asset.media_type in allowed_media

    asset_ctx = _build_asset_context(event_context)
    semantic_by_asset = _semantic_asset_scores(request)
    ranked_segments = _score_segments(
        request,
        event_context,
        request.output_type,
        asset_ctx,
        semantic_by_asset,
    )
    ranked_asset_ids = [row["asset_id"] for row in ranked_segments]
    ranked_segment_ids = [row["segment_id"] for row in ranked_segments]

    if not ranked_asset_ids:
        for bucket in ("vlm_caption", "vlm_tags", "face_matches"):
            for item in event_context.get(bucket, []):
                aid = item.get("asset_id")
                if not aid:
                    continue
                if aid in request.excluded_asset_ids:
                    continue
                if not media_type_allowed(aid):
                    continue
                ranked_asset_ids.append(aid)

    if request.prompt and len(ranked_asset_ids) < 12:
        try:
            hits = semantic_search(
                tenant_id=request.tenant_id,
                event_id=request.event_id,
                query=request.prompt,
                limit=30,
            )
            ranked_asset_ids.extend([h["asset_id"] for h in hits])
        except Exception:
            pass

    dedup_ids = [aid for aid in dict.fromkeys(ranked_asset_ids) if aid not in request.excluded_asset_ids and media_type_allowed(aid)]
    include_first = [asset_id for asset_id in request.include_asset_ids if asset_id in dedup_ids]
    remaining = [asset_id for asset_id in dedup_ids if asset_id not in include_first]
    dedup_ids = (include_first + remaining)[:30]
    ranked_segment_ids = ranked_segment_ids[:80]

    order_strategy = _set_order_strategy(request.output_type)
    rationale = (
        f"Plan: output_type={request.output_type.value}, order={order_strategy}, "
        "segment scoring blends lexical overlap + semantic retrieval + cull score."
    )

    actions = [
        PlannerAction(action="select_segments", params={"asset_ids": dedup_ids, "segment_ids": ranked_segment_ids}),
        PlannerAction(action="set_order", params={"strategy": order_strategy}),
        PlannerAction(action="set_duration", params={"seconds": request.target_duration_seconds}),
        PlannerAction(
            action="render_preview",
            params={"format": "mp4", "orientation": request.video_orientation},
        ),
    ]
    plan = PlannerPlan(
        tenant_id=request.tenant_id,
        event_id=request.event_id,
        output_type=request.output_type,
        rationale=rationale,
        actions=actions,
    )
    validate_plan(plan)
    plan_id = PlanRepository.create(plan)
    audit_action(request.tenant_id, request.event_id, "plan_created", {"plan_id": plan_id})
    return plan


def validate_plan(plan: PlannerPlan) -> None:
    allowed = {
        "select_segments",
        "set_order",
        "set_duration",
        "render_preview",
        "exclude_segments",
    }
    if not plan.actions:
        raise PlannerValidationError("Planner returned no actions.")
    for action in plan.actions:
        if action.action not in allowed:
            raise PlannerValidationError(f"Unsupported action: {action.action}")
    has_select = any(a.action == "select_segments" for a in plan.actions)
    has_render = any(a.action == "render_preview" for a in plan.actions)
    if not (has_select and has_render):
        raise PlannerValidationError("Plan must include select_segments and render_preview.")
