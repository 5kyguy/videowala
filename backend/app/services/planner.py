from __future__ import annotations

from ..repositories import AssetRepository, PlanRepository
from ..schemas import ContentRequestCreate, PlannerAction, PlannerPlan
from .search import semantic_search
from .privacy import audit_action


class PlannerValidationError(ValueError):
    pass


def build_plan(request: ContentRequestCreate, event_context: dict) -> PlannerPlan:
    # Stage-1 deterministic planner skeleton with strict schema.
    allowed_media = set(request.include_media_types)

    def media_type_allowed(asset_id: str) -> bool:
        if not allowed_media:
            return True
        asset = AssetRepository.get(asset_id)
        if asset is None:
            return False
        return asset.media_type in allowed_media

    candidate_asset_ids: list[str] = []
    for bucket in ("vlm_caption", "vlm_tags", "face_matches"):
        for item in event_context.get(bucket, []):
            if (
                item["asset_id"] not in request.excluded_asset_ids
                and media_type_allowed(item["asset_id"])
            ):
                candidate_asset_ids.append(item["asset_id"])

    if request.prompt:
        try:
            hits = semantic_search(
                tenant_id=request.tenant_id,
                event_id=request.event_id,
                query=request.prompt,
                limit=30,
            )
            candidate_asset_ids = [h["asset_id"] for h in hits] + candidate_asset_ids
        except Exception:
            pass

    dedup_ids = list(dict.fromkeys(candidate_asset_ids))
    # Respect explicit includes first for user-guided regenerate loops.
    include_first = [asset_id for asset_id in request.include_asset_ids if asset_id in dedup_ids]
    remaining = [asset_id for asset_id in dedup_ids if asset_id not in include_first]
    dedup_ids = (include_first + remaining)[:30]
    actions = [
        PlannerAction(action="select_segments", params={"asset_ids": dedup_ids}),
        PlannerAction(action="set_order", params={"strategy": "chronological"}),
        PlannerAction(action="set_duration", params={"seconds": request.target_duration_seconds}),
        PlannerAction(action="render_preview", params={"format": "mp4"}),
    ]
    if request.render_subtitles:
        actions.append(PlannerAction(action="render_subtitles", params={"source": "asr", "style": "default"}))
    if request.render_overlays:
        actions.append(
            PlannerAction(
                action="render_overlays",
                params={"source": "ocr", "max_items": 5, "strategy": "keyframes"},
            )
        )
    plan = PlannerPlan(
        tenant_id=request.tenant_id,
        event_id=request.event_id,
        output_type=request.output_type,
        rationale="Plan generated from VLM context and face match context.",
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
        "render_subtitles",
        "render_overlays",
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
