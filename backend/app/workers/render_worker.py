from __future__ import annotations

from ..schemas import PlannerPlan
from ..services.rendering import create_render_job


def run_render_job(tenant_id: str, event_id: str, plan: PlannerPlan) -> str:
    job = create_render_job(tenant_id=tenant_id, event_id=event_id, plan=plan)
    return job.id
