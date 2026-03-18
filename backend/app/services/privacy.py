from __future__ import annotations

import shutil
from pathlib import Path

from ..config import settings
from ..repositories import AuditRepository


def assert_tenant_scope(request_tenant_id: str, resource_tenant_id: str) -> None:
    if request_tenant_id != resource_tenant_id:
        raise PermissionError("Tenant scope violation.")


def cleanup_tenant_scratch(tenant_id: str, event_id: str) -> None:
    scratch = Path(settings.scratch_root) / tenant_id / event_id
    if scratch.exists():
        shutil.rmtree(scratch)


def audit_action(tenant_id: str, event_id: str | None, action: str, payload: dict) -> None:
    AuditRepository.create(tenant_id=tenant_id, event_id=event_id, action=action, payload=payload)
