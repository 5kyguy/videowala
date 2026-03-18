from __future__ import annotations

import pytest

from app.services.privacy import assert_tenant_scope


def test_tenant_scope_allows_matching_tenant() -> None:
    assert_tenant_scope("tenant_a", "tenant_a")


def test_tenant_scope_rejects_cross_tenant_access() -> None:
    with pytest.raises(PermissionError):
        assert_tenant_scope("tenant_a", "tenant_b")
