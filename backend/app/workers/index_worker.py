from __future__ import annotations

from ..services.indexing import index_asset


def run_index_job(asset_id: str) -> int:
    insights = index_asset(asset_id)
    return len(insights)
