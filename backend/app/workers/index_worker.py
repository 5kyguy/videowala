from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..config import settings
from ..db import now_utc
from ..repositories import AssetRepository, IndexJobRepository, RenderRepository, next_id
from ..schemas import IndexJob
from ..services.indexing import index_event_by_model_stages
from ..services.rendering import execute_render_job

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=settings.index_workers, thread_name_prefix="videowala-worker")
# Renders use a separate pool so a render is never queued behind long-running index jobs (same global pool starved the UI).
_RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="videowala-render")
_LOCK = Lock()
_INFLIGHT_INDEX: dict[str, str] = {}
_INFLIGHT_RENDER: set[str] = set()
_LAST_INDEX_LOG_MILESTONE: dict[str, int] = {}


def _emit_indexing_progress_log(event_id: str) -> None:
    """Aggregate indexing progress (25% milestones), not per-asset lines."""
    counts = IndexJobRepository.count_by_status_for_event(event_id)
    total = sum(counts.values())
    if total == 0:
        return
    done = counts["completed"] + counts["failed"]
    pct = (done * 100) // total
    milestone = max((m for m in (25, 50, 75, 100) if pct >= m), default=0)
    prev = _LAST_INDEX_LOG_MILESTONE.get(event_id, 0)
    if milestone <= prev:
        return
    _LAST_INDEX_LOG_MILESTONE[event_id] = milestone
    if milestone == 100:
        _LAST_INDEX_LOG_MILESTONE.pop(event_id, None)
    logger.info(
        "Indexing progress event_id=%s: %d/%d jobs finished (%d queued, %d running, %d failed)",
        event_id,
        done,
        total,
        counts["queued"],
        counts["running"],
        counts["failed"],
    )


def run_index_job(job_id: str) -> int:
    job = IndexJobRepository.get(job_id)
    if job is None:
        raise KeyError(f"Index job not found: {job_id}")
    asset_ids = list(job.staged_asset_ids) if job.staged_asset_ids else [job.asset_id]
    insights = index_event_by_model_stages(asset_ids, semantic_prompt=job.semantic_prompt)
    return len(insights)


def submit_index_job(asset_id: str, semantic_prompt: str | None = None) -> IndexJob:
    asset = AssetRepository.get(asset_id)
    if asset is None:
        raise KeyError(f"Asset not found: {asset_id}")
    with _LOCK:
        existing_id = _INFLIGHT_INDEX.get(asset_id)
        if existing_id:
            existing = IndexJobRepository.get(existing_id)
            if existing is not None and existing.status in {"queued", "running"}:
                return existing

        job = IndexJob(
            id=next_id("idxjob"),
            tenant_id=asset.tenant_id,
            event_id=asset.event_id,
            asset_id=asset.id,
            status="queued",
            progress_percent=0,
            insights_generated=0,
            error_message=None,
            created_at=now_utc(),
            semantic_prompt=semantic_prompt,
            staged_asset_ids=None,
        )
        IndexJobRepository.create(job)
        _INFLIGHT_INDEX[asset_id] = job.id

        def _task() -> None:
            IndexJobRepository.mark_running(job.id)
            try:
                IndexJobRepository.set_progress(job.id, 30)
                count = run_index_job(job.id)
                IndexJobRepository.mark_completed(job.id, count)
                _emit_indexing_progress_log(asset.event_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Index job failed job_id=%s asset_id=%s event_id=%s: %s",
                    job.id,
                    asset.id,
                    asset.event_id,
                    exc,
                )
                IndexJobRepository.mark_failed(job.id, str(exc))
                _emit_indexing_progress_log(asset.event_id)
            finally:
                with _LOCK:
                    if _INFLIGHT_INDEX.get(asset.id) == job.id:
                        _INFLIGHT_INDEX.pop(asset.id, None)

        _EXECUTOR.submit(_task)
        return job


def submit_staged_index_job(
    tenant_id: str,
    event_id: str,
    staged_asset_ids: list[str],
    semantic_prompt: str | None = None,
) -> IndexJob:
    """One index job that runs the GPU pipeline model-by-model across all listed assets."""
    if not staged_asset_ids:
        raise ValueError("staged_asset_ids must be non-empty")
    first = AssetRepository.get(staged_asset_ids[0])
    if first is None:
        raise KeyError(f"Asset not found: {staged_asset_ids[0]}")
    if first.tenant_id != tenant_id or first.event_id != event_id:
        raise ValueError("Staged assets must match tenant_id and event_id")
    for aid in staged_asset_ids[1:]:
        a = AssetRepository.get(aid)
        if a is None:
            raise KeyError(f"Asset not found: {aid}")
        if a.tenant_id != tenant_id or a.event_id != event_id:
            raise ValueError("All staged assets must belong to the same event and tenant")

    with _LOCK:
        for aid in staged_asset_ids:
            existing_id = _INFLIGHT_INDEX.get(aid)
            if existing_id:
                existing = IndexJobRepository.get(existing_id)
                if existing is not None and existing.status in {"queued", "running"}:
                    return existing

        job = IndexJob(
            id=next_id("idxjob"),
            tenant_id=tenant_id,
            event_id=event_id,
            asset_id=staged_asset_ids[0],
            status="queued",
            progress_percent=0,
            insights_generated=0,
            error_message=None,
            created_at=now_utc(),
            semantic_prompt=semantic_prompt,
            staged_asset_ids=staged_asset_ids,
        )
        IndexJobRepository.create(job)
        for aid in staged_asset_ids:
            _INFLIGHT_INDEX[aid] = job.id

        def _task() -> None:
            IndexJobRepository.mark_running(job.id)
            try:
                IndexJobRepository.set_progress(job.id, 30)
                count = run_index_job(job.id)
                IndexJobRepository.mark_completed(job.id, count)
                _emit_indexing_progress_log(event_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Index job failed job_id=%s event_id=%s staged=%s: %s",
                    job.id,
                    event_id,
                    staged_asset_ids,
                    exc,
                )
                IndexJobRepository.mark_failed(job.id, str(exc))
                _emit_indexing_progress_log(event_id)
            finally:
                with _LOCK:
                    for aid in staged_asset_ids:
                        if _INFLIGHT_INDEX.get(aid) == job.id:
                            _INFLIGHT_INDEX.pop(aid, None)

        _EXECUTOR.submit(_task)
        return job


def submit_render_job(job_id: str) -> None:
    with _LOCK:
        if job_id in _INFLIGHT_RENDER:
            return
        _INFLIGHT_RENDER.add(job_id)

    def _task() -> None:
        try:
            RenderRepository.update_status(job_id, "running", progress_percent=5)
            execute_render_job(job_id)
        except Exception as exc:  # noqa: BLE001
            RenderRepository.update_status(job_id, "failed", progress_percent=100, error_message=str(exc))
        finally:
            with _LOCK:
                _INFLIGHT_RENDER.discard(job_id)

    _RENDER_EXECUTOR.submit(_task)
