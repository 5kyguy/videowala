from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from ..db import now_utc
from ..repositories import AssetRepository, IndexJobRepository, RenderRepository, next_id
from ..schemas import IndexJob
from ..services.indexing import index_asset
from ..services.rendering import execute_render_job

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="videowala-worker")
_LOCK = Lock()
_INFLIGHT_INDEX: dict[str, str] = {}
_INFLIGHT_RENDER: set[str] = set()


def run_index_job(asset_id: str) -> int:
    insights = index_asset(asset_id)
    return len(insights)


def submit_index_job(asset_id: str) -> IndexJob:
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
        )
        IndexJobRepository.create(job)
        _INFLIGHT_INDEX[asset_id] = job.id

        def _task() -> None:
            IndexJobRepository.mark_running(job.id)
            try:
                IndexJobRepository.set_progress(job.id, 30)
                count = run_index_job(asset.id)
                IndexJobRepository.mark_completed(job.id, count)
            except Exception as exc:  # noqa: BLE001
                IndexJobRepository.mark_failed(job.id, str(exc))
            finally:
                with _LOCK:
                    if _INFLIGHT_INDEX.get(asset.id) == job.id:
                        _INFLIGHT_INDEX.pop(asset.id, None)

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

    _EXECUTOR.submit(_task)
