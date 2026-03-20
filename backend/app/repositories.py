from __future__ import annotations

from uuid import uuid4

from .db import db_cursor, decode_json, encode_json, from_iso, now_utc, to_iso
from .schemas import (
    Asset,
    AssetProxy,
    AssetSegment,
    AssetInsight,
    Event,
    IndexJob,
    Person,
    PersonReference,
    PlannerAction,
    PlannerPlan,
    RenderJob,
)


def next_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class EventRepository:
    @staticmethod
    def create(event: Event) -> Event:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (id, tenant_id, title, event_type, venue, date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.tenant_id,
                    event.title,
                    event.event_type,
                    event.venue,
                    event.date,
                    to_iso(event.created_at),
                ),
            )
        return event

    @staticmethod
    def get(event_id: str) -> Event | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return Event(
            id=row["id"],
            tenant_id=row["tenant_id"],
            title=row["title"],
            event_type=row["event_type"],
            venue=row["venue"],
            date=row["date"],
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def list_for_tenant(tenant_id: str) -> list[Event]:
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM events WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
        return [
            Event(
                id=row["id"],
                tenant_id=row["tenant_id"],
                title=row["title"],
                event_type=row["event_type"],
                venue=row["venue"],
                date=row["date"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]


class AssetRepository:
    @staticmethod
    def create(asset: Asset) -> Asset:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO assets (id, tenant_id, event_id, media_path, media_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.id,
                    asset.tenant_id,
                    asset.event_id,
                    asset.media_path,
                    asset.media_type,
                    to_iso(asset.created_at),
                ),
            )
        return asset

    @staticmethod
    def get(asset_id: str) -> Asset | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            return None
        return Asset(
            id=row["id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            media_path=row["media_path"],
            media_type=row["media_type"],
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def list_for_event(event_id: str) -> list[Asset]:
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM assets WHERE event_id = ? ORDER BY created_at ASC",
                (event_id,),
            ).fetchall()
        return [
            Asset(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                media_path=row["media_path"],
                media_type=row["media_type"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]


class AssetProxyRepository:
    @staticmethod
    def upsert(proxy: AssetProxy) -> AssetProxy:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset_proxies (asset_id, proxy_path, metadata_json, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    proxy_path = excluded.proxy_path,
                    metadata_json = excluded.metadata_json,
                    manifest_json = excluded.manifest_json,
                    created_at = excluded.created_at
                """,
                (
                    proxy.asset_id,
                    proxy.proxy_path,
                    encode_json(proxy.metadata),
                    encode_json(proxy.manifest),
                    to_iso(proxy.created_at),
                ),
            )
        return proxy

    @staticmethod
    def get(asset_id: str) -> AssetProxy | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM asset_proxies WHERE asset_id = ?", (asset_id,)).fetchone()
        if row is None:
            return None
        return AssetProxy(
            asset_id=row["asset_id"],
            proxy_path=row["proxy_path"],
            metadata=decode_json(row["metadata_json"]),
            manifest=decode_json(row["manifest_json"]),
            created_at=from_iso(row["created_at"]),
        )


class InsightRepository:
    @staticmethod
    def create_many(insights: list[AssetInsight]) -> None:
        if not insights:
            return
        with db_cursor() as cur:
            for insight in insights:
                cur.execute(
                    """
                    INSERT INTO asset_insights
                    (id, tenant_id, event_id, asset_id, insight_type, payload_json, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        insight.id,
                        insight.tenant_id,
                        insight.event_id,
                        insight.asset_id,
                        insight.insight_type.value,
                        encode_json(insight.payload),
                        insight.confidence,
                        to_iso(insight.created_at),
                    ),
                )

    @staticmethod
    def list_for_event(event_id: str, insight_type: str | None = None) -> list[AssetInsight]:
        with db_cursor() as cur:
            if insight_type is None:
                rows = cur.execute(
                    "SELECT * FROM asset_insights WHERE event_id = ? ORDER BY created_at ASC",
                    (event_id,),
                ).fetchall()
            else:
                rows = cur.execute(
                    """
                    SELECT * FROM asset_insights
                    WHERE event_id = ? AND insight_type = ?
                    ORDER BY created_at ASC
                    """,
                    (event_id, insight_type),
                ).fetchall()
        return [
            AssetInsight(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                asset_id=row["asset_id"],
                insight_type=row["insight_type"],
                payload=decode_json(row["payload_json"]),
                confidence=row["confidence"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def delete_for_asset(event_id: str, asset_id: str, insight_types: list[str]) -> None:
        if not insight_types:
            return
        placeholders = ",".join(["?"] * len(insight_types))
        with db_cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM asset_insights
                WHERE event_id = ? AND asset_id = ? AND insight_type IN ({placeholders})
                """,
                (event_id, asset_id, *insight_types),
            )


class SegmentRepository:
    @staticmethod
    def replace_for_asset(asset_id: str, event_id: str, segments: list[AssetSegment]) -> None:
        with db_cursor() as cur:
            cur.execute("DELETE FROM asset_segments WHERE event_id = ? AND asset_id = ?", (event_id, asset_id))
            for segment in segments:
                cur.execute(
                    """
                    INSERT INTO asset_segments
                    (id, tenant_id, event_id, asset_id, start_s, end_s, score, keep, is_duplicate, reject_reasons_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment.id,
                        segment.tenant_id,
                        segment.event_id,
                        segment.asset_id,
                        segment.start_s,
                        segment.end_s,
                        segment.score,
                        1 if segment.keep else 0,
                        1 if segment.is_duplicate else 0,
                        encode_json(segment.reject_reasons),
                        to_iso(segment.created_at),
                    ),
                )

    @staticmethod
    def list_for_event(event_id: str, *, keep_only: bool = False) -> list[AssetSegment]:
        sql = "SELECT * FROM asset_segments WHERE event_id = ?"
        params: tuple[object, ...] = (event_id,)
        if keep_only:
            sql += " AND keep = 1"
        sql += " ORDER BY score DESC, created_at ASC"
        with db_cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [
            AssetSegment(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                asset_id=row["asset_id"],
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                score=float(row["score"]),
                keep=bool(row["keep"]),
                is_duplicate=bool(row["is_duplicate"]),
                reject_reasons=decode_json(row["reject_reasons_json"]),
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def get(segment_id: str) -> AssetSegment | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM asset_segments WHERE id = ?", (segment_id,)).fetchone()
        if row is None:
            return None
        return AssetSegment(
            id=row["id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            asset_id=row["asset_id"],
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            score=float(row["score"]),
            keep=bool(row["keep"]),
            is_duplicate=bool(row["is_duplicate"]),
            reject_reasons=decode_json(row["reject_reasons_json"]),
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def update_culling(
        segment_id: str,
        *,
        score: float,
        keep: bool,
        is_duplicate: bool,
        reject_reasons: list[str],
    ) -> None:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE asset_segments
                SET score = ?, keep = ?, is_duplicate = ?, reject_reasons_json = ?
                WHERE id = ?
                """,
                (score, 1 if keep else 0, 1 if is_duplicate else 0, encode_json(reject_reasons), segment_id),
            )

    @staticmethod
    def batch_update_culling(
        updates: list[tuple[str, float, bool, bool, list[str]]],
    ) -> None:
        """Batch version of update_culling. Each tuple: (segment_id, score, keep, is_duplicate, reject_reasons)."""
        if not updates:
            return
        with db_cursor() as cur:
            for segment_id, score, keep, is_duplicate, reject_reasons in updates:
                cur.execute(
                    """
                    UPDATE asset_segments
                    SET score = ?, keep = ?, is_duplicate = ?, reject_reasons_json = ?
                    WHERE id = ?
                    """,
                    (score, 1 if keep else 0, 1 if is_duplicate else 0, encode_json(reject_reasons), segment_id),
                )


class PlanRepository:
    @staticmethod
    def create(plan: PlannerPlan) -> str:
        plan_id = next_id("plan")
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO planner_plans (id, tenant_id, event_id, output_type, rationale, actions_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    plan.tenant_id,
                    plan.event_id,
                    plan.output_type.value,
                    plan.rationale,
                    encode_json([a.model_dump() for a in plan.actions]),
                    to_iso(now_utc()),
                ),
            )
        return plan_id

    @staticmethod
    def get(plan_id: str) -> PlannerPlan | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM planner_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            return None
        actions = [PlannerAction(**item) for item in decode_json(row["actions_json"])]
        return PlannerPlan(
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            output_type=row["output_type"],
            rationale=row["rationale"],
            actions=actions,
        )


class RenderRepository:
    @staticmethod
    def create(
        job: RenderJob,
        *,
        input_files: list[str],
        duration_seconds: int,
        scratch_dir: str,
        subtitles_enabled: bool = False,
        overlays_enabled: bool = False,
        clip_inputs: list[dict] | None = None,
    ) -> RenderJob:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO render_jobs (id, tenant_id, event_id, plan_id, status, output_path, progress_percent, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.event_id,
                    job.plan_id,
                    job.status,
                    job.output_path,
                    job.progress_percent,
                    job.error_message,
                    to_iso(job.created_at),
                ),
            )
            cur.execute(
                """
                INSERT INTO render_specs (render_job_id, input_files_json, duration_seconds, scratch_dir, subtitles_enabled, overlays_enabled, clip_inputs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    encode_json(input_files),
                    duration_seconds,
                    scratch_dir,
                    1 if subtitles_enabled else 0,
                    1 if overlays_enabled else 0,
                    encode_json(clip_inputs or []),
                ),
            )
        return job

    @staticmethod
    def get(job_id: str) -> RenderJob | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM render_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return RenderJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            plan_id=row["plan_id"],
            status=row["status"],
            output_path=row["output_path"],
            progress_percent=int(row["progress_percent"]) if "progress_percent" in row.keys() else 0,
            error_message=row["error_message"] if "error_message" in row.keys() else None,
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def list_for_event(tenant_id: str, event_id: str) -> list[RenderJob]:
        with db_cursor() as cur:
            rows = cur.execute(
                """
                SELECT * FROM render_jobs
                WHERE tenant_id = ? AND event_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id, event_id),
            ).fetchall()
        return [
            RenderJob(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                plan_id=row["plan_id"],
                status=row["status"],
                output_path=row["output_path"],
                progress_percent=int(row["progress_percent"]) if "progress_percent" in row.keys() else 0,
                error_message=row["error_message"] if "error_message" in row.keys() else None,
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def get_spec(job_id: str) -> dict | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM render_specs WHERE render_job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "input_files": decode_json(row["input_files_json"]),
            "duration_seconds": row["duration_seconds"],
            "scratch_dir": row["scratch_dir"],
            "subtitles_enabled": bool(row["subtitles_enabled"]) if "subtitles_enabled" in row.keys() else False,
            "overlays_enabled": bool(row["overlays_enabled"]) if "overlays_enabled" in row.keys() else False,
            "clip_inputs": decode_json(row["clip_inputs_json"]) if "clip_inputs_json" in row.keys() and row["clip_inputs_json"] else [],
        }

    @staticmethod
    def update_status(
        job_id: str,
        status: str,
        output_path: str | None = None,
        *,
        progress_percent: int | None = None,
        error_message: str | None = None,
    ) -> None:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE render_jobs
                SET status = ?,
                    output_path = COALESCE(?, output_path),
                    progress_percent = COALESCE(?, progress_percent),
                    error_message = COALESCE(?, error_message)
                WHERE id = ?
                """,
                (status, output_path, progress_percent, error_message, job_id),
            )


class IndexJobRepository:
    @staticmethod
    def create(job: IndexJob) -> IndexJob:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO index_jobs
                (id, tenant_id, event_id, asset_id, status, progress_percent, insights_generated, error_message, created_at, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.event_id,
                    job.asset_id,
                    job.status,
                    job.progress_percent,
                    job.insights_generated,
                    job.error_message,
                    to_iso(job.created_at),
                ),
            )
        return job

    @staticmethod
    def get(job_id: str) -> IndexJob | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM index_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return IndexJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            asset_id=row["asset_id"],
            status=row["status"],
            progress_percent=int(row["progress_percent"]),
            insights_generated=int(row["insights_generated"]),
            error_message=row["error_message"],
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def list_for_event(event_id: str) -> list[IndexJob]:
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM index_jobs WHERE event_id = ? ORDER BY created_at DESC",
                (event_id,),
            ).fetchall()
        return [
            IndexJob(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                asset_id=row["asset_id"],
                status=row["status"],
                progress_percent=int(row["progress_percent"]),
                insights_generated=int(row["insights_generated"]),
                error_message=row["error_message"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def count_by_status_for_event(event_id: str) -> dict[str, int]:
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT status, COUNT(*) AS c FROM index_jobs WHERE event_id = ? GROUP BY status",
                (event_id,),
            ).fetchall()
        for row in rows:
            counts[row["status"]] = int(row["c"])
        return counts

    @staticmethod
    def mark_running(job_id: str) -> None:
        with db_cursor() as cur:
            cur.execute(
                "UPDATE index_jobs SET status = 'running', started_at = ?, progress_percent = 5 WHERE id = ?",
                (to_iso(now_utc()), job_id),
            )

    @staticmethod
    def set_progress(job_id: str, progress_percent: int) -> None:
        with db_cursor() as cur:
            cur.execute("UPDATE index_jobs SET progress_percent = ? WHERE id = ?", (progress_percent, job_id))

    @staticmethod
    def mark_completed(job_id: str, insights_generated: int) -> None:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE index_jobs
                SET status = 'completed', progress_percent = 100, insights_generated = ?, finished_at = ?
                WHERE id = ?
                """,
                (insights_generated, to_iso(now_utc()), job_id),
            )

    @staticmethod
    def mark_failed(job_id: str, error_message: str) -> None:
        with db_cursor() as cur:
            cur.execute(
                """
                UPDATE index_jobs
                SET status = 'failed', error_message = ?, finished_at = ?
                WHERE id = ?
                """,
                (error_message, to_iso(now_utc()), job_id),
            )


class AuditRepository:
    @staticmethod
    def create(tenant_id: str, event_id: str | None, action: str, payload: dict) -> None:
        audit_id = next_id("audit")
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (id, tenant_id, event_id, action, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (audit_id, tenant_id, event_id, action, encode_json(payload), to_iso(now_utc())),
            )


class PersonRepository:
    @staticmethod
    def create(person: Person) -> Person:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (id, tenant_id, event_id, display_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person.id, person.tenant_id, person.event_id, person.display_name, to_iso(person.created_at)),
            )
        return person

    @staticmethod
    def get(person_id: str) -> Person | None:
        with db_cursor() as cur:
            row = cur.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            return None
        return Person(
            id=row["id"],
            tenant_id=row["tenant_id"],
            event_id=row["event_id"],
            display_name=row["display_name"],
            created_at=from_iso(row["created_at"]),
        )

    @staticmethod
    def list_for_event(tenant_id: str, event_id: str) -> list[Person]:
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM persons WHERE tenant_id = ? AND event_id = ? ORDER BY created_at DESC",
                (tenant_id, event_id),
            ).fetchall()
        return [
            Person(
                id=row["id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                display_name=row["display_name"],
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]


class PersonReferenceRepository:
    @staticmethod
    def create(reference: PersonReference) -> PersonReference:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO person_references (id, person_id, tenant_id, event_id, image_path, embedding_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reference.id,
                    reference.person_id,
                    reference.tenant_id,
                    reference.event_id,
                    reference.image_path,
                    encode_json(reference.embedding),
                    to_iso(reference.created_at),
                ),
            )
        return reference

    @staticmethod
    def list_for_event(tenant_id: str, event_id: str) -> list[PersonReference]:
        with db_cursor() as cur:
            rows = cur.execute(
                """
                SELECT * FROM person_references
                WHERE tenant_id = ? AND event_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id, event_id),
            ).fetchall()
        return [
            PersonReference(
                id=row["id"],
                person_id=row["person_id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                image_path=row["image_path"],
                embedding=decode_json(row["embedding_json"]),
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def list_for_person(person_id: str) -> list[PersonReference]:
        with db_cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM person_references WHERE person_id = ? ORDER BY created_at DESC",
                (person_id,),
            ).fetchall()
        return [
            PersonReference(
                id=row["id"],
                person_id=row["person_id"],
                tenant_id=row["tenant_id"],
                event_id=row["event_id"],
                image_path=row["image_path"],
                embedding=decode_json(row["embedding_json"]),
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]
