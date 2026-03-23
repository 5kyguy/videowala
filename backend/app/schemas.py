from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class OutputType(str, Enum):
    highlight_reel = "highlight_reel"
    chronological_film = "chronological_film"
    person_focus_reel = "person_focus_reel"


VideoOrientation = Literal["landscape", "portrait"]


class InsightType(str, Enum):
    vlm_caption = "vlm_caption"
    vlm_tags = "vlm_tags"
    face_detections = "face_detections"
    face_matches = "face_matches"
    # Stage 2 (deferred) hooks
    ocr_text = "ocr_text"
    asr_transcript = "asr_transcript"
    semantic_embedding = "semantic_embedding"
    cull_metrics = "cull_metrics"


class EventCreate(BaseModel):
    tenant_id: str
    title: str
    event_type: str
    venue: str | None = None
    date: str | None = None
    predefined_tags: list[str] = Field(default_factory=list)
    ocr_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="BCP-47 style hints for PaddleOCR (e.g. en, hi, gu).",
    )


class Event(BaseModel):
    id: str
    tenant_id: str
    title: str
    event_type: str
    venue: str | None = None
    date: str | None = None
    predefined_tags: list[str] = Field(default_factory=list)
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])
    created_at: datetime


class EventUpdate(BaseModel):
    """Partial update for event metadata used by indexing (PoC)."""

    title: str | None = None
    event_type: str | None = None
    venue: str | None = None
    date: str | None = None
    predefined_tags: list[str] | None = None
    ocr_languages: list[str] | None = None


class AssetRegister(BaseModel):
    tenant_id: str
    event_id: str
    media_path: str
    media_type: Literal["image", "video"]
    captured_at: str | None = None


class AssetIngestBody(BaseModel):
    """Single file: `media_path` + `media_type`. Batch: `path` (file or folder) with auto-detected types."""

    tenant_id: str
    event_id: str
    media_path: str | None = None
    media_type: Literal["image", "video"] | None = None
    path: str | None = Field(
        default=None,
        description="Path to one media file or a folder; image/video extensions are auto-detected.",
    )
    recursive: bool = Field(default=True, description="If path is a folder, scan subfolders when true.")

    @model_validator(mode="after")
    def one_mode(self) -> AssetIngestBody:
        p = (self.path or "").strip()
        if p:
            self.path = p
            return self
        if self.media_path and self.media_type:
            self.path = None
            return self
        raise ValueError(
            "Provide 'path' (file or folder for batch ingest) or both 'media_path' and 'media_type' (single file)."
        )


class Asset(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    media_path: str
    media_type: Literal["image", "video"]
    created_at: datetime


class AssetProxy(BaseModel):
    asset_id: str
    proxy_path: str
    metadata: dict = Field(default_factory=dict)
    manifest: dict = Field(default_factory=dict)
    created_at: datetime


class AssetSegment(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    asset_id: str
    start_s: float
    end_s: float
    score: float = 0.0
    keep: bool = True
    is_duplicate: bool = False
    reject_reasons: list[str] = Field(default_factory=list)
    created_at: datetime


class AssetInsight(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    asset_id: str
    insight_type: InsightType
    payload: dict
    confidence: float = 0.0
    created_at: datetime


class ContentRequestCreate(BaseModel):
    tenant_id: str
    event_id: str
    output_type: OutputType
    prompt: str = Field(min_length=5)
    target_duration_seconds: int = Field(default=60, ge=5, le=3600)
    include_faces: list[str] = Field(default_factory=list)
    include_asset_ids: list[str] = Field(default_factory=list)
    excluded_asset_ids: list[str] = Field(default_factory=list)
    include_media_types: list[Literal["image", "video"]] = Field(
        default_factory=list,
        description="Planner/render pool: video only. Empty means ['video'].",
    )
    video_orientation: VideoOrientation = Field(
        default="landscape",
        description="Output framing: landscape (16:9 crop) or portrait / reels (9:16 crop). No resolution or fps targets.",
    )

    @model_validator(mode="after")
    def video_only_planner_pool(self) -> Self:
        """Video render/plan track uses video assets only; images are curated elsewhere."""
        allowed: set[str] = set()
        for m in self.include_media_types:
            if m in ("image", "video"):
                allowed.add(m)
        if not allowed:
            object.__setattr__(self, "include_media_types", ["video"])
            return self
        if "video" not in allowed:
            raise ValueError("Planning and render use video only; include_media_types cannot be image-only.")
        object.__setattr__(self, "include_media_types", ["video"])
        return self


class PlannerAction(BaseModel):
    action: Literal[
        "select_segments",
        "set_order",
        "set_duration",
        "render_preview",
        "exclude_segments",
    ]
    params: dict = Field(default_factory=dict)


class PlannerPlan(BaseModel):
    tenant_id: str
    event_id: str
    output_type: OutputType
    rationale: str
    actions: list[PlannerAction]


class RenderJobCreate(BaseModel):
    tenant_id: str
    event_id: str
    plan_id: str
    preview: bool = True


class RenderJob(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    plan_id: str
    status: Literal["queued", "running", "completed", "failed"]
    output_path: str | None = None
    progress_percent: int = 0
    error_message: str | None = None
    created_at: datetime
    planner_prompt: str | None = Field(
        default=None,
        description="User prompt from the content request used to build the plan for this render.",
    )


class IndexJob(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    asset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress_percent: int = 0
    insights_generated: int = 0
    error_message: str | None = None
    created_at: datetime


class FeedbackUpdate(BaseModel):
    tenant_id: str
    event_id: str
    output_type: OutputType = OutputType.highlight_reel
    prompt: str = Field(default="Regenerate with updated include/exclude constraints.", min_length=5)
    target_duration_seconds: int = Field(default=60, ge=5, le=3600)
    include_asset_ids: list[str] = Field(default_factory=list)
    exclude_asset_ids: list[str] = Field(default_factory=list)
    include_media_types: list[Literal["image", "video"]] = Field(
        default_factory=list,
        description="Regenerate uses the video-only render pool; empty means ['video'].",
    )
    video_orientation: VideoOrientation = Field(
        default="landscape",
        description="Output framing: landscape (16:9 crop) or portrait / reels (9:16 crop).",
    )

    @model_validator(mode="after")
    def video_only_regenerate_pool(self) -> Self:
        allowed: set[str] = set()
        for m in self.include_media_types:
            if m in ("image", "video"):
                allowed.add(m)
        if not allowed:
            object.__setattr__(self, "include_media_types", ["video"])
            return self
        if "video" not in allowed:
            raise ValueError("Regenerate uses video only; include_media_types cannot be image-only.")
        object.__setattr__(self, "include_media_types", ["video"])
        return self


class PhotoCurationItem(BaseModel):
    asset_id: str
    segment_id: str
    score: float
    keep: bool
    is_duplicate: bool
    reject_reasons: list[str] = Field(default_factory=list)


class PhotoCurationListResponse(BaseModel):
    event_id: str
    items: list[PhotoCurationItem]


class EventSummaryStats(BaseModel):
    assets_total: int
    images_total: int
    videos_total: int
    has_media: bool
    persons_total: int
    face_references_total: int
    faces_saved: bool
    face_match_insights_total: int
    has_face_matches: bool
    renders_total: int
    renders_queued: int
    renders_running: int
    renders_completed: int
    renders_failed: int
    index_jobs_total: int
    index_jobs_queued: int
    index_jobs_running: int
    index_jobs_completed: int
    index_jobs_failed: int


class EventSummary(BaseModel):
    event: Event
    stats: EventSummaryStats


class RenderJobList(BaseModel):
    event_id: str
    renders: list[RenderJob]


class PersonCreate(BaseModel):
    tenant_id: str
    event_id: str
    display_name: str = Field(min_length=1)


class Person(BaseModel):
    id: str
    tenant_id: str
    event_id: str
    display_name: str
    created_at: datetime


class PersonReferenceCreate(BaseModel):
    tenant_id: str
    event_id: str
    image_path: str


class PersonReference(BaseModel):
    id: str
    person_id: str
    tenant_id: str
    event_id: str
    image_path: str
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime
