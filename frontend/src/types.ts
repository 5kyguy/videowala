export type OutputType = "highlight_reel" | "chronological_film" | "person_focus_reel";
export type MediaType = "image" | "video";
/** Landscape ≈ 16:9 center crop; portrait ≈ 9:16 reels-style center crop (no resolution/fps targets). */
export type VideoOrientation = "landscape" | "portrait";

export type Event = {
  id: string;
  tenant_id: string;
  title: string;
  event_type: string;
  venue?: string | null;
  date?: string | null;
  predefined_tags?: string[];
  ocr_languages?: string[];
  created_at: string;
};

export type Person = {
  id: string;
  tenant_id: string;
  event_id: string;
  display_name: string;
  created_at: string;
};

export type PersonReference = {
  id: string;
  person_id: string;
  tenant_id: string;
  event_id: string;
  image_path: string;
  embedding: number[];
  created_at: string;
};

/** Metadata from GET /events/.../person-references (no embedding). */
export type PersonFaceReferenceListItem = {
  id: string;
  person_id: string;
  event_id: string;
  display_name: string;
  image_path: string;
  created_at: string;
};

export type PlannerAction = {
  action: string;
  params: Record<string, unknown>;
};

export type PlannerPlan = {
  tenant_id: string;
  event_id: string;
  output_type: OutputType;
  rationale: string;
  actions: PlannerAction[];
};

export type RenderJob = {
  id: string;
  tenant_id: string;
  event_id: string;
  plan_id: string;
  status: "queued" | "running" | "completed" | "failed";
  output_path: string | null;
  progress_percent?: number;
  error_message?: string | null;
  created_at: string;
  /** Present when the job was created from a content request with a prompt (stored in render spec). */
  planner_prompt?: string | null;
};

export type MediaExtensionCount = {
  extension: string;
  count: number;
};

export type EventSummaryStats = {
  assets_total: number;
  images_total: number;
  videos_total: number;
  has_media: boolean;
  persons_total: number;
  face_references_total: number;
  faces_saved: boolean;
  face_match_insights_total: number;
  has_face_matches: boolean;
  renders_total: number;
  renders_queued: number;
  renders_running: number;
  renders_completed: number;
  renders_failed: number;
  index_jobs_total: number;
  index_jobs_queued: number;
  index_jobs_running: number;
  index_jobs_completed: number;
  index_jobs_failed: number;
  /** Bytes on disk for resolved source files */
  media_storage_bytes: number;
  media_storage_files_found: number;
  media_storage_files_missing: number;
  media_bytes_images: number;
  media_bytes_videos: number;
  /** Bytes for completed render MP4 outputs */
  renders_storage_bytes: number;
  /** Sum of (finished − started) for terminal index jobs */
  index_duration_seconds_total: number;
  index_duration_job_count: number;
  media_extension_top: MediaExtensionCount[];
  /** Active queued/running index job: current model stage label from the backend */
  index_current_stage?: string | null;
  /** Overall pipeline progress 0–100 for the active index job */
  index_current_progress_percent?: number | null;
};

export type EventSummary = {
  event: Event;
  stats: EventSummaryStats;
};

export type RenderJobListItem = RenderJob;

/** Image culling row from GET /events/.../photos/curation (video assets excluded). */
export type PhotoCurationItem = {
  asset_id: string;
  segment_id: string;
  score: number;
  keep: boolean;
  is_duplicate: boolean;
  reject_reasons: string[];
};

export type PhotoCurationListResponse = {
  event_id: string;
  items: PhotoCurationItem[];
};
