export type OutputType = "highlight_reel" | "chronological_film" | "person_focus_reel";
export type MediaType = "image" | "video";

export type Event = {
  id: string;
  tenant_id: string;
  title: string;
  event_type: string;
  venue?: string | null;
  date?: string | null;
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
  created_at: string;
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
};

export type EventSummary = {
  event: Event;
  stats: EventSummaryStats;
};

export type RenderJobListItem = RenderJob;
