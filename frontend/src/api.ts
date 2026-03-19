import type { Event, EventSummary, MediaType, OutputType, Person, PersonReference, PlannerPlan, RenderJob, RenderJobListItem } from "./types";

export type ApiClientConfig = {
  baseUrl: string;
};

export class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = typeof payload?.detail === "string" ? payload.detail : "Request failed";
    throw new ApiError(detail, response.status, payload);
  }
  return payload as T;
}

export function createApiClient(config: ApiClientConfig) {
  const baseUrl = config.baseUrl.replace(/\/+$/, "");
  return {
    listEvents: (tenantId: string) =>
      request<{ events: Event[] }>(baseUrl, `/events?tenant_id=${encodeURIComponent(tenantId)}`),
    getEventSummary: (tenantId: string, eventId: string) =>
      request<EventSummary>(
        baseUrl,
        `/events/${encodeURIComponent(eventId)}/summary?tenant_id=${encodeURIComponent(tenantId)}`
      ),
    listEventRenders: (tenantId: string, eventId: string) =>
      request<{ event_id: string; renders: RenderJobListItem[] }>(
        baseUrl,
        `/events/${encodeURIComponent(eventId)}/renders?tenant_id=${encodeURIComponent(tenantId)}`
      ),
    createEvent: (payload: {
      tenant_id: string;
      title: string;
      event_type: string;
      venue?: string;
      date?: string;
    }) => request<Event>(baseUrl, "/events", { method: "POST", body: JSON.stringify(payload) }),
    ingestAsset: (payload: {
      tenant_id: string;
      event_id: string;
      media_path: string;
      media_type: MediaType;
    }) => request<{ asset_id: string; insights_generated: number }>(baseUrl, "/assets", { method: "POST", body: JSON.stringify(payload) }),
    /** File or folder path; image/video types auto-detected. */
    ingestFromPath: (payload: {
      tenant_id: string;
      event_id: string;
      path: string;
      recursive?: boolean;
    }) =>
      request<{
        batch: true;
        count: number;
        assets: Array<{ asset_id: string; media_path: string; media_type: string; insights_generated: number }>;
      }>(baseUrl, "/assets", {
        method: "POST",
        body: JSON.stringify({
          tenant_id: payload.tenant_id,
          event_id: payload.event_id,
          path: payload.path.trim(),
          recursive: payload.recursive !== false
        })
      }),
    getContext: (eventId: string, tenantId: string) =>
      request<{ event_id: string; context: Record<string, unknown[]> }>(
        baseUrl,
        `/events/${encodeURIComponent(eventId)}/context?tenant_id=${encodeURIComponent(tenantId)}`
      ),
    createPlan: (payload: {
      tenant_id: string;
      event_id: string;
      output_type: OutputType;
      prompt: string;
      target_duration_seconds: number;
      include_faces: string[];
      include_asset_ids: string[];
      excluded_asset_ids: string[];
      include_media_types: MediaType[];
      render_subtitles: boolean;
      render_overlays: boolean;
    }) => request<{ plan: PlannerPlan }>(baseUrl, "/requests/plan", { method: "POST", body: JSON.stringify(payload) }),
    render: (payload: {
      tenant_id: string;
      event_id: string;
      output_type: OutputType;
      prompt: string;
      target_duration_seconds: number;
      include_faces: string[];
      include_asset_ids: string[];
      excluded_asset_ids: string[];
      include_media_types: MediaType[];
      render_subtitles: boolean;
      render_overlays: boolean;
    }) =>
      request<{ plan: PlannerPlan; render_job: RenderJob }>(baseUrl, "/requests/render", {
        method: "POST",
        body: JSON.stringify(payload)
      }),
    regenerate: (payload: {
      tenant_id: string;
      event_id: string;
      output_type: OutputType;
      prompt: string;
      target_duration_seconds: number;
      include_asset_ids: string[];
      exclude_asset_ids: string[];
      include_media_types: MediaType[];
      render_subtitles: boolean;
      render_overlays: boolean;
    }) =>
      request<{ status: string; plan: PlannerPlan; render_job: RenderJob }>(baseUrl, "/requests/feedback/regenerate", {
        method: "POST",
        body: JSON.stringify(payload)
      }),
    createPerson: (payload: { tenant_id: string; event_id: string; display_name: string }) =>
      request<Person>(baseUrl, "/persons", { method: "POST", body: JSON.stringify(payload) }),
    listPersons: (tenantId: string, eventId: string) =>
      request<{ persons: Person[] }>(
        baseUrl,
        `/persons?tenant_id=${encodeURIComponent(tenantId)}&event_id=${encodeURIComponent(eventId)}`
      ),
    addPersonReference: (personId: string, payload: { tenant_id: string; event_id: string; image_path: string }) =>
      request<PersonReference>(baseUrl, `/persons/${encodeURIComponent(personId)}/references`, {
        method: "POST",
        body: JSON.stringify(payload)
      }),
    reindexFaces: (eventId: string, tenantId: string) =>
      request<{ status: string; asset_count: number }>(
        baseUrl,
        `/events/${encodeURIComponent(eventId)}/faces/reindex?tenant_id=${encodeURIComponent(tenantId)}`,
        { method: "POST" }
      ),
    listFaceMatches: (eventId: string, tenantId: string, personId?: string) => {
      const personParam = personId ? `&person_id=${encodeURIComponent(personId)}` : "";
      return request<{ event_id: string; matches: unknown[] }>(
        baseUrl,
        `/events/${encodeURIComponent(eventId)}/faces/matches?tenant_id=${encodeURIComponent(tenantId)}${personParam}`
      );
    },
    getRenderVideoUrl: (renderJobId: string, tenantId: string) =>
      `${baseUrl}/renders/${encodeURIComponent(renderJobId)}/video?tenant_id=${encodeURIComponent(tenantId)}`
  };
}

export function getDefaultApiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";
}
