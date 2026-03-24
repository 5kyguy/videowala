import { describe, expect, it, vi } from "vitest";

import { ApiError, createApiClient } from "../api";

describe("api client", () => {
  it("calls listEvents with tenant query", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      text: async () => JSON.stringify({ events: [] })
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient({ baseUrl: "http://localhost:8000/" });
    await client.listEvents("tenant_a");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/events?tenant_id=tenant_a",
      expect.any(Object)
    );
  });

  it("throws ApiError for non-2xx", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 403,
      text: async () => JSON.stringify({ detail: "Tenant scope violation." })
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient({ baseUrl: "http://localhost:8000" });
    await expect(client.listEvents("tenant_a")).rejects.toBeInstanceOf(ApiError);
  });

  it("calls event summary endpoint with tenant scope", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      text: async () =>
        JSON.stringify({
          event: {
            id: "event_1",
            tenant_id: "tenant_a",
            title: "Demo",
            event_type: "wedding",
            created_at: new Date().toISOString()
          },
          stats: {
            assets_total: 0,
            images_total: 0,
            videos_total: 0,
            has_media: false,
            persons_total: 0,
            face_references_total: 0,
            faces_saved: false,
            face_match_insights_total: 0,
            has_face_matches: false,
            renders_total: 0,
            renders_queued: 0,
            renders_running: 0,
            renders_completed: 0,
            renders_failed: 0,
            index_jobs_total: 0,
            index_jobs_queued: 0,
            index_jobs_running: 0,
            index_jobs_completed: 0,
            index_jobs_failed: 0,
            media_storage_bytes: 0,
            media_storage_files_found: 0,
            media_storage_files_missing: 0,
            media_bytes_images: 0,
            media_bytes_videos: 0,
            renders_storage_bytes: 0,
            index_duration_seconds_total: 0,
            index_duration_job_count: 0,
            media_extension_top: []
          }
        })
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient({ baseUrl: "http://localhost:8000" });
    await client.getEventSummary("tenant_a", "event_1");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/events/event_1/summary?tenant_id=tenant_a",
      expect.any(Object)
    );
  });

  it("builds photo curation and export URLs", () => {
    const client = createApiClient({ baseUrl: "http://localhost:8000" });
    expect(client.exportKeptPhotosUrl("event_x", "tenant_a")).toBe(
      "http://localhost:8000/events/event_x/photos/export-kept?tenant_id=tenant_a"
    );
    expect(client.getAssetMediaUrl("event_x", "asset_y", "tenant_a")).toBe(
      "http://localhost:8000/events/event_x/assets/asset_y/media?tenant_id=tenant_a"
    );
    expect(client.getAssetMediaUrl("event_x", "asset_y", "tenant_a", { maxEdge: 1280 })).toBe(
      "http://localhost:8000/events/event_x/assets/asset_y/media?tenant_id=tenant_a&max_edge=1280"
    );
  });

  it("calls delete endpoints with tenant scope", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      text: async () => JSON.stringify({ status: "deleted" })
    }));
    vi.stubGlobal("fetch", fetchMock);
    const client = createApiClient({ baseUrl: "http://localhost:8000" });
    await client.deleteEvent("event_1", "tenant_a");
    await client.deleteRenderJob("render_1", "tenant_a");
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/events/event_1?tenant_id=tenant_a",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/renders/render_1?tenant_id=tenant_a",
      expect.objectContaining({ method: "DELETE" })
    );
  });
});
