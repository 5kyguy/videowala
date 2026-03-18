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
});
