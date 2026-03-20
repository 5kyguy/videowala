import { FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError, createApiClient, getDefaultApiBaseUrl } from "./api";
import type { Event, EventSummary, OutputType, RenderJobListItem } from "./types";

const PROFILE_STORAGE_KEY = "videowala_profiles";
const DEFAULT_OUTPUT_TYPE: OutputType = "highlight_reel";

function asErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return `HTTP ${error.status}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return "Unknown error";
}

function loadProfiles(): string[] {
  const fallback = ["tenant_a"];
  const storage = globalThis.localStorage;
  const raw = typeof storage?.getItem === "function" ? storage.getItem(PROFILE_STORAGE_KEY) : null;
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as string[];
    const cleaned = parsed.map((item) => item.trim()).filter(Boolean);
    return cleaned.length > 0 ? cleaned : fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(getDefaultApiBaseUrl());
  const api = useMemo(() => createApiClient({ baseUrl: apiBaseUrl }), [apiBaseUrl]);

  const [profiles, setProfiles] = useState<string[]>(() => loadProfiles());
  const [newProfile, setNewProfile] = useState("");
  const [tenantId, setTenantId] = useState(() => loadProfiles()[0]);

  const [events, setEvents] = useState<Event[]>([]);
  const [selectedEventId, setSelectedEventId] = useState("");
  const [summary, setSummary] = useState<EventSummary | null>(null);
  const [renders, setRenders] = useState<RenderJobListItem[]>([]);

  const [eventTitle, setEventTitle] = useState("Demo Event");
  const [eventType, setEventType] = useState("wedding");
  const [eventVenue, setEventVenue] = useState("");
  const [eventDate, setEventDate] = useState("");

  const [ingestPath, setIngestPath] = useState("media");
  const [ingestRecursive, setIngestRecursive] = useState(true);
  const [ingestNote, setIngestNote] = useState("");

  const [prompt, setPrompt] = useState("Create a 60-second highlight focused on dancing.");
  const [outputType, setOutputType] = useState<OutputType>(DEFAULT_OUTPUT_TYPE);
  const [durationSeconds, setDurationSeconds] = useState(60);
  const [includeAssetIds, setIncludeAssetIds] = useState("");
  const [excludeAssetIds, setExcludeAssetIds] = useState("");
  const [wantSubtitles, setWantSubtitles] = useState(false);
  const [wantOverlays, setWantOverlays] = useState(false);
  const [planPreviewJson, setPlanPreviewJson] = useState("");
  const [lastRenderNote, setLastRenderNote] = useState("");

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const storage = globalThis.localStorage;
    if (typeof storage?.setItem === "function") {
      storage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profiles));
    }
  }, [profiles]);

  useEffect(() => {
    void refreshEvents();
  }, [tenantId, apiBaseUrl]);

  useEffect(() => {
    if (!selectedEventId) {
      setSummary(null);
      setRenders([]);
      return;
    }
    void refreshSelectedEvent();
  }, [selectedEventId, tenantId, apiBaseUrl]);

  useEffect(() => {
    if (!selectedEventId || !summary) return;
    const pending = summary.stats.index_jobs_queued + summary.stats.index_jobs_running;
    if (pending <= 0) return;
    const id = window.setInterval(() => {
      void (async () => {
        try {
          const [eventSummary, eventRenders] = await Promise.all([
            api.getEventSummary(tenantId, selectedEventId),
            api.listEventRenders(tenantId, selectedEventId)
          ]);
          setSummary(eventSummary);
          setRenders(eventRenders.renders);
          const p = eventSummary.stats.index_jobs_queued + eventSummary.stats.index_jobs_running;
          if (p === 0) {
            setStatus(
              `Indexing finished for ${selectedEventId} (${eventSummary.stats.index_jobs_completed} ok` +
                (eventSummary.stats.index_jobs_failed > 0 ? `, ${eventSummary.stats.index_jobs_failed} failed` : "") +
                ")."
            );
          }
        } catch {
          /* keep polling; next tick may succeed */
        }
      })();
    }, 10_000);
    return () => window.clearInterval(id);
  }, [
    selectedEventId,
    tenantId,
    api,
    summary?.stats.index_jobs_queued,
    summary?.stats.index_jobs_running
  ]);

  async function runAction(action: () => Promise<void>) {
    setLoading(true);
    setErrorMessage("");
    try {
      await action();
    } catch (error) {
      setErrorMessage(asErrorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  function parseIdList(value: string): string[] {
    return value
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  }

  function addProfile() {
    const profile = newProfile.trim();
    if (!profile) return;
    if (!profiles.includes(profile)) {
      setProfiles((prev) => [...prev, profile]);
    }
    setTenantId(profile);
    setSelectedEventId("");
    setNewProfile("");
  }

  async function refreshEvents() {
    await runAction(async () => {
      const response = await api.listEvents(tenantId);
      setEvents(response.events);
      if (!response.events.some((event) => event.id === selectedEventId)) {
        setSelectedEventId(response.events[0]?.id ?? "");
      }
      setStatus(`Loaded ${response.events.length} event(s) for ${tenantId}.`);
    });
  }

  async function refreshSelectedEvent() {
    if (!selectedEventId) return;
    if (
      events.length > 0 &&
      !events.some((e) => e.id === selectedEventId && e.tenant_id === tenantId)
    ) {
      return;
    }
    await runAction(async () => {
      const [eventSummary, eventRenders] = await Promise.all([
        api.getEventSummary(tenantId, selectedEventId),
        api.listEventRenders(tenantId, selectedEventId)
      ]);
      setSummary(eventSummary);
      setRenders(eventRenders.renders);
      setStatus(`Loaded dashboard for ${selectedEventId}.`);
    });
  }

  async function handleCreateEvent(e: FormEvent) {
    e.preventDefault();
    if (!eventTitle.trim() || !eventType.trim()) {
      setErrorMessage("Event title and event type are required.");
      return;
    }
    await runAction(async () => {
      const created = await api.createEvent({
        tenant_id: tenantId,
        title: eventTitle.trim(),
        event_type: eventType.trim(),
        venue: eventVenue.trim() || undefined,
        date: eventDate.trim() || undefined
      });
      setSelectedEventId(created.id);
      await refreshEvents();
      setStatus(`Created event ${created.id} for ${tenantId}.`);
    });
  }

  async function handleIngest() {
    if (!selectedEventId || !ingestPath.trim()) {
      setErrorMessage("Select an event and enter a path (file or folder on the backend host).");
      return;
    }
    await runAction(async () => {
      const response = await api.ingestFromPath({
        tenant_id: tenantId,
        event_id: selectedEventId,
        path: ingestPath.trim(),
        recursive: ingestRecursive
      });
      setIngestNote(
        `Registered ${response.count} file(s)${response.failed != null && response.failed > 0 ? `, ${response.failed} failed` : ""}. ` +
          `Indexing runs in the background; watch Event Summary for progress. First asset: ${response.assets[0]?.asset_id ?? "—"}.`
      );
      setStatus(`Ingest registered ${response.count} file(s) for ${selectedEventId}; indexing continues in the background.`);
      await refreshSelectedEvent();
    });
  }

  async function handleCreatePlan() {
    if (!selectedEventId || !prompt.trim()) {
      setErrorMessage("Select an event and enter a prompt.");
      return;
    }
    await runAction(async () => {
      const response = await api.createPlan({
        tenant_id: tenantId,
        event_id: selectedEventId,
        output_type: outputType,
        prompt,
        target_duration_seconds: durationSeconds,
        include_faces: [],
        include_asset_ids: parseIdList(includeAssetIds),
        excluded_asset_ids: parseIdList(excludeAssetIds),
        include_media_types: [],
        render_subtitles: wantSubtitles,
        render_overlays: wantOverlays
      });
      setPlanPreviewJson(JSON.stringify(response.plan, null, 2));
      setStatus("Plan created (preview below).");
    });
  }

  async function handleRender() {
    if (!selectedEventId || !prompt.trim()) {
      setErrorMessage("Select an event and enter a prompt.");
      return;
    }
    await runAction(async () => {
      const response = await api.render({
        tenant_id: tenantId,
        event_id: selectedEventId,
        output_type: outputType,
        prompt,
        target_duration_seconds: durationSeconds,
        include_faces: [],
        include_asset_ids: parseIdList(includeAssetIds),
        excluded_asset_ids: parseIdList(excludeAssetIds),
        include_media_types: [],
        render_subtitles: wantSubtitles,
        render_overlays: wantOverlays
      });
      setPlanPreviewJson(JSON.stringify(response.plan, null, 2));
      setLastRenderNote(`Render job ${response.render_job.id} — ${response.render_job.status}.`);
      setStatus(`Queued render ${response.render_job.id}.`);
      await refreshSelectedEvent();
    });
  }

  async function handleRegenerate() {
    if (!selectedEventId || !prompt.trim()) {
      setErrorMessage("Select an event and enter a prompt.");
      return;
    }
    await runAction(async () => {
      const response = await api.regenerate({
        tenant_id: tenantId,
        event_id: selectedEventId,
        output_type: outputType,
        prompt,
        target_duration_seconds: durationSeconds,
        include_asset_ids: parseIdList(includeAssetIds),
        exclude_asset_ids: parseIdList(excludeAssetIds),
        include_media_types: [],
        render_subtitles: wantSubtitles,
        render_overlays: wantOverlays
      });
      setPlanPreviewJson(JSON.stringify(response.plan, null, 2));
      setLastRenderNote(`Render job ${response.render_job.id} — ${response.render_job.status}.`);
      setStatus(`Regenerate queued: ${response.render_job.id}.`);
      await refreshSelectedEvent();
    });
  }

  const selectedEvent = events.find((event) => event.id === selectedEventId) ?? null;

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>VideoWala PoC Dashboard</h1>
          <p>Profile-based event workspace with event health and render visibility.</p>
        </div>
        <label>
          API base URL
          <input value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
        </label>
      </header>

      <section className="card">
        <h2>Profiles</h2>
        <div className="profiles-row">
          <label>
            Active profile
            <select
              value={tenantId}
              onChange={(e) => {
                setTenantId(e.target.value);
                setSelectedEventId("");
              }}
            >
              {profiles.map((profile) => (
                <option key={profile} value={profile}>
                  {profile}
                </option>
              ))}
            </select>
          </label>
          <label>
            Add profile (tenant_id)
            <input value={newProfile} onChange={(e) => setNewProfile(e.target.value)} placeholder="tenant_x" />
          </label>
          <button onClick={addProfile} disabled={loading}>
            Add + Switch
          </button>
          <button onClick={() => void refreshEvents()} disabled={loading}>
            Refresh Events
          </button>
        </div>
      </section>

      <section className="layout">
        <aside className="card">
          <h2>Events</h2>
          <form className="stack" onSubmit={(e) => void handleCreateEvent(e)}>
            <h3>Create Event</h3>
            <input value={eventTitle} onChange={(e) => setEventTitle(e.target.value)} placeholder="Event title" />
            <input value={eventType} onChange={(e) => setEventType(e.target.value)} placeholder="Event type" />
            <input value={eventVenue} onChange={(e) => setEventVenue(e.target.value)} placeholder="Venue (optional)" />
            <input value={eventDate} onChange={(e) => setEventDate(e.target.value)} placeholder="Date (optional)" />
            <button type="submit" disabled={loading}>
              Create Event
            </button>
          </form>
          <div className="events-list">
            {events.map((event) => (
              <button
                key={event.id}
                className={event.id === selectedEventId ? "event-item active" : "event-item"}
                onClick={() => setSelectedEventId(event.id)}
                disabled={loading}
              >
                <strong>{event.title}</strong>
                <span>{event.event_type}</span>
              </button>
            ))}
            {events.length === 0 ? <p className="muted">No events for this profile yet.</p> : null}
          </div>
        </aside>

        <main className="main-pane">
          <section className="card">
            <h2>Event Summary</h2>
            {selectedEvent ? (
              <div>
                <p>
                  <strong>{selectedEvent.title}</strong> ({selectedEvent.event_type}) - <code>{selectedEvent.id}</code>
                </p>
                {summary ? (
                  <div className="stats-grid">
                    <div className="stat-card">
                      <h3>Media</h3>
                      <p>{summary.stats.assets_total} assets</p>
                      <p className="muted">
                        {summary.stats.images_total} images / {summary.stats.videos_total} videos
                      </p>
                    </div>
                    <div className="stat-card">
                      <h3>Faces</h3>
                      <p>{summary.stats.faces_saved ? "Saved" : "Not saved"}</p>
                      <p className="muted">
                        {summary.stats.face_references_total} references / {summary.stats.face_match_insights_total} matches
                      </p>
                    </div>
                    <div className="stat-card">
                      <h3>Indexing</h3>
                      <p>
                        {summary.stats.index_jobs_completed + summary.stats.index_jobs_failed} /{" "}
                        {summary.stats.index_jobs_total || summary.stats.assets_total} jobs done
                      </p>
                      <p className="muted">
                        {summary.stats.index_jobs_queued} queued, {summary.stats.index_jobs_running} running,{" "}
                        {summary.stats.index_jobs_failed} failed
                      </p>
                    </div>
                    <div className="stat-card">
                      <h3>Renders</h3>
                      <p>{summary.stats.renders_total} jobs</p>
                      <p className="muted">
                        {summary.stats.renders_completed} done, {summary.stats.renders_running} running,{" "}
                        {summary.stats.renders_failed} failed
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="muted">Loading event summary...</p>
                )}
              </div>
            ) : (
              <p className="muted">Select an event to view profile-specific dashboard stats.</p>
            )}
          </section>

          <section className="card">
            <h2>Ingest media</h2>
            <p className="muted">
              Path must be reachable from the <strong>backend</strong> host (file or folder). Known image/video
              extensions are registered and indexed.
            </p>
            <div className="workflow-grid">
              <label>
                Path (file or folder)
                <input
                  value={ingestPath}
                  onChange={(e) => setIngestPath(e.target.value)}
                  placeholder="e.g. test/media or /data/wedding"
                  disabled={!selectedEventId}
                />
              </label>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={ingestRecursive}
                  onChange={(e) => setIngestRecursive(e.target.checked)}
                  disabled={!selectedEventId}
                />
                Include subfolders
              </label>
              <button type="button" onClick={() => void handleIngest()} disabled={loading || !selectedEventId}>
                Ingest
              </button>
            </div>
            {ingestNote ? <p className="pipeline-note">{ingestNote}</p> : null}
          </section>

          <section className="card">
            <h2>Plan + render</h2>
            <p className="muted">
              Uses the same backend routes as the API docs: <code>/requests/plan</code>, <code>/requests/render</code>,{" "}
              <code>/requests/feedback/regenerate</code>.
            </p>
            <div className="workflow-grid">
              <label className="span-2">
                Prompt
                <input value={prompt} onChange={(e) => setPrompt(e.target.value)} disabled={!selectedEventId} />
              </label>
              <label>
                Output type
                <select
                  value={outputType}
                  onChange={(e) => setOutputType(e.target.value as OutputType)}
                  disabled={!selectedEventId}
                >
                  <option value="highlight_reel">highlight_reel</option>
                  <option value="chronological_film">chronological_film</option>
                  <option value="person_focus_reel">person_focus_reel</option>
                </select>
              </label>
              <label>
                Target duration (s)
                <input
                  type="number"
                  min={10}
                  max={3600}
                  value={durationSeconds}
                  onChange={(e) => setDurationSeconds(Number(e.target.value))}
                  disabled={!selectedEventId}
                />
              </label>
              <label>
                Include asset IDs (comma-separated)
                <input
                  value={includeAssetIds}
                  onChange={(e) => setIncludeAssetIds(e.target.value)}
                  disabled={!selectedEventId}
                />
              </label>
              <label>
                Exclude asset IDs (comma-separated)
                <input
                  value={excludeAssetIds}
                  onChange={(e) => setExcludeAssetIds(e.target.value)}
                  disabled={!selectedEventId}
                />
              </label>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={wantSubtitles}
                  onChange={(e) => setWantSubtitles(e.target.checked)}
                  disabled={!selectedEventId}
                />
                Burn ASR subtitles
              </label>
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={wantOverlays}
                  onChange={(e) => setWantOverlays(e.target.checked)}
                  disabled={!selectedEventId}
                />
                Draw OCR overlays
              </label>
              <div className="button-row">
                <button type="button" onClick={() => void handleCreatePlan()} disabled={loading || !selectedEventId}>
                  Create plan
                </button>
                <button type="button" onClick={() => void handleRender()} disabled={loading || !selectedEventId}>
                  Render
                </button>
                <button type="button" onClick={() => void handleRegenerate()} disabled={loading || !selectedEventId}>
                  Regenerate
                </button>
              </div>
            </div>
            {lastRenderNote ? <p className="pipeline-note">{lastRenderNote}</p> : null}
            {planPreviewJson ? (
              <details className="plan-details">
                <summary>Latest plan JSON</summary>
                <pre>{planPreviewJson}</pre>
              </details>
            ) : null}
          </section>

          <section className="card">
            <h2>Render Jobs</h2>
            {renders.length === 0 ? <p className="muted">No renders found for selected event.</p> : null}
            <div className="render-list">
              {renders.map((job) => (
                <article key={job.id} className="render-item">
                  <div>
                    <strong>{job.id}</strong>
                    <p className="muted">
                      {job.status}
                      {job.progress_percent != null ? ` · ${job.progress_percent}%` : ""}
                      {job.error_message ? ` · ${job.error_message}` : ""} — {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>
                  {job.status === "completed" ? (
                    <a href={api.getRenderVideoUrl(job.id, tenantId)} target="_blank" rel="noreferrer">
                      Open Video
                    </a>
                  ) : null}
                </article>
              ))}
            </div>
          </section>
        </main>
      </section>

      <footer>
        <strong>Status:</strong> {status}
        {errorMessage ? <p className="error">{errorMessage}</p> : null}
      </footer>
    </div>
  );
}
