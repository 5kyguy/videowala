import { FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError, createApiClient, getDefaultApiBaseUrl } from "./api";
import type {
  Event,
  EventSummary,
  OutputType,
  Person,
  PersonFaceReferenceListItem,
  PhotoCurationItem,
  PhotoCurationListResponse,
  RenderJobListItem,
  VideoOrientation
} from "./types";

function partitionPhotoCuration(items: PhotoCurationItem[]) {
  const kept = items.filter((i) => i.keep && !i.is_duplicate);
  const duplicates = items.filter((i) => i.is_duplicate);
  const rejected = items.filter((i) => !i.keep && !i.is_duplicate);
  return { kept, duplicates, rejected };
}

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
  /** When set, preview panel shows this job (inline video if completed). */
  const [selectedRenderJobId, setSelectedRenderJobId] = useState<string | null>(null);
  const [persons, setPersons] = useState<Person[]>([]);
  const [faceRefs, setFaceRefs] = useState<PersonFaceReferenceListItem[]>([]);
  const [photoCuration, setPhotoCuration] = useState<PhotoCurationListResponse | null>(null);
  const [photoGalleryBusy, setPhotoGalleryBusy] = useState(false);
  const [activePage, setActivePage] = useState<"dashboard" | "album">("dashboard");
  const [albumSectionFilter, setAlbumSectionFilter] = useState<"all" | "kept" | "duplicates" | "rejected">("all");

  const photoParts = useMemo(
    () => partitionPhotoCuration(photoCuration?.items ?? []),
    [photoCuration?.items]
  );

  const [eventTitle, setEventTitle] = useState("wedding1");
  const [eventType, setEventType] = useState("wedding");
  const [eventVenue, setEventVenue] = useState("");
  const [eventDate, setEventDate] = useState("");

  const [ingestPath, setIngestPath] = useState("media");
  const [ingestNote, setIngestNote] = useState("");

  const [newPersonName, setNewPersonName] = useState("");
  const [newPersonPhoto, setNewPersonPhoto] = useState<File | null>(null);
  const [extraRefPersonId, setExtraRefPersonId] = useState("");
  const [extraRefPhoto, setExtraRefPhoto] = useState<File | null>(null);

  const [prompt, setPrompt] = useState("Create a 60-second highlight focused on dancing.");
  const [outputType, setOutputType] = useState<OutputType>(DEFAULT_OUTPUT_TYPE);
  const [durationSeconds, setDurationSeconds] = useState(60);
  const [includeAssetIds, setIncludeAssetIds] = useState("");
  const [excludeAssetIds, setExcludeAssetIds] = useState("");
  const [videoOrientation, setVideoOrientation] = useState<VideoOrientation>("landscape");
  const [planPreviewJson, setPlanPreviewJson] = useState("");
  const [lastRenderNote, setLastRenderNote] = useState("");

  const [loading, setLoading] = useState(false);
  const [faceReindexBusy, setFaceReindexBusy] = useState(false);
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
      setPersons([]);
      setFaceRefs([]);
      setPhotoCuration(null);
      setSelectedRenderJobId(null);
      return;
    }
    setPhotoCuration(null);
    setSelectedRenderJobId(null);
    void refreshSelectedEvent();
  }, [selectedEventId, tenantId, apiBaseUrl]);

  useEffect(() => {
    if (selectedRenderJobId && !renders.some((r) => r.id === selectedRenderJobId)) {
      setSelectedRenderJobId(null);
    }
  }, [renders, selectedRenderJobId]);

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

  useEffect(() => {
    if (!selectedEventId) return;
    const busy = renders.some((r) => r.status === "queued" || r.status === "running");
    if (!busy) return;
    const id = window.setInterval(() => {
      void (async () => {
        try {
          const eventRenders = await api.listEventRenders(tenantId, selectedEventId);
          setRenders(eventRenders.renders);
        } catch {
          /* ignore */
        }
      })();
    }, 3000);
    return () => window.clearInterval(id);
  }, [selectedEventId, tenantId, api, renders]);

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

  async function loadEventDashboard() {
    if (!selectedEventId) return;
    if (
      events.length > 0 &&
      !events.some((e) => e.id === selectedEventId && e.tenant_id === tenantId)
    ) {
      return;
    }
    const [eventSummary, eventRenders, personsRes, refsRes, photoRes] = await Promise.all([
      api.getEventSummary(tenantId, selectedEventId),
      api.listEventRenders(tenantId, selectedEventId),
      api.listPersons(tenantId, selectedEventId),
      api.listEventPersonReferences(tenantId, selectedEventId),
      api.getPhotoCuration(tenantId, selectedEventId)
    ]);
    setSummary(eventSummary);
    setRenders(eventRenders.renders);
    setPersons(personsRes.persons);
    setFaceRefs(refsRes.references);
    setPhotoCuration(photoRes);
    setStatus(`Loaded dashboard for ${selectedEventId}.`);
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
      await loadEventDashboard();
    });
  }

  async function refreshPhotoCuration() {
    if (!selectedEventId) return;
    setPhotoGalleryBusy(true);
    setErrorMessage("");
    try {
      const res = await api.getPhotoCuration(tenantId, selectedEventId);
      setPhotoCuration(res);
      setStatus(`Photo curation: ${res.items.length} indexed image(s).`);
    } catch (error) {
      setErrorMessage(asErrorMessage(error));
    } finally {
      setPhotoGalleryBusy(false);
    }
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

  async function handleDeleteEvent(eventId: string) {
    if (!window.confirm(`Delete event ${eventId}? This removes assets, insights, and render history.`)) {
      return;
    }
    await runAction(async () => {
      await api.deleteEvent(eventId, tenantId);
      if (selectedEventId === eventId) {
        setSelectedEventId("");
        setPhotoCuration(null);
        setSelectedRenderJobId(null);
      }
      await refreshEvents();
      setStatus(`Deleted event ${eventId}.`);
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
        recursive: true
      });
      setIngestNote(
        `Registered ${response.count} file(s)${response.failed != null && response.failed > 0 ? `, ${response.failed} failed` : ""}. ` +
          `Indexing runs in the background; watch Event Summary for progress. First asset: ${response.assets[0]?.asset_id ?? "—"}.`
      );
      setStatus(`Ingest registered ${response.count} file(s) for ${selectedEventId}; indexing continues in the background.`);
      await refreshSelectedEvent();
    });
  }

  async function handleAddPersonWithPhoto(e: FormEvent) {
    e.preventDefault();
    if (!selectedEventId || !newPersonName.trim()) {
      setErrorMessage("Enter a display name for the person.");
      return;
    }
    if (!newPersonPhoto) {
      setErrorMessage("Choose a reference photo (stored for this event on the backend).");
      return;
    }
    await runAction(async () => {
      const person = await api.createPerson({
        tenant_id: tenantId,
        event_id: selectedEventId,
        display_name: newPersonName.trim()
      });
      await api.uploadFaceReference(tenantId, selectedEventId, person.id, newPersonPhoto);
      setNewPersonName("");
      setNewPersonPhoto(null);
      setStatus(`Added person "${person.display_name}" with a face reference.`);
      await refreshSelectedEvent();
    });
  }

  async function handleAddExtraFaceReference(e: FormEvent) {
    e.preventDefault();
    if (!selectedEventId || !extraRefPersonId || !extraRefPhoto) {
      setErrorMessage("Select an existing person and choose a photo.");
      return;
    }
    await runAction(async () => {
      await api.uploadFaceReference(tenantId, selectedEventId, extraRefPersonId, extraRefPhoto);
      setExtraRefPhoto(null);
      setStatus("Added another reference photo for matching.");
      await refreshSelectedEvent();
    });
  }

  async function handleReindexFaces() {
    if (!selectedEventId || faceReindexBusy) return;
    setFaceReindexBusy(true);
    setErrorMessage("");
    try {
      const result = await api.reindexFaces(selectedEventId, tenantId);
      setStatus(`Face matching updated for ${result.asset_count} asset(s) (fast path — face insights only).`);
      await loadEventDashboard();
    } catch (error) {
      setErrorMessage(asErrorMessage(error));
    } finally {
      setFaceReindexBusy(false);
    }
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
        include_media_types: ["video"],
        video_orientation: videoOrientation
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
        include_media_types: ["video"],
        video_orientation: videoOrientation
      });
      setPlanPreviewJson(JSON.stringify(response.plan, null, 2));
      setLastRenderNote(`Render job ${response.render_job.id} — ${response.render_job.status}.`);
      setStatus(`Render ${response.render_job.id} — ${response.render_job.status} (runs in background; list updates every few seconds).`);
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
        include_media_types: ["video"],
        video_orientation: videoOrientation
      });
      setPlanPreviewJson(JSON.stringify(response.plan, null, 2));
      setLastRenderNote(`Render job ${response.render_job.id} — ${response.render_job.status}.`);
      setStatus(`Regenerate ${response.render_job.id} — ${response.render_job.status} (background).`);
      await refreshSelectedEvent();
    });
  }

  async function handleDeleteRender(renderJobId: string) {
    if (!window.confirm(`Delete render ${renderJobId}? This also removes generated files.`)) return;
    await runAction(async () => {
      await api.deleteRenderJob(renderJobId, tenantId);
      if (selectedRenderJobId === renderJobId) {
        setSelectedRenderJobId(null);
      }
      if (selectedEventId) {
        const eventRenders = await api.listEventRenders(tenantId, selectedEventId);
        setRenders(eventRenders.renders);
      }
      setStatus(`Deleted render ${renderJobId}.`);
    });
  }

  const selectedEvent = events.find((event) => event.id === selectedEventId) ?? null;
  const selectedRenderJob = selectedRenderJobId
    ? renders.find((r) => r.id === selectedRenderJobId) ?? null
    : null;

  const albumSectionVisible = {
    kept: albumSectionFilter === "all" || albumSectionFilter === "kept",
    duplicates: albumSectionFilter === "all" || albumSectionFilter === "duplicates",
    rejected: albumSectionFilter === "all" || albumSectionFilter === "rejected"
  };

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>VideoWala PoC Dashboard</h1>
          <p>(something here)</p>
        </div>
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
              <div key={event.id} className={event.id === selectedEventId ? "event-item active" : "event-item"}>
                <button type="button" className="event-item-select" onClick={() => setSelectedEventId(event.id)} disabled={loading}>
                  <strong>{event.title}</strong>
                  <span>{event.event_type}</span>
                </button>
                <button
                  type="button"
                  className="danger-button"
                  onClick={() => void handleDeleteEvent(event.id)}
                  disabled={loading}
                >
                  Delete
                </button>
              </div>
            ))}
            {events.length === 0 ? <p className="muted">No events for this profile yet.</p> : null}
          </div>
        </aside>

        <main className="main-pane">
          {activePage === "album" ? (
            <section className="card photo-album-page">
              <div className="photo-album-header">
                <div>
                  <h2>Photo album</h2>
                  <p className="muted">
                    Review curated images in a dedicated page view. Use filters to focus each bucket.
                  </p>
                </div>
                <div className="button-row">
                  <button type="button" onClick={() => setActivePage("dashboard")}>
                    Back to dashboard
                  </button>
                  <button
                    type="button"
                    onClick={() => void refreshPhotoCuration()}
                    disabled={!selectedEventId || photoGalleryBusy}
                  >
                    {photoGalleryBusy ? "Refreshing…" : "Refresh picks"}
                  </button>
                  {selectedEventId ? (
                    <a
                      className="button-link"
                      href={api.exportKeptPhotosUrl(selectedEventId, tenantId)}
                      download="kept_photos.zip"
                    >
                      Download kept photos (ZIP)
                    </a>
                  ) : null}
                </div>
              </div>
              <div className="callout callout-warning">
                <strong>Heads up — data transfer:</strong> Opening the album page fetches full-resolution images from
                the server and consumes bandwidth, similar to downloading the ZIP. Only open it when you need to
                review photos interactively.
              </div>
              <div className="workflow-grid">
                <label>
                  Show section
                  <select
                    value={albumSectionFilter}
                    onChange={(e) =>
                      setAlbumSectionFilter(e.target.value as "all" | "kept" | "duplicates" | "rejected")
                    }
                  >
                    <option value="all">All</option>
                    <option value="kept">Kept only</option>
                    <option value="duplicates">Duplicates only</option>
                    <option value="rejected">Rejected only</option>
                  </select>
                </label>
              </div>
              {!selectedEventId ? (
                <p className="muted">Select an event to load album images.</p>
              ) : photoCuration && photoCuration.items.length === 0 ? (
                <p className="muted">No indexed images for this event yet.</p>
              ) : photoCuration ? (
                <div className="photo-curation-panels">
                  {albumSectionVisible.kept ? (
                    <div className="photo-curation-panel">
                      <h3>Kept ({photoParts.kept.length})</h3>
                      <div className="photo-curation-grid">
                        {photoParts.kept.map((item) => (
                          <figure key={item.segment_id} className="photo-curation-thumb">
                            <img
                              src={api.getAssetMediaUrl(selectedEventId, item.asset_id, tenantId)}
                              alt=""
                              loading="lazy"
                            />
                            <figcaption>
                              <code>{item.asset_id}</code>
                              <span className="muted"> · score {item.score.toFixed(2)}</span>
                            </figcaption>
                          </figure>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {albumSectionVisible.duplicates ? (
                    <div className="photo-curation-panel">
                      <h3>Duplicates ({photoParts.duplicates.length})</h3>
                      <div className="photo-curation-grid">
                        {photoParts.duplicates.map((item) => (
                          <figure key={item.segment_id} className="photo-curation-thumb is-dim">
                            <img
                              src={api.getAssetMediaUrl(selectedEventId, item.asset_id, tenantId)}
                              alt=""
                              loading="lazy"
                            />
                            <figcaption>
                              <code>{item.asset_id}</code>
                              <span className="muted"> · dup</span>
                            </figcaption>
                          </figure>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {albumSectionVisible.rejected ? (
                    <div className="photo-curation-panel">
                      <h3>Rejected / low score ({photoParts.rejected.length})</h3>
                      <div className="photo-curation-grid">
                        {photoParts.rejected.map((item) => (
                          <figure key={item.segment_id} className="photo-curation-thumb is-dim">
                            <img
                              src={api.getAssetMediaUrl(selectedEventId, item.asset_id, tenantId)}
                              alt=""
                              loading="lazy"
                            />
                            <figcaption>
                              <code>{item.asset_id}</code>
                              {item.reject_reasons.length > 0 ? (
                                <span className="muted"> · {item.reject_reasons.join(", ")}</span>
                              ) : null}
                            </figcaption>
                          </figure>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="muted">Loading photo picks…</p>
              )}
            </section>
          ) : null}
          {activePage === "dashboard" ? (
            <>
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
                        {summary.stats.index_jobs_total || summary.stats.assets_total} assets indexed
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
              Paths are resolved on the machine running the backend: use an absolute path, or a path relative to the Videowala repository root. Files or folders are accepted.
            </p>
            <div className="workflow-grid">
              <label>
                Path
                <input
                  value={ingestPath}
                  onChange={(e) => setIngestPath(e.target.value)}
                  placeholder="e.g. test/media or /data/wedding"
                  disabled={!selectedEventId}
                />
              </label>
              <button type="button" onClick={() => void handleIngest()} disabled={loading || !selectedEventId}>
                Ingest
              </button>
            </div>
            {ingestNote ? <p className="pipeline-note">{ingestNote}</p> : null}
          </section>

          <section className="card photo-curation-card">
            <h2>Photo curation</h2>
            <p className="muted">
              Indexed stills are scored for duplicates and weak takes. Use this gallery to review picks; exports include
              <strong> kept</strong> images only (not duplicates). Video reels and films use <strong>video clips only</strong>—see
              the next section.
            </p>
            <div className="photo-curation-toolbar">
              <button
                type="button"
                onClick={() => void refreshPhotoCuration()}
                disabled={!selectedEventId || photoGalleryBusy}
              >
                {photoGalleryBusy ? "Refreshing…" : "Refresh picks"}
              </button>
              {selectedEventId ? (
                <a
                  className="button-link"
                  href={api.exportKeptPhotosUrl(selectedEventId, tenantId)}
                  download="kept_photos.zip"
                >
                  Download kept photos (ZIP)
                </a>
              ) : null}
              <button type="button" onClick={() => setActivePage("album")} disabled={!selectedEventId}>
                Open album page
              </button>
            </div>
            {!selectedEventId ? (
              <p className="muted">Select an event to load photo picks.</p>
            ) : photoCuration && photoCuration.items.length === 0 ? (
              <p className="muted">No indexed images for this event yet.</p>
            ) : (
              <p className="muted">
                {photoCuration!.items.length} indexed image(s). Use{" "}
                <strong>Open album page</strong> to review them — opening the album page fetches images and
                consumes server bandwidth, similar to downloading the ZIP.
              </p>
            )}
          </section>

          <section className="card">
            <h2>People &amp; face references</h2>
            <p className="muted">
              People are stored for this <strong>event</strong> only. Reference photos are uploaded to the backend and used
              when indexing runs to match faces in your media. Add names and photos anytime; if media is already indexed,
              use <strong>Re-run face matching</strong> to refresh face matches only.
            </p>
            <form className="stack face-form" onSubmit={(e) => void handleAddPersonWithPhoto(e)}>
              <h3>New person</h3>
              <div className="workflow-grid">
                <label>
                  Person name
                  <input
                    value={newPersonName}
                    onChange={(e) => setNewPersonName(e.target.value)}
                    placeholder="e.g. Alex"
                    disabled={!selectedEventId}
                    autoComplete="off"
                  />
                </label>
                <label>
                  Reference photo
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/gif"
                    onChange={(e) => setNewPersonPhoto(e.target.files?.[0] ?? null)}
                    disabled={!selectedEventId}
                  />
                </label>
                <button type="submit" disabled={loading || !selectedEventId}>
                  Add person
                </button>
              </div>
            </form>
            <form className="stack face-form" onSubmit={(e) => void handleAddExtraFaceReference(e)}>
              <h3>Re-upload photo for an existing person</h3>
              <div className="workflow-grid">
                <label>
                  Person
                  <select
                    value={extraRefPersonId}
                    onChange={(e) => setExtraRefPersonId(e.target.value)}
                    disabled={!selectedEventId || persons.length === 0}
                  >
                    <option value="">— select —</option>
                    {persons.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Photo
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/gif"
                    onChange={(e) => setExtraRefPhoto(e.target.files?.[0] ?? null)}
                    disabled={!selectedEventId || persons.length === 0}
                  />
                </label>
                <button type="submit" disabled={loading || !selectedEventId || persons.length === 0}>
                  Re-upload photo
                </button>
              </div>
            </form>
            <div className="button-row" style={{ marginTop: 12 }}>
              <button
                type="button"
                onClick={() => void handleReindexFaces()}
                disabled={loading || faceReindexBusy || !selectedEventId}
              >
                {faceReindexBusy ? "Updating face matches…" : "Re-run face matching on all media"}
              </button>
            </div>
            {faceRefs.length === 0 ? (
              <p className="muted" style={{ marginTop: 12 }}>
                No reference photos yet for this event.
              </p>
            ) : (
              <ul className="face-ref-list">
                {faceRefs.map((ref) => (
                  <li key={ref.id} className="face-ref-item">
                    <img
                      className="face-ref-thumb"
                      src={api.getFaceReferenceImageUrl(selectedEventId, ref.id, tenantId)}
                      alt=""
                      loading="lazy"
                    />
                    <div>
                      <strong>{ref.display_name}</strong>
                      <p className="muted small">
                        {ref.id} · {new Date(ref.created_at).toLocaleString()}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="card">
            <h2>Video plan + render</h2>
            <p className="muted">
              Planner and render use <strong>video assets only</strong> (still photos are not stitched into the MP4). Create a
              plan, then render or regenerate. Orientation is a center crop on source resolution—no subtitle burn-in.
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
                  min={5}
                  max={3600}
                  value={durationSeconds}
                  onChange={(e) => setDurationSeconds(Number(e.target.value))}
                  disabled={!selectedEventId}
                />
              </label>
              {/* <label>
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
              </label> */}
              <label>
                Output orientation
                <select
                  value={videoOrientation}
                  onChange={(e) => setVideoOrientation(e.target.value as VideoOrientation)}
                  disabled={!selectedEventId}
                >
                  <option value="landscape">Landscape (16:9 center crop)</option>
                  <option value="portrait">Portrait / reels (9:16 center crop)</option>
                </select>
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
            <p className="muted">Click a job to play the video here when it is completed.</p>
            {renders.length === 0 ? <p className="muted">No renders found for selected event.</p> : null}
            <div className="render-jobs-layout">
              <div className="render-list">
                {renders.map((job) => (
                  <article
                    key={job.id}
                    className={
                      "render-item" + (selectedRenderJobId === job.id ? " render-item-selected" : "")
                    }
                  >
                    <button
                      type="button"
                      className="render-item-select"
                      onClick={() => setSelectedRenderJobId(job.id)}
                    >
                      <strong>{job.id}</strong>
                      <p className="muted">
                        {job.status}
                        {job.progress_percent != null ? ` · ${job.progress_percent}%` : ""}
                        {job.error_message ? ` · ${job.error_message}` : ""} — {new Date(job.created_at).toLocaleString()}
                      </p>
                    </button>
                    {job.status === "completed" ? (
                      <div className="button-row" style={{ padding: "10px" }}>
                        <a
                          className="render-open-tab"
                          href={api.getRenderVideoUrl(job.id, tenantId)}
                          target="_blank"
                          rel="noreferrer"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Open in new tab
                        </a>
                        <button
                          type="button"
                          className="danger-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            void handleDeleteRender(job.id);
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    ) : (
                      <div className="button-row" style={{ padding: "10px" }}>
                        <button
                          type="button"
                          className="danger-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            void handleDeleteRender(job.id);
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    )}
                  </article>
                ))}
              </div>
              <div className="render-preview-panel">
                {selectedRenderJob ? (
                  selectedRenderJob.status === "completed" ? (
                    <div className="render-video-wrap">
                      {selectedRenderJob.planner_prompt ? (
                        <div className="render-prompt-for-video">
                          <p className="render-prompt-for-video-title">Prompt for this render was</p>
                          <p className="render-prompt-for-video-body">{selectedRenderJob.planner_prompt}</p>
                        </div>
                      ) : null}
                      <video
                        key={selectedRenderJob.id}
                        className="render-preview-video"
                        controls
                        playsInline
                        preload="metadata"
                        src={api.getRenderVideoUrl(selectedRenderJob.id, tenantId)}
                      />
                      <p className="muted small render-preview-caption">{selectedRenderJob.id}</p>
                    </div>
                  ) : (
                    <div className="render-preview-placeholder-wrap">
                      {selectedRenderJob.planner_prompt ? (
                        <div className="render-prompt-for-video">
                          <p className="render-prompt-for-video-title">Prompt for this render was</p>
                          <p className="render-prompt-for-video-body">{selectedRenderJob.planner_prompt}</p>
                        </div>
                      ) : null}
                      <p className="muted render-preview-placeholder">
                        <strong>{selectedRenderJob.id}</strong> is <strong>{selectedRenderJob.status}</strong>. The player
                        appears when the job completes.
                      </p>
                    </div>
                  )
                ) : (
                  <p className="muted render-preview-placeholder">Select a render to preview.</p>
                )}
              </div>
            </div>
          </section>
            </>
          ) : null}
        </main>
      </section>

      <footer>
        <strong>Status:</strong> {status}
        {errorMessage ? <p className="error">{errorMessage}</p> : null}
      </footer>
    </div>
  );
}
