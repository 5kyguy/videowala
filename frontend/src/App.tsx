import { FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError, createApiClient, getDefaultApiBaseUrl } from "./api";
import type { Event, EventSummary, RenderJobListItem } from "./types";

const PROFILE_STORAGE_KEY = "videowala_profiles";

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

  function addProfile() {
    const profile = newProfile.trim();
    if (!profile) return;
    if (!profiles.includes(profile)) {
      setProfiles((prev) => [...prev, profile]);
    }
    setTenantId(profile);
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
            <select value={tenantId} onChange={(e) => setTenantId(e.target.value)}>
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
            <h2>Render Jobs</h2>
            {renders.length === 0 ? <p className="muted">No renders found for selected event.</p> : null}
            <div className="render-list">
              {renders.map((job) => (
                <article key={job.id} className="render-item">
                  <div>
                    <strong>{job.id}</strong>
                    <p className="muted">
                      {job.status} - {new Date(job.created_at).toLocaleString()}
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
