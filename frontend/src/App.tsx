import { FormEvent, useMemo, useState } from "react";

import { ApiError, createApiClient, getDefaultApiBaseUrl } from "./api";
import type { Event, OutputType, Person } from "./types";

const DEFAULT_OUTPUT_TYPE: OutputType = "highlight_reel";

function asErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `HTTP ${error.status}: ${error.message}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown error";
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(getDefaultApiBaseUrl());
  const api = useMemo(() => createApiClient({ baseUrl: apiBaseUrl }), [apiBaseUrl]);

  const [tenantId, setTenantId] = useState("tenant_a");
  const [eventId, setEventId] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [persons, setPersons] = useState<Person[]>([]);
  const [contextJson, setContextJson] = useState("{}");
  const [planJson, setPlanJson] = useState("{}");
  const [renderJson, setRenderJson] = useState("{}");
  const [faceMatchesJson, setFaceMatchesJson] = useState("[]");
  const [renderVideoSrc, setRenderVideoSrc] = useState<string | null>(null);

  const [eventTitle, setEventTitle] = useState("Demo Event");
  const [eventType, setEventType] = useState("wedding");
  const [ingestPath, setIngestPath] = useState("media");
  const [ingestRecursive, setIngestRecursive] = useState(true);
  const [prompt, setPrompt] = useState("Create a 60-second highlight focused on dancing.");
  const [durationSeconds, setDurationSeconds] = useState(60);
  const [includeAssetIds, setIncludeAssetIds] = useState("");
  const [excludeAssetIds, setExcludeAssetIds] = useState("");
  const [wantSubtitles, setWantSubtitles] = useState(false);
  const [wantOverlays, setWantOverlays] = useState(false);
  const [personName, setPersonName] = useState("Alice");
  const [personRefPath, setPersonRefPath] = useState("media/20250111_141008.jpg");
  const [selectedPersonId, setSelectedPersonId] = useState("");
  const [outputType, setOutputType] = useState<OutputType>(DEFAULT_OUTPUT_TYPE);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [errorMessage, setErrorMessage] = useState("");

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

  async function handleCreateEvent(e: FormEvent) {
    e.preventDefault();
    if (!tenantId || !eventTitle || !eventType) {
      setErrorMessage("tenant_id, title, and event_type are required.");
      return;
    }
    await runAction(async () => {
      const event = await api.createEvent({ tenant_id: tenantId, title: eventTitle, event_type: eventType });
      setEventId(event.id);
      setStatus(`Created event ${event.id}.`);
      await refreshEvents();
    });
  }

  async function refreshEvents() {
    await runAction(async () => {
      const response = await api.listEvents(tenantId);
      setEvents(response.events);
      if (!eventId && response.events.length > 0) {
        setEventId(response.events[0].id);
      }
      setStatus(`Loaded ${response.events.length} events.`);
    });
  }

  async function ingestFromPath() {
    if (!eventId || !ingestPath.trim()) {
      setErrorMessage("event_id and path (file or folder) are required.");
      return;
    }
    await runAction(async () => {
      const response = await api.ingestFromPath({
        tenant_id: tenantId,
        event_id: eventId,
        path: ingestPath.trim(),
        recursive: ingestRecursive
      });
      const totalInsights = response.assets.reduce((s, a) => s + a.insights_generated, 0);
      setStatus(
        `Ingested ${response.count} file(s) (${totalInsights} total insight rows). First: ${response.assets[0]?.asset_id ?? "—"}.`
      );
    });
  }

  async function loadContext() {
    if (!eventId) {
      setErrorMessage("event_id is required.");
      return;
    }
    await runAction(async () => {
      const response = await api.getContext(eventId, tenantId);
      setContextJson(JSON.stringify(response.context, null, 2));
      setStatus("Loaded event context.");
    });
  }

  async function requestPlan() {
    if (!eventId || !prompt) {
      setErrorMessage("event_id and prompt are required.");
      return;
    }
    await runAction(async () => {
      const response = await api.createPlan({
        tenant_id: tenantId,
        event_id: eventId,
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
      setPlanJson(JSON.stringify(response.plan, null, 2));
      setStatus("Planner response received.");
    });
  }

  async function requestRender() {
    if (!eventId || !prompt) {
      setErrorMessage("event_id and prompt are required.");
      return;
    }
    await runAction(async () => {
      const response = await api.render({
        tenant_id: tenantId,
        event_id: eventId,
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
      setPlanJson(JSON.stringify(response.plan, null, 2));
      setRenderJson(JSON.stringify(response.render_job, null, 2));
      setRenderVideoSrc(`${api.getRenderVideoUrl(response.render_job.id, tenantId)}&t=${Date.now()}`);
      setStatus(`Render job ${response.render_job.id} is ${response.render_job.status}.`);
    });
  }

  async function requestRegenerate() {
    if (!eventId || !prompt) {
      setErrorMessage("event_id and prompt are required.");
      return;
    }
    await runAction(async () => {
      const response = await api.regenerate({
        tenant_id: tenantId,
        event_id: eventId,
        output_type: outputType,
        prompt,
        target_duration_seconds: durationSeconds,
        include_asset_ids: parseIdList(includeAssetIds),
        exclude_asset_ids: parseIdList(excludeAssetIds),
        include_media_types: [],
        render_subtitles: wantSubtitles,
        render_overlays: wantOverlays
      });
      setPlanJson(JSON.stringify(response.plan, null, 2));
      setRenderJson(JSON.stringify(response.render_job, null, 2));
      setRenderVideoSrc(`${api.getRenderVideoUrl(response.render_job.id, tenantId)}&t=${Date.now()}`);
      setStatus(`Regenerated; render ${response.render_job.id} is ${response.render_job.status}.`);
    });
  }

  async function createPerson() {
    if (!eventId || !personName) {
      setErrorMessage("event_id and display name are required.");
      return;
    }
    await runAction(async () => {
      const created = await api.createPerson({
        tenant_id: tenantId,
        event_id: eventId,
        display_name: personName
      });
      setSelectedPersonId(created.id);
      setStatus(`Created person ${created.id}.`);
      await loadPersons();
    });
  }

  async function loadPersons() {
    if (!eventId) {
      setErrorMessage("event_id is required.");
      return;
    }
    await runAction(async () => {
      const response = await api.listPersons(tenantId, eventId);
      setPersons(response.persons);
      if (!selectedPersonId && response.persons.length > 0) {
        setSelectedPersonId(response.persons[0].id);
      }
      setStatus(`Loaded ${response.persons.length} persons.`);
    });
  }

  async function addReference() {
    if (!selectedPersonId || !eventId || !personRefPath) {
      setErrorMessage("person_id, event_id, and image_path are required.");
      return;
    }
    await runAction(async () => {
      await api.addPersonReference(selectedPersonId, {
        tenant_id: tenantId,
        event_id: eventId,
        image_path: personRefPath
      });
      setStatus("Person reference added.");
    });
  }

  async function reindexFaces() {
    if (!eventId) {
      setErrorMessage("event_id is required.");
      return;
    }
    await runAction(async () => {
      const response = await api.reindexFaces(eventId, tenantId);
      setStatus(`Face reindex done for ${response.asset_count} assets.`);
    });
  }

  async function loadFaceMatches() {
    if (!eventId) {
      setErrorMessage("event_id is required.");
      return;
    }
    await runAction(async () => {
      const response = await api.listFaceMatches(eventId, tenantId, selectedPersonId || undefined);
      setFaceMatchesJson(JSON.stringify(response.matches, null, 2));
      setStatus("Face matches loaded.");
    });
  }

  return (
    <div className="page">
      <header>
        <h1>VideoWala MVP Frontend</h1>
        <p>Simple UI for event ingest, context, planning, rendering, regenerate feedback, and face APIs.</p>
      </header>

      <section className="card">
        <h2>Connection</h2>
        <div className="grid">
          <label>
            API base URL
            <input value={apiBaseUrl} onChange={(e) => setApiBaseUrl(e.target.value)} />
          </label>
          <label>
            Tenant ID
            <input value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
          </label>
          <button onClick={() => void refreshEvents()} disabled={loading}>
            Refresh Events
          </button>
        </div>
      </section>

      <section className="card">
        <h2>Event</h2>
        <form className="grid" onSubmit={(e) => void handleCreateEvent(e)}>
          <label>
            Event title
            <input value={eventTitle} onChange={(e) => setEventTitle(e.target.value)} />
          </label>
          <label>
            Event type
            <input value={eventType} onChange={(e) => setEventType(e.target.value)} />
          </label>
          <button type="submit" disabled={loading}>
            Create Event
          </button>
          <label>
            Selected event ID
            <input value={eventId} onChange={(e) => setEventId(e.target.value)} />
          </label>
        </form>
        <pre>{JSON.stringify(events, null, 2)}</pre>
      </section>

      <section className="card">
        <h2>Ingest + Context</h2>
        <p className="hint">
          Enter a <strong>file</strong> or <strong>folder</strong> path on the backend host. Images and videos are auto-detected by extension (jpg, png, mp4, mov, …).
        </p>
        <div className="grid">
          <label>
            Path (file or folder)
            <input value={ingestPath} onChange={(e) => setIngestPath(e.target.value)} placeholder="e.g. /data/wedding or test/media" />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={ingestRecursive}
              onChange={(e) => setIngestRecursive(e.target.checked)}
            />
            Include subfolders
          </label>
          <button type="button" onClick={() => void ingestFromPath()} disabled={loading}>
            Ingest all media
          </button>
          <button type="button" onClick={() => void loadContext()} disabled={loading}>
            Load Event Context
          </button>
        </div>
        <pre>{contextJson}</pre>
      </section>

      <section className="card">
        <h2>Plan + Render + Regenerate</h2>
        <div className="grid">
          <label>
            Prompt
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </label>
          <label>
            Output type
            <select value={outputType} onChange={(e) => setOutputType(e.target.value as OutputType)}>
              <option value="highlight_reel">highlight_reel</option>
              <option value="chronological_film">chronological_film</option>
              <option value="person_focus_reel">person_focus_reel</option>
            </select>
          </label>
          <label>
            Target duration seconds
            <input
              type="number"
              min={10}
              max={3600}
              value={durationSeconds}
              onChange={(e) => setDurationSeconds(Number(e.target.value))}
            />
          </label>
          <label>
            Include asset IDs (comma-separated)
            <input value={includeAssetIds} onChange={(e) => setIncludeAssetIds(e.target.value)} />
          </label>
          <label>
            Exclude asset IDs (comma-separated)
            <input value={excludeAssetIds} onChange={(e) => setExcludeAssetIds(e.target.value)} />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={wantSubtitles}
              onChange={(e) => setWantSubtitles(e.target.checked)}
            />
            Burn ASR subtitles
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={wantOverlays}
              onChange={(e) => setWantOverlays(e.target.checked)}
            />
            Draw OCR overlays
          </label>
          <button onClick={() => void requestPlan()} disabled={loading}>
            Create Plan
          </button>
          <button onClick={() => void requestRender()} disabled={loading}>
            Render
          </button>
          <button onClick={() => void requestRegenerate()} disabled={loading}>
            Regenerate
          </button>
        </div>
        <div className="two-col">
          <pre>{planJson}</pre>
          <pre>{renderJson}</pre>
        </div>
        {renderVideoSrc ? (
          <div className="video-wrap">
            <h3>Rendered video</h3>
            <video controls src={renderVideoSrc} />
          </div>
        ) : null}
      </section>

      <section className="card">
        <h2>Face APIs</h2>
        <div className="grid">
          <label>
            Person name
            <input value={personName} onChange={(e) => setPersonName(e.target.value)} />
          </label>
          <button onClick={() => void createPerson()} disabled={loading}>
            Create Person
          </button>
          <button onClick={() => void loadPersons()} disabled={loading}>
            Load Persons
          </button>
          <label>
            Selected person ID
            <input value={selectedPersonId} onChange={(e) => setSelectedPersonId(e.target.value)} />
          </label>
          <label>
            Person reference image path
            <input value={personRefPath} onChange={(e) => setPersonRefPath(e.target.value)} />
          </label>
          <button onClick={() => void addReference()} disabled={loading}>
            Add Reference
          </button>
          <button onClick={() => void reindexFaces()} disabled={loading}>
            Reindex Faces
          </button>
          <button onClick={() => void loadFaceMatches()} disabled={loading}>
            Load Face Matches
          </button>
        </div>
        <pre>{JSON.stringify(persons, null, 2)}</pre>
        <pre>{faceMatchesJson}</pre>
      </section>

      <footer>
        <strong>Status:</strong> {status}
        {errorMessage ? <p className="error">{errorMessage}</p> : null}
      </footer>
    </div>
  );
}
