# API (current)

**Product context:** root [`README.md`](../README.md).

Base URL: `http://localhost:8000`

All endpoints are **tenant-scoped** by `tenant_id` and (where relevant) `event_id`.

## Health

- `GET /health`

Returns `{"status": "ok"}`.

## Events

- `POST /events`
- `GET /events?tenant_id=...`

Example:

```bash
curl -s -X POST http://localhost:8000/events \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","title":"Demo Event","event_type":"wedding"}' | jq .
```

## Assets (register-by-path + index)

- `POST /assets`

**Single file (legacy):** `tenant_id`, `event_id`, `media_path`, `media_type` (`image` | `video`).

**File or folder (batch):** `tenant_id`, `event_id`, `path` (absolute or relative to backend project root), optional `recursive` (default `true`). All files with known image/video extensions under that path are registered and indexed in one request.

Response — single: `{ "asset_id", "insights_generated" }`.  
Response — batch: `{ "batch": true, "count", "assets": [{ "asset_id", "media_path", "media_type", "insights_generated" }, ...] }`.

Examples:

```bash
# One file (explicit type)
curl -s -X POST http://localhost:8000/assets \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","event_id":"event_...","media_path":"media/clip.mp4","media_type":"video"}' | jq .

# Whole folder, recursive
curl -s -X POST http://localhost:8000/assets \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","event_id":"event_...","path":"/path/to/event_media","recursive":true}' | jq .
```

## Event context + semantic search

- `GET /events/{event_id}/context?tenant_id=...`
  - Optional filters: `insight_type=...`, `person_id=...`
- `GET /events/{event_id}/search?tenant_id=...&q=...&limit=20`
  - Requires stage2 semantic enabled *and* pgvector available; otherwise returns best-effort results (may be empty).

## Planner + rendering

- `POST /requests/plan`
- `POST /requests/render`
- `POST /requests/feedback/regenerate`

Render output streaming (MP4):

- `GET /renders/{render_job_id}/video?tenant_id=...`

Notes:

- Only works when the render job status is `completed`.
- Intended for the frontend to embed via a `<video>` tag.

All three accept mostly the same core request fields:

- `tenant_id`, `event_id`
- `output_type`: `highlight_reel | chronological_film | person_focus_reel`
- `prompt`
- `target_duration_seconds`
- `include_asset_ids` (array of asset IDs to pin early)
- `excluded_asset_ids` (array; used by `/requests/plan` and `/requests/render`)
- `exclude_asset_ids` (array; used by `/requests/feedback/regenerate`)
- `include_media_types` (array of `image|video`)

Regenerate endpoint differences:

- Uses `exclude_asset_ids` (note name change vs `excluded_asset_ids`)
- Does not accept `include_faces`

## Cleanup

- `POST /events/{event_id}/cleanup?tenant_id=...`

Deletes scratch directory for the tenant/event.

## Face APIs

- `POST /persons`
- `GET /persons?tenant_id=...&event_id=...`
- `POST /persons/{person_id}/references`
- `POST /events/{event_id}/faces/reindex?tenant_id=...`
- `GET /events/{event_id}/faces/matches?tenant_id=...&person_id=...`

Notes:

- Face recognition is **tenant/event scoped** and driven by person reference embeddings.
- The “real” recognition stack is gated (see `enable_real_face_recognition` in backend settings); indexing works in stub mode as well.
