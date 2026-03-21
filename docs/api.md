# API (current)

**Product context:** root [`README.md`](../README.md).

Base URL: `http://localhost:8000`

All endpoints are **tenant-scoped** by `tenant_id` and (where relevant) `event_id`.

**OCR:** indexing uses **PaddleOCR** only. OCR runs **after** the VLM step when any VLM tag intersects `OCR_TRIGGER_TAGS` (see `backend/app/config.py`). Event fields `predefined_tags`, `ocr_languages`, and `PATCH /events/{event_id}` control vocabulary and Paddle language codes.

**Indexing progress:** when stderr is a TTY, `tqdm` shows a **per-asset** bar (faces → ASR → VLM → gated OCR) and a **batch** bar over files. Set `INDEXING_PROGRESS=0` to disable, or `INDEXING_PROGRESS=1` to force on (e.g. piped logs).

## Health

- `GET /health`

Returns `{"status": "ok"}`.

## Events

- `POST /events`
- `PATCH /events/{event_id}?tenant_id=...` — update title, `predefined_tags`, `ocr_languages`, etc.
- `GET /events?tenant_id=...`
- `GET /events/{event_id}/summary?tenant_id=...`
- `GET /events/{event_id}/renders?tenant_id=...`

Example:

```bash
curl -s -X POST http://localhost:8000/events \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","title":"Demo Event","event_type":"wedding"}' | jq .
```

Event dashboard summary response includes:

- media totals (`assets_total`, image/video split, `has_media`)
- face readiness (`persons_total`, `face_references_total`, `faces_saved`, face-match counters)
- render status counters (`renders_total`, queued/running/completed/failed)

Event renders response includes render jobs for that event (latest first), suitable for profile dashboard listing. Each job may include `planner_prompt` (the user prompt from the content request that produced the plan for that render) when the job was created via `/requests/render` or `/requests/feedback/regenerate`.

## Assets (register-by-path + index)

- `POST /assets`

**Single file (legacy):** `tenant_id`, `event_id`, `media_path`, `media_type` (`image` | `video`).

**File or folder (batch):** `tenant_id`, `event_id`, `path` (absolute or relative to backend project root), optional `recursive` (default `true`). All files with known image/video extensions under that path are registered and indexed in one request.

Response — single: `{ "asset_id", "insights_generated" }`.  
Response — batch: `{ "batch": true, "count", "failed", "assets": [{ "asset_id", "media_path", "media_type", "insights_generated", "error" }, ...] }`.  
`failed` is how many files raised during indexing; `error` is `null` on success or a string message on failure (batch still returns **200** so the rest of the folder can finish).

Examples:

```bash
# One file (explicit type)
curl -s -X POST http://localhost:8000/assets \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","event_id":"event_...","media_path":"media/clip.mp4","media_type":"video"}' | jq .

# Whole folder, recursive
curl -s -X POST http://localhost:8000/assets \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_a","event_id":"event_...","path":"/absolute/or/relative/path/to/event_media","recursive":true}' | jq .
```

## Event context + semantic search

- `GET /events/{event_id}/context?tenant_id=...`
  - Optional filters: `insight_type=...`, `person_id=...`
- `GET /events/{event_id}/search?tenant_id=...&q=...&limit=20`
  - Requires pgvector data to exist; returns best-effort results (may be empty if Postgres/pgvector is unavailable or no vectors were upserted yet).

## Planner + rendering

- `POST /requests/plan`
- `POST /requests/render`
- `POST /requests/feedback/regenerate`
- `GET /renders/{render_job_id}/video?tenant_id=...`

All three accept mostly the same core request fields:

- `tenant_id`, `event_id`
- `output_type`: `highlight_reel | chronological_film | person_focus_reel`
- `prompt`
- `target_duration_seconds`
- `include_asset_ids` (array of asset IDs to pin early)
- `excluded_asset_ids` (array; used by `/requests/plan` and `/requests/render`)
- `exclude_asset_ids` (array; used by `/requests/feedback/regenerate`)
- `include_media_types` (array of `image|video`)
- `video_orientation`: `landscape` | `portrait` — center crop to 16:9 or 9:16 (reels-style). ASR/OCR are for planner/indexing only; they are not burned into renders.

Regenerate endpoint differences:

- Uses `exclude_asset_ids` (note name change vs `excluded_asset_ids`)
- Does not accept `include_faces`

Render video fetch:

- Once a render job has `status="completed"` and `output_path` set, download it via:

  - `GET /renders/{render_job_id}/video?tenant_id=...` → `video/mp4` file response

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
