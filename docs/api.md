# API (current)

**Product context:** root [`README.md`](../README.md).

Base URL: `http://localhost:8000`

All endpoints are **tenant-scoped** by `tenant_id` and (where relevant) `event_id`.

**OCR:** indexing uses **PaddleOCR** only. OCR runs **after** the VLM step when any VLM tag intersects `OCR_TRIGGER_TAGS` (see `backend/app/config.py`). Event fields `predefined_tags`, `ocr_languages`, and `PATCH /events/{event_id}` control vocabulary and Paddle language codes.

**Indexing progress:** when stderr is a TTY, `tqdm` shows a **batch** bar over files in folder ingest. Set `INDEXING_PROGRESS=0` to disable, or `INDEXING_PROGRESS=1` to force on (e.g. piped logs).

**Indexing execution:** each asset is indexed on a **background thread pool** (default **`INDEX_WORKERS=1`**, strictly serial). Heavy models (faces → ASR on video → VLM → gated OCR → embeddings) load and unload **one at a time** per asset for GPU PoC quality.

**Image ingest theme (optional):** `POST /assets` may include **`semantic_prompt`** (string). For **image** assets this runs the same semantic + cull pass as photo curation after embedding (re-ranks image segments in the event). Ignored for video assets. The created **`index_jobs`** row stores `semantic_prompt` when provided.

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
- **resource footprint**: `media_storage_bytes` and per-type `media_bytes_images` / `media_bytes_videos` from files the backend can resolve on disk; `media_storage_files_found` / `media_storage_files_missing`; `media_extension_top` (top extensions by asset count)
- **index timing**: `index_duration_seconds_total` and `index_duration_job_count` — sum of `(finished_at − started_at)` over completed/failed index jobs with both timestamps
- **render storage**: `renders_storage_bytes` — size of completed render output files
- face readiness (`persons_total`, `face_references_total`, `faces_saved`, face-match counters)
- render status counters (`renders_total`, queued/running/completed/failed)

Event renders response includes render jobs for that event (latest first), suitable for profile dashboard listing. Each job may include `planner_prompt` (the user prompt from the content request that produced the plan for that render) when the job was created via `/requests/render` or `/requests/feedback/regenerate`.

## Assets (register-by-path + index)

- `POST /assets`

**Single file (legacy):** `tenant_id`, `event_id`, `media_path`, `media_type` (`image` | `video`), optional **`semantic_prompt`**.

**File or folder (batch):** `tenant_id`, `event_id`, `path` (absolute or relative to backend project root), optional `recursive` (default `true`), optional **`semantic_prompt`** (applies to each registered image in the batch). All files with known image/video extensions under that path are registered and indexed in one request.

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

## Photo curation (images only)

- `GET /events/{event_id}/photos/curation?tenant_id=...` — returns `{ event_id, items }` where each item is an indexed **image** segment: `asset_id`, `segment_id`, `score`, `keep`, `is_duplicate`, `reject_reasons`. Video assets are not listed here.
- `GET /events/{event_id}/assets/{asset_id}/media?tenant_id=...` — serves the original image file for that event asset (images only; tenant/event scoped).
- `GET /events/{event_id}/photos/export-kept?tenant_id=...` — downloads a ZIP of **kept** photos (`keep=true` and not duplicate) under `kept/` in the archive.

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
- `include_media_types` — optional; **empty defaults to `["video"]`**. Image-only values are rejected. Planning and render pools are **video-only**; stills use the photo curation endpoints above.
- `video_orientation`: `landscape` | `portrait` — center crop to 16:9 or 9:16 (reels-style). ASR/OCR are for planner/indexing only; they are not burned into renders.

**Planning behavior:** segment *selection* and culling remain deterministic (scores + semantic retrieval). When **`PLANNER_MODEL_ENABLED`** is on (default), the plan’s **`segment_ids`** order is produced by a **text LLM** (`Qwen2.5-7B-Instruct`) so clips can be grouped by source video and story order instead of only highlight/chronological ordering; the plan action **`set_order`** uses **`preserve_planner`** so the renderer concatenates clips in that order. Invalid model output returns **400** with a clear error. Disable with `PLANNER_MODEL_ENABLED=false` for legacy ordering strategies only.

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
