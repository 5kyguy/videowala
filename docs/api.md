# API (current)

**Product context:** root [`README.md`](../README.md).

Base URL: `http://localhost:8000`

All endpoints are **tenant-scoped** by `tenant_id` and (where relevant) `event_id`.

**OCR:** set `OCR_ENGINE=easyocr` (default) for EasyOCR + PyTorch — uses **GPU when `torch.cuda.is_available()`** (NVIDIA CUDA or AMD ROCm builds). Use `OCR_ENGINE=paddle` only if PaddleOCR works on your machine (slow cold start; static inference can fail with `std::exception` on some setups).

**Indexing progress:** when stderr is a TTY, `tqdm` shows a **per-asset** bar (purge → faces → OCR → ASR → VLM) and a **batch** bar over files. Set `INDEXING_PROGRESS=0` to disable, or `INDEXING_PROGRESS=1` to force on (e.g. piped logs).

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
