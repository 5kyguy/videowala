---
name: Stage 2 Index+Render Plan
overview: Add OCR (PaddleOCR), ASR (faster-whisper), and semantic embeddings (pgvector) into the indexing pipeline, then extend planner+rendering so OCR/ASR can drive selection and optionally produce subtitles/overlays in final renders.
todos:
  - id: stage2-pgvector-foundation
    content: Add Postgres+pgvector schema, connection layer, and vector upsert/query APIs scoped by tenant/event.
    status: completed
  - id: stage2-embeddings-service
    content: Implement embedding generation for query and asset text sources; persist vectors to pgvector and metadata to AssetInsight.
    status: completed
  - id: stage2-ocr-indexing
    content: Implement OCR extraction (PaddleOCR) for images and sampled video frames; persist structured ocr_text insights.
    status: completed
  - id: stage2-asr-indexing
    content: Implement audio extraction + faster-whisper transcription; persist structured asr_transcript insights.
    status: completed
  - id: stage2-context-and-search
    content: Extend event context builder and add semantic search endpoint using embeddings + optional reranking signals.
    status: completed
  - id: stage2-planner-actions
    content: Extend planner schema/actions to support subtitles and OCR overlays while remaining backward compatible.
    status: completed
  - id: stage2-render-subtitles
    content: Generate SRT/VTT from ASR and extend ffmpeg pipeline to burn-in or mux subtitles safely.
    status: completed
  - id: stage2-render-ocr-overlays
    content: Select OCR overlay items and apply safe ffmpeg drawtext overlays with escaping and bounds.
    status: completed
  - id: stage2-tests-hardening
    content: Add unit/integration tests for OCR/ASR/embeddings/search and render subtitle/overlay safety + cross-tenant checks.
    status: completed
isProject: false
---

# Stage 2 Plan: OCR + ASR + Semantic Embeddings (Indexing + Rendering)

## Goal

Extend the existing Stage 1 pipeline (ingest → index → plan → render) so:

- Indexing produces OCR text, ASR transcripts, and semantic embeddings persisted as `AssetInsight` records.
- Retrieval/context building can filter/rank assets using embeddings + OCR/ASR signals.
- Rendering can optionally produce subtitle tracks from ASR and text overlays from OCR.

This plan assumes **Postgres + pgvector** for embeddings search (per your selection) and keeps all models local/self-hosted.

## Current Integration Points (Existing)

- **Indexing**: [backend/app/services/indexing.py](/home/skyguy/foss/videowala/backend/app/services/indexing.py)
  - Already writes `AssetInsight` records and has Stage 2 flags in [backend/app/config.py](/home/skyguy/foss/videowala/backend/app/config.py).
- **Insight types**: [backend/app/schemas.py](/home/skyguy/foss/videowala/backend/app/schemas.py)
  - Has `ocr_text`, `asr_transcript`, `semantic_embedding` enums.
- **Planner**: [backend/app/services/planner.py](/home/skyguy/foss/videowala/backend/app/services/planner.py)
  - Builds plan from event context buckets.
- **Rendering**: [backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py)
  - Deterministic ffmpeg pipeline (concat), currently no subtitle/overlay steps.
- **Persistence layer**: SQLite is current, but infra already provides Postgres in [infra/docker-compose.yml](/home/skyguy/foss/videowala/infra/docker-compose.yml).

## Architecture Change: Add Postgres+pgvector (Embeddings)

### Add a Postgres-backed vector index table

- Create a dedicated table (or reuse `asset_insights`) for vectors, with columns:
  - `tenant_id`, `event_id`, `asset_id`
  - `kind` (e.g. `vlm_caption`, `asr_transcript`, `ocr_text`, `multi`) so multiple embeddings per asset are possible
  - `vector` (`pgvector` type)
  - `text_source` (optional: the normalized text used to embed)
  - `created_at`
- Add query APIs for similarity search scoped by `(tenant_id, event_id)`.

### Add embeddings service

- Implement embedding generation for text (ASR transcript + OCR text + VLM caption) and optionally image/text dual encoders later.
- Store vectors in pgvector table; store the raw text and provenance in `AssetInsight` payload.

## Indexing Pipeline (Stage 2)

### 1) OCR extraction

- New OCR module/service:
  - For images: run OCR on the image.
  - For videos: sample frames (e.g., 1 fps or scene-change based) and OCR sampled frames.
- Persist results:
  - `InsightType.ocr_text` payload should include:
    - `items`: list of `{text, bbox, confidence, frame_time?}`
    - `language?`
    - `model`: `PaddleOCR`

### 2) ASR transcription

- New ASR module/service:
  - Extract audio track (ffmpeg) from video.
  - Run `faster-whisper` on extracted audio.
- Persist results:
  - `InsightType.asr_transcript` payload:
    - `segments`: list of `{start, end, text, confidence?}`
    - `language?`
    - `model`: `faster-whisper (large-v3-turbo)`

### 3) Semantic embeddings creation

- Combine text sources into an embedding input (configurable):
  - VLM caption text
  - ASR transcript text (joined)
  - OCR text (joined)
- Generate embeddings and upsert into pgvector table.
- Persist `InsightType.semantic_embedding` payload as an audit stub (vector not stored in SQLite JSON):
  - `{ "kind": "multi", "vector_ref": {"store": "pgvector", "row_id": ...}, "enabled": true }`

### 4) Event context read model upgrades

- Extend [backend/app/services/indexing.py](/home/skyguy/foss/videowala/backend/app/services/indexing.py) context builder to include:
  - `ocr_text` bucket
  - `asr_transcript` bucket
  - `semantic_embedding` bucket (metadata only)

## Planner + Retrieval Upgrades

### 1) Add retrieval endpoint for semantic search

- Add a new endpoint (tenant/event-scoped):
  - `GET /events/{event_id}/search?q=...` → returns ranked `asset_id` list + supporting snippets.
- Implementation:
  - embed query text → pgvector similarity search
  - optionally re-rank using heuristic signals (face matches, duration, OCR density, ASR confidence)

### 2) Use OCR/ASR signals in planning

- Planner should incorporate:
  - person-focused reels: boost segments with matching faces + associated ASR mentions
  - event highlight reels: boost assets with keywords found in OCR/ASR

### 3) Extend `PlannerAction` set for Stage 2 render

- Add actions to [backend/app/schemas.py](/home/skyguy/foss/videowala/backend/app/schemas.py) (keeping Stage 1 actions valid):
  - `render_subtitles`: `{ "source": "asr", "style": "default" }`
  - `render_overlays`: `{ "source": "ocr", "max_items": N, "strategy": "keyframes" }`
  - `set_subtitle_language` (optional)

## Rendering Upgrades (Subtitles + OCR Overlays)

### 1) Subtitle generation

- Convert ASR segments into SRT (or WebVTT) per render job.
- Add an ffmpeg step to burn subtitles (or mux as a separate track depending on target):
  - For MVP: burn-in subtitles for maximum compatibility.

### 2) OCR overlay generation

- Select a small number of OCR texts to overlay:
  - strategy: most confident, most frequent, or time-window key moments
- Render overlays via ffmpeg `drawtext` filters with safe escaping.
- Ensure overlays are bounded (max characters, no untrusted injection) similar to existing path-safety checks.

### 3) Render spec persistence

- Persist subtitle/overlay artifacts paths as part of render spec so restarts are safe.

## Privacy, Tenant Isolation, and Cleanup

- Scope all vector queries and OCR/ASR artifacts by `(tenant_id, event_id)`.
- Clean scratch audio files, sampled frames, and generated subtitle files after render completion.
- Add audit events:
  - `ocr_indexed`, `asr_indexed`, `embedding_indexed`, `semantic_search`, `subtitles_rendered`, `overlays_rendered`.

## Testing

- Unit tests:
  - OCR payload schema stability
  - ASR payload schema stability
  - Embedding upsert/query functions (mock pgvector)
  - Subtitle generation from ASR segments
  - Overlay filter escaping/safety
- Integration tests:
  - index → search → plan → render with subtitles enabled
  - cross-tenant rejection for search and embedding queries

## Incremental Delivery Sequence

```mermaid
flowchart LR
pgvectorSetup[PostgresPgvectorSetup] --> embeddingSvc[EmbeddingService]
embeddingSvc --> ocrSvc[OcrService]
ocrSvc --> asrSvc[AsrService]
asrSvc --> indexingStage2[IndexingStage2WriteInsights]
indexingStage2 --> semanticSearchApi[SemanticSearchApi]
semanticSearchApi --> plannerUpgrade[PlannerUpgrade]
plannerUpgrade --> renderSubtitles[RenderSubtitles]
renderSubtitles --> renderOverlays[RenderOcrOverlays]
renderOverlays --> stage2Tests[Stage2Tests]
```

## Done When

- Indexing produces real `ocr_text`, `asr_transcript`, and embedding entries (pgvector) for an event.
- `/events/{event_id}/search` returns sensible results for natural language queries.
- Render can optionally output MP4 with burned-in subtitles and OCR overlays, driven by planner actions or request flags.
- All additions remain tenant-scoped, auditable, and restart-safe.
