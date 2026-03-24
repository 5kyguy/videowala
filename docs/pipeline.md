# Indexing pipeline (PoC)

This document describes how **image** and **video** assets are indexed in Videowala today. Implementation lives mainly in:

- [`backend/app/services/indexing.py`](../backend/app/services/indexing.py) — `index_image_asset`, `index_video_asset`, `index_asset` (dispatcher)
- [`backend/app/workers/index_worker.py`](../backend/app/workers/index_worker.py) — queued jobs, serial worker pool by default
- [`backend/app/config.py`](../backend/app/config.py) — `INDEX_WORKERS`, `IMAGE_INDEX_SEMANTIC_CULL_PERCENT`, model IDs

**Rendering** is **not** part of indexing. After video (or any) assets are indexed, users trigger renders via the existing render APIs (see [`docs/api.md`](api.md)).

---

## End-to-end: ingest to persisted insights

Indexing runs **after** `POST /assets` registers an asset. The API creates an **`index_jobs`** row and returns quickly; a background worker runs the appropriate pipeline.

```mermaid
flowchart LR
  client[Client]
  api[FastAPI POST /assets]
  sqlite[(SQLite assets + index_jobs)]
  worker[Index worker pool]
  pipe{media_type}

  client --> api
  api --> sqlite
  api --> worker
  worker --> pipe
  pipe -->|image| imgPipe[index_image_asset]
  pipe -->|video| vidPipe[index_video_asset]
  imgPipe --> sqlite
  vidPipe --> sqlite
```

- **`INDEX_WORKERS`** defaults to **`1`**: at most one index job runs at a time (PoC: avoid overlapping GPU-heavy work across assets).
- Each pipeline stage uses **one model family at a time**, then calls **`release()`** on that service so weights can leave GPU memory before the next stage loads.

---

## Image pipeline

Analysis uses the **original file path** (no video proxy). Stages run in order; after each heavy stage the corresponding service **`release()`** runs (InsightFace stub/real, VLM, PaddleOCR, sentence-transformers).

```mermaid
flowchart TD
  start([index_image_asset])
  del[Delete prior insights for asset]
  proxy[ensure_asset_proxy metadata or manifest]
  face[Face detect plus match to event references]
  relF[face_service.release]
  vlm[VLM caption and tags]
  relV[vlm_service.release]
  gate{VLM tags hit OCR triggers?}
  ocr[PaddleOCR extract]
  skipO[Skip OCR]
  relO[ocr_service.release]
  seg[Build segments plus base cull score]
  ins[Persist VLM face OCR insights]
  emb[Build text plus embed plus pgvector upsert]
  relE[embedding_service.release]
  prompt{semantic_prompt set?}
  rank[apply_photo_semantic_cull_for_event]
  done([Done])

  start --> del --> proxy --> face --> relF --> vlm --> relV --> gate
  gate -->|yes| ocr
  gate -->|no| skipO
  ocr --> relO
  skipO --> relO
  relO --> seg --> ins --> emb --> relE --> prompt
  prompt -->|yes| rank --> done
  prompt -->|no| done
```

**Semantic prompt (optional):** if ingest included **`semantic_prompt`**, after embeddings exist the event’s **image** segments are re-scored with the same logic as photo curation (blend base score with semantic search), using **`IMAGE_INDEX_SEMANTIC_CULL_PERCENT`**. Query-time curation via the photo API still works when no ingest prompt is used.

---

## Video pipeline

Analysis uses the **proxy MP4** when available (see `ensure_asset_proxy`). Order: faces → **ASR** (Whisper on extracted audio) → VLM (multi-frame) → gated OCR → embedding. Each stage ends with **`release()`** on the service that held the model for that stage.

```mermaid
flowchart TD
  start([index_video_asset])
  del[Delete prior insights for asset]
  proxy[ensure_asset_proxy plus manifest]
  path[analysis path equals proxy file]
  face[Face detect plus match]
  relF[face_service.release]
  asr[ASR transcribe proxy]
  relA[asr_service.release]
  vlm[VLM caption and tags from frames]
  relV[vlm_service.release]
  gate{VLM tags hit OCR triggers?}
  ocr[PaddleOCR on sampled frames]
  skipO[Skip OCR]
  relO[ocr_service.release]
  seg[Build segments plus base cull score]
  ins[Persist insights including ASR]
  emb[Build text plus embed plus pgvector upsert]
  relE[embedding_service.release]
  done([Done no auto render])

  start --> del --> proxy --> path --> face --> relF --> asr --> relA --> vlm --> relV --> gate
  gate -->|yes| ocr
  gate -->|no| skipO
  ocr --> relO
  skipO --> relO
  relO --> seg --> ins --> emb --> relE --> done
```

**Rendering:** indexing only writes insights, segments, and vectors. The user (or UI) calls the **render** endpoints separately when a video output is needed.

---

## Worker versus render pool

Index jobs and render jobs use **different** thread pools so a long index queue does not block **`submit_render_job`**.

```mermaid
flowchart LR
  idxPool[Index ThreadPoolExecutor size INDEX_WORKERS]
  renPool[Render ThreadPoolExecutor max_workers 1]

  idxPool --> indexAsset[index_asset]
  renPool --> executeRender[execute_render_job]
```

---

## Configuration knobs (summary)

| Setting | Role |
| ------- | ---- |
| `INDEX_WORKERS` | Parallelism of **index jobs** across assets (default `1`). |
| `INDEXING_PROGRESS` | tqdm on **batch** folder ingest file list (stderr TTY behavior). |
| `IMAGE_INDEX_SEMANTIC_CULL_PERCENT` | Keep fraction when **`semantic_prompt`** is used at image ingest. |
| `OCR_TRIGGER_TAGS` | VLM tags that **enable** OCR after captioning. |
| `VLM_MODEL_ID`, `EMBEDDING_MODEL_ID`, etc. | Model selection (see [`docs/running.md`](running.md)). |

For PoC defaults and AI behavior expectations, see [`.cursor/rules/poc-first.mdc`](../.cursor/rules/poc-first.mdc).
