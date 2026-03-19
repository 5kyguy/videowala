# Model Stack

## Implementation status

The backend supports two runtime modes, but the intended path for this project is **real model mode**:

- **Real mode (GPU preferred)**: set `DEV_MODE=false` (or omit `DEV_MODE`)
  - attempts real integrations via `transformers` (VLM), `sentence-transformers` (embeddings), OCR, and `faster-whisper`
  - falls back gracefully when a dependency/model/media cannot be loaded
- **Dev mode (stubbed)**: set `DEV_MODE=true` in `backend/.env`
  - VLM/OCR/ASR/embeddings return deterministic stub outputs (fast + test-friendly, suitable for CI)

The code references these model IDs/names today (configurable via `backend/.env`):

- **VLM**: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (via `VLM_MODEL_ID`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (via `EMBEDDING_MODEL_ID`)
- **Faces**: `insightface` (real inference is gated by `enable_real_face_recognition`)
- **OCR**: `PaddleOCR` (images only; video frame sampling not implemented yet)
- **ASR**: `faster-whisper (large-v3-turbo)` (CPU default in code path)

```mermaid
flowchart TB
  ASSET[Asset media_path] --> IDX[Indexing]
  IDX --> CAP[Caption/tags]
  IDX --> FACE[Faces]
  IDX --> OCR[OCR]
  IDX --> ASR[ASR]
  CAP --> CTX[Event context]
  FACE --> CTX
  OCR --> CTX
  ASR --> CTX
  CTX --> PLAN[Planner]
  PLAN --> RENDER[Renderer (ffmpeg)]
```

## Recommended MVP Stack

Use a compositional stack rather than a single multimodal dependency.

## Primary Choices

### Multimodal Understanding

Primary: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

Why:

- open-weight and practical to self-host
- handles both images and videos
- has a much smaller GPU footprint than larger VLMs
- is sufficient for the MVP goal of indexing, captioning, and request-guided media selection

Lightweight Experiment Path:

- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` for lower-resource experiments

Evaluation Candidate:

- `Molmo2-8B` only if later benchmarks justify the extra hardware cost

### Semantic Embeddings

Primary: `SigLIP2`

Why:

- strong open image-text retrieval performance
- multilingual support is useful for broader event contexts
- a separate embedding model keeps retrieval cheaper than routing all search through the main VLM

Suggested approach:

- use embeddings for coarse recall
- use the main VLM and ranking logic for re-ranking and explanation

### Face Detection And Recognition

Primary: `InsightFace`

Why:

- practical local deployment
- proven tooling for detection, embeddings, and clustering
- suitable for tenant-scoped face libraries and event-specific recognition workflows

Recommended product stance:

- face recognition should be opt-in
- recognition should be limited to tenant-provided references and tenant data

### OCR

Primary: `PaddleOCR`

Why:

- mature open-source OCR stack
- broad language support
- usable on CPU or GPU depending throughput needs

OCR can improve:

- venue signage extraction
- invitation and banner text
- scoreboards, stage displays, and name cards

### Speech-To-Text

Primary: `faster-whisper` using `large-v3-turbo`

Why:

- strong local ASR performance
- practical VRAM usage relative to multimodal models
- useful for indexing speeches, announcements, vows, and performances

## Why Not One Giant Model

Trying to do retrieval, face handling, OCR, and ASR through one multimodal model would be slower, more expensive, and less reliable. The right tradeoff is specialization:

- VLM for understanding
- embeddings for search
- task-specific tools for faces, OCR, and audio

## Deployment Notes

- reserve the main GPU budget for the multimodal model
- keep OCR and ASR independently scalable
- use async workers so indexing and rendering do not compete with user-facing APIs

## Selection Summary

- first practical choice: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- first lightweight experiment path: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- first higher-end evaluation path: `Molmo2-8B`
- retrieval backbone: `SigLIP2`
- face stack: `InsightFace`
- OCR: `PaddleOCR`
- ASR: `faster-whisper`

## Privacy Posture

- all models are intended to run on infrastructure we control
- media is never sent to third-party model APIs
- model outputs should be stored tenant-scoped and auditable
