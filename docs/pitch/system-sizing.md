# System Sizing

## Implementation status

The repo supports two runtime modes at the config level, but sizing and operational guidance here assumes **real model mode**:

- **Real mode (GPU preferred)**: `DEV_MODE=false` (or unset) → real model execution (VLM/OCR/ASR/embeddings), which drives the actual hardware needs below.
- **Dev mode (stubbed)**: `DEV_MODE=true` → deterministic stubbed model outputs for tests/CI only; not representative of production sizing.

Rendering relies on `ffmpeg` (CPU).

```mermaid
flowchart LR
  CPU[CPU box] --> API[FastAPI + SQLite]
  API --> FF[ffmpeg render]
  API -. optional .-> GPU[GPU worker\n(if real VLM/face)]
  API -. optional .-> PG[(Postgres + pgvector\n(semantic))]
```

## Sizing Philosophy

Use sizing tiers for the pitch, not false precision. Real production sizing will depend on:

- average event size
- photo and video mix
- target turnaround time
- concurrency
- how much video sampling is performed

## Tier 1: Development Workstation

Good for local development, experiments, and small demos.

- GPU: 6 to 8 GB VRAM
- RAM: 32 GB
- CPU: 8 to 12 modern cores
- Storage: 1 to 2 TB NVMe

Expected use:

- photo indexing
- short-video proxy analysis
- low concurrency
- slower async batch jobs

Notes:

- `SmolVLM2-2.2B-Instruct` makes workstation development realistic on smaller GPUs and is roughly in the ~5 GB VRAM class for inference
- 12 GB remains comfortable, but no longer needs to be treated as the minimum target for image and short-video tests

## Tier 2: Pilot Deployment

Good for a small studio or controlled customer pilot.

- GPU: 16 to 24 GB VRAM
- RAM: 64 to 128 GB
- CPU: 16+ cores
- Scratch storage: 2+ TB NVMe
- Object storage: external S3-compatible store or dedicated local store

Expected use:

- small number of concurrent indexing jobs
- real event uploads
- preview rendering and final rendering
- a few active users per tenant

Why this tier matters:

- it creates enough headroom for `SmolVLM2`, transcription, OCR, and render queues without everything contending on one small workstation

## Tier 3: Early Production

Good for a multi-tenant hosted product with predictable isolation.

- API node: CPU-focused app server
- DB node: dedicated PostgreSQL instance
- Storage node: object storage and backup flows
- GPU worker pool: one or more isolated inference/render workers

Sizing rule:

- scale by concurrent jobs and turnaround expectations, not raw tenant count

## Workload Cost By Component

### Cheap

- metadata extraction with `ffprobe`
- thumbnail generation
- speech-to-text relative to multimodal video inference

### Moderate

- embeddings for photos and sampled frames
- OCR on selected frames
- face detection and clustering for event-sized batches

### Expensive

- multimodal video understanding
- large batch re-processing
- rendering many outputs from the same event at high resolution

## Practical Cost Controls

- always create proxies for video analysis
- sample frames or scene segments instead of exhaustive analysis
- cache embeddings and intermediate insights
- separate indexing queues from rendering queues
- keep user-facing APIs off the GPU worker path

## Useful Pitch Message

This does not require hyperscale infrastructure to get started. A credible pilot can run on one solid GPU box plus standard database and storage services, provided the pipeline is sample-based and asynchronous.

## Privacy Posture

- media remains on infrastructure we control
- scratch media generated during inference and rendering should be cleaned automatically
- tenant isolation applies to storage, metadata, caches, and job artifacts
