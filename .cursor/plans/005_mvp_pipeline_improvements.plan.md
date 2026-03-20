---
name: MVP Pipeline Improvements
overview: Add proxy-first ingest and deterministic culling/ranking so planner and renderer operate on higher-signal candidates faster, while staying local-only and open-weight.
todos:
  - id: proxy-prep-layer
    content: Add proxy/metadata preprocessing stage and persistence before heavy indexing.
    status: completed
  - id: culling-scorer
    content: Implement deterministic culling/ranking signals and store keep/reject decisions.
    status: completed
  - id: planner-ranked-inputs
    content: Update planner to prioritize ranked keep-set candidates and degrade gracefully.
    status: completed
  - id: segment-selection
    content: Introduce segment-level candidate selection and map segments into render actions.
    status: completed
  - id: render-weighted-assembly
    content: Apply score-aware duration allocation and continuity heuristics in renderer.
    status: completed
  - id: async-reliability
    content: Move long indexing/render paths to async worker execution with progress tracking.
    status: completed
isProject: false
---

# MVP Pipeline Upgrades (Proxy + Culling First)

## Current Fit vs Your Reference Pipelines

The current codebase already matches the **local-system PoC flow** in broad strokes:

- ingest/register media
- index with VLM/faces/OCR/ASR/embeddings
- plan from indexed context
- render deterministic preview/output

So yes: it is on a similar path to your referenced diagrams, but it is still missing two key production-like MVP stages:

- explicit **proxy-first preprocessing**
- explicit **culling/quality filtering before planning**

## Proposed Target Flow

```mermaid
flowchart LR
  ingest[IngestPathOrFolder] --> proxy[ProxyAndMetadataBuild]
  proxy --> segment[SceneAndSegmentPrep]
  segment --> score[QualityAndRelevanceScoring]
  score --> cull[CullingKeepSet]
  cull --> index[IndexRichInsights]
  index --> plan[PlannerUsesRankedCandidates]
  plan --> render[RendererPreviewThenFinal]
```

## Phase 1 (Highest ROI): Proxy-First Media Preparation

- Add a preprocess step that generates per-video low-res proxy files and frame-sampling manifests.
- Persist proxy paths + technical metadata (`duration`, `fps`, `resolution`, `audio presence`) in asset-linked records.
- Ensure indexers consume proxies for heavy analysis, while render still uses originals for output.

Primary files to extend:

- [backend/app/services/ingest.py](backend/app/services/ingest.py)
- [backend/app/services/indexing.py](backend/app/services/indexing.py)
- [backend/app/repositories.py](backend/app/repositories.py)
- [backend/app/db.py](backend/app/db.py)
- [backend/app/schemas.py](backend/app/schemas.py)

## Phase 2 (Highest ROI): Deterministic Culling + Ranking

- Add lightweight culling signals per asset/segment:
  - technical quality (blur/exposure/audio present/corruption)
  - duplicate-near-duplicate suppression
  - semantic relevance to request prompt/person focus
- Store a normalized `cull_score` and reject reason flags.
- Planner should select from top-ranked keep set first; fallback to broader pool if sparse.

Primary files to extend:

- [backend/app/services/indexing.py](backend/app/services/indexing.py)
- [backend/app/services/planner.py](backend/app/services/planner.py)
- [backend/app/schemas.py](backend/app/schemas.py)

## Phase 3: Segment-Level Planning (Not Just Asset-Level)

- Move planner selection unit from whole assets to candidate segments (scene ranges / sampled windows).
- Keep deterministic actions but include `segment_ranges` and ordering hints.
- Preserve include/exclude UX by mapping selected segments back to source assets.

Primary files to extend:

- [backend/app/services/planner.py](backend/app/services/planner.py)
- [backend/app/services/rendering.py](backend/app/services/rendering.py)
- [backend/app/repositories.py](backend/app/repositories.py)

## Phase 4: Renderer Quality Improvements for MVP

- Replace equal-time-per-asset split with score-weighted duration allocation.
- Add simple continuity heuristics (avoid back-to-back near-duplicates, preserve chronology by default).
- Keep deterministic ffmpeg pipeline; no API-based model dependency.

Primary files to extend:

- [backend/app/services/rendering.py](backend/app/services/rendering.py)
- [backend/tests/test_rendering_safety.py](backend/tests/test_rendering_safety.py)

## Phase 5: Operational Reliability (Still MVP-safe)

- Add async job execution for indexing/rendering to avoid blocking API calls.
- Add resumable status for long preprocess/index jobs and event-level progress counters.
- Keep single-node local deployment, but separate worker paths for GPU-heavy indexing vs CPU-heavy rendering.

Primary files to extend:

- [backend/app/api/routes.py](backend/app/api/routes.py)
- [backend/app/workers/index_worker.py](backend/app/workers/index_worker.py)

## Open-Weight Model Constraints (Aligned with Your Requirement)

- Keep all inference local/self-hosted.
- No third-party inference APIs.
- Prioritize robust GPU mode path for:
  - VLM caption/tag extraction
  - embeddings
  - OCR and ASR
  - optional face matching

## MVP Acceptance Criteria

- End-to-end ingest-to-render works with proxy-first indexing enabled.
- Planner chooses from culled/ranked candidate set and produces better first cuts than current baseline.
- Render output remains deterministic and reproducible.
- Throughput improves on long wedding videos due to proxy/sample-based analysis.
- Existing include/exclude regenerate workflow remains intact.
