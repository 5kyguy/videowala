---
name: Fix video planning quality
overview: Enforce strict target-duration rendering and improve continuity so generated videos avoid repeated micro-cuts and fragmented storytelling. The plan focuses on planner + renderer guardrails in GPU mode without fallback shortcuts.
todos:
  - id: renderer-duration-invariants
    content: Refactor renderer allocation/export path to guarantee strict target duration and enforce final duration cap.
    status: completed
  - id: segment-boundary-clamp
    content: Clamp per-clip extraction to indexed segment bounds and handle invalid windows robustly.
    status: completed
  - id: duration-aware-selection
    content: Introduce duration-aware segment count limits in planner/config to avoid micro-cut-heavy timelines.
    status: completed
  - id: continuity-merge-pass
    content: Add same-asset contiguous merge logic and tighten sequencing continuity behavior.
    status: completed
  - id: tests-and-gpu-validation
    content: Add regression tests and run GPU-mode plan/render validation against a 60s scenario.
    status: completed
isProject: false
---

# Fix Duration and Continuity in Planning + Rendering

## Goal

Produce outputs that reliably match requested duration (e.g., 60s) while avoiding repeated 1-2s fragments of the same scene and improving narrative continuity.

## What Is Failing Today

- `target_duration_seconds` is set in the plan but rendering can overshoot because per-clip allocation enforces a 1s minimum for every selected clip.
- Renderer uses `start_s` but does not strictly cap each clip to indexed `end_s` boundaries.
- Planner can pass too many segments for short targets, causing dense hard cuts.
- Sequencing may interleave sources, fragmenting moments that should play contiguously.

## Implementation Plan

### 1) Enforce hard duration correctness in renderer

- Update `[/home/skyguy/foss/videowala/backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py)`:
  - Make `_allocate_clip_seconds()` mathematically guarantee `sum(allocations) == target_duration_seconds`.
  - If selected clips exceed feasible count for the target budget, drop lowest-value clips before allocation.
  - Add a final output cap at concat/export stage so produced MP4 cannot exceed `duration_seconds`.
- Acceptance criterion: 60s request yields ~60s output (within container rounding tolerance).

### 2) Respect indexed segment boundaries

- In `[/home/skyguy/foss/videowala/backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py)`:
  - Clamp per-clip render duration to `(end_s - start_s)` so extraction never spills beyond indexed segment windows.
  - Skip zero/invalid windows with structured logging.
- Acceptance criterion: each rendered clip stays inside planned segment time span.

### 3) Reduce micro-cuts by limiting clip count for short targets

- In `[/home/skyguy/foss/videowala/backend/app/services/planner.py](/home/skyguy/foss/videowala/backend/app/services/planner.py)`:
  - Derive a duration-aware max segment count (e.g., based on target duration and minimum hold time).
  - Keep top-ranked segments only up to this budget rather than feeding all candidates to renderer.
- In `[/home/skyguy/foss/videowala/backend/app/config.py](/home/skyguy/foss/videowala/backend/app/config.py)`:
  - Add explicit knobs for minimum hold duration and duration-aware cap behavior.
- Acceptance criterion: a 60s edit no longer contains dozens of 1s clips.

### 4) Improve continuity by grouping same-source adjacent windows

- In `[/home/skyguy/foss/videowala/backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py)` (or planner pre-processing if cleaner):
  - Add a merge pass that combines contiguous/near-contiguous segments from the same `asset_id` into longer spans before final allocation.
- In `[/home/skyguy/foss/videowala/backend/app/services/plan_sequencer.py](/home/skyguy/foss/videowala/backend/app/services/plan_sequencer.py)`:
  - Strengthen sequencing constraints toward grouped continuity unless prompt explicitly asks for rapid montage.
- Acceptance criterion: repeated camel-riding snippets become one coherent longer block when continuity is preferred.

### 5) Add guardrail tests + quality checks

- Add/extend backend tests for:
  - exact duration budget behavior,
  - no overshoot when clip count > duration seconds,
  - same-asset contiguous merge behavior,
  - no out-of-bound clip extraction relative to `start_s/end_s`.
- Validate with real GPU-mode run (`/requests/plan` + `/requests/render`) and inspect both timeline stats and resulting MP4.

## Validation Checklist

- 60s request returns MP4 close to 60s, never 150m.
- Clip count aligns with continuity objective (fewer, longer shots).
- Same action sequence (e.g., camel ride) appears as contiguous block when semantically appropriate.
- No duplicated planned windows and no segment spill beyond indexed bounds.
- Logs include useful diagnostics when candidates are dropped/merged for duration compliance.
