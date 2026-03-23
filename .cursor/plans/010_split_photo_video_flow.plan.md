---
name: Split Photo Video Flow
overview: "Refocus the app into two product tracks: photo curation for images and video rendering for videos only. The plan keeps the current indexing foundation, removes mixed-media render behavior, and adds a curated photo gallery plus export path."
todos:
  - id: video-only-render
    content: Constrain render and regenerate flows to video assets only across frontend, API, planner fallback, and renderer validation.
    status: completed
  - id: photo-curation-api
    content: Add photo-curation read models or endpoints that expose keep/reject/duplicate/score data plus safe image access.
    status: completed
  - id: photo-gallery-ui
    content: Build the curated photo gallery UI with kept vs rejected groupings and photographer-friendly browsing.
    status: completed
  - id: photo-export
    content: Implement export/download of the curated kept-photo set.
    status: completed
  - id: tests-split-workflow
    content: Add backend and frontend tests that lock in the photo/video workflow split.
    status: completed
isProject: false
---

# Split Photo And Video Tracks

## Goal

Turn the current mixed-media pipeline into two clear workflows:

- Photos: automatically cull, rank, and present the best images to reduce photographer review time.
- Videos: index and analyze videos, then render movies/reels from video assets only.

## What Exists Today

The current system already has the foundations, but they are mixed together:

- The planner and render request schema already support `include_media_types`, but the frontend always sends an empty list, so renders are not constrained to video. See [frontend/src/App.tsx](/home/skyguy/foss/videowala/frontend/src/App.tsx), [frontend/src/api.ts](/home/skyguy/foss/videowala/frontend/src/api.ts), and [backend/app/schemas.py](/home/skyguy/foss/videowala/backend/app/schemas.py).
- The planner scores image and video segments together, and its semantic fallback can still pull any asset type. See [backend/app/services/planner.py](/home/skyguy/foss/videowala/backend/app/services/planner.py) and [backend/app/services/search.py](/home/skyguy/foss/videowala/backend/app/services/search.py).
- Indexing already computes useful curation signals such as segment scores, duplicate flags, reject reasons, VLM tags, OCR, ASR, and cull metrics, but the UI does not surface them as a photo-review product. See [backend/app/services/indexing.py](/home/skyguy/foss/videowala/backend/app/services/indexing.py), [backend/app/repositories.py](/home/skyguy/foss/videowala/backend/app/repositories.py), and [backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py).
- Rendering still accepts image inputs and converts them into MP4 clips, which is the behavior this pivot should remove. See [backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py).

## Proposed Product Split

### Video Track

- Treat render as a video-only feature.
- Enforce video-only selection in both API and planner so prompts cannot silently fall back to images.
- Keep prompt-driven planning, clip selection, orientation, and render output, but only across video assets.
- Tighten semantic fallback so it cannot reintroduce images into a video render plan.

### Photo Track

- Treat images as a curation product, not render inputs.
- Surface kept photos, rejected photos, duplicates, and low-quality takes in the UI using existing indexing and segment signals.
- Add a first-class curated gallery for kept photos, plus export/download of the kept set.
- Preserve room for later additions such as manual overrides, albums, and stronger quality heuristics.

## Implementation Plan

1. Enforce video-only render semantics.

- Update request handling in [backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py) so render and regenerate requests default to or require `include_media_types=["video"]`.
- Update planner behavior in [backend/app/services/planner.py](/home/skyguy/foss/videowala/backend/app/services/planner.py) so semantic fallback and ranked asset selection remain constrained to videos for render flows.
- Add validation in [backend/app/services/rendering.py](/home/skyguy/foss/videowala/backend/app/services/rendering.py) so image assets are rejected before render job creation instead of being converted into clips.

2. Separate photo curation from render planning.

- Add dedicated backend read models or endpoints for photo curation, using existing cull and segment data from [backend/app/repositories.py](/home/skyguy/foss/videowala/backend/app/repositories.py) and [backend/app/services/indexing.py](/home/skyguy/foss/videowala/backend/app/services/indexing.py).
- Expose enough data for the frontend to show: asset id, media path or safe thumbnail URL, keep/reject status, duplicate flag, reject reasons, and scores.
- Decide whether the curation view is driven primarily by `asset_segments` or by a photo-level aggregation built from those segments; favor a photo-level view so photographers review photos, not synthetic time slices.

3. Build the curated photo gallery.

- Extend [frontend/src/types.ts](/home/skyguy/foss/videowala/frontend/src/types.ts) and [frontend/src/api.ts](/home/skyguy/foss/videowala/frontend/src/api.ts) with photo-curation response types and API calls.
- Add a photo-review section to [frontend/src/App.tsx](/home/skyguy/foss/videowala/frontend/src/App.tsx) that separates kept photos from rejected or duplicate photos.
- Reuse the existing image-serving approach used for face references, or add a safe asset thumbnail route in [backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py).
- Add focused styling in [frontend/src/styles.css](/home/skyguy/foss/videowala/frontend/src/styles.css) for a review gallery that feels like a photographer workflow, not debug output.

4. Add kept-photo export.

- Implement an export path for the curated keepers, likely as a zip of selected original images or a generated manifest plus downloadable package.
- Add a backend export endpoint and job shape if packaging may take time.
- Add a frontend export action near the curated gallery so the output is directly usable by photographers.

5. Realign the UI around the two-track model.

- Make the dashboard clearly separate photo curation from video rendering in [frontend/src/App.tsx](/home/skyguy/foss/videowala/frontend/src/App.tsx).
- Remove or hide image-based assumptions from the render flow.
- Keep video orientation and prompt controls inside the video workflow only.

6. Strengthen tests around the split.

- Add planner and route tests to confirm video render requests cannot include images and that semantic fallback stays video-only.
- Add tests for photo-curation aggregation and export output.
- Add frontend tests for the new gallery and the render/photo workflow separation.

## Key Design Notes

- This pivot does not require a full storage rewrite. The existing `media_type` field and indexing outputs are sufficient for a clean product split if the API and UI are reorganized carefully.
- The current photo signals are good enough for a first version, but the gallery should be built around photographer outcomes: kept, rejected, duplicate, weak take. It should not expose raw planner internals unless needed.
- Render quality improvements like person-aware crop can stay in scope for the video track later, but this pivot should first stop mixed-media rendering and establish the new product boundary clearly.
