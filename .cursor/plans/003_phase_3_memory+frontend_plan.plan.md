---
name: Phase 3 Memory+Frontend Plan
overview: Align the current MVP backend with persistent storage and real face-recognition APIs, then add a minimal frontend that integrates with existing FastAPI endpoints for event creation, ingest, context viewing, planning, rendering, and regenerate feedback.
todos:
  - id: backend-persistence
    content: Design and implement persistent data layer with migrations and repository interfaces replacing InMemoryStore.
    status: completed
  - id: backend-memory-readmodel
    content: Move event context/planner memory assembly to DB-backed read models and queries.
    status: completed
  - id: backend-face-service
    content: Implement InsightFace detection/embedding/matching service and persist face insights.
    status: completed
  - id: backend-face-apis
    content: Add person enrollment and face index/match endpoints with tenant isolation + audit.
    status: completed
  - id: backend-hardening-tests
    content: Add integration/privacy tests for persistence, face APIs, and cross-tenant boundaries.
    status: completed
  - id: frontend-scaffold
    content: Create minimal frontend app with typed API client and environment configuration.
    status: completed
  - id: frontend-core-flow
    content: Implement event, ingest, context, plan/render, and feedback/regenerate screens.
    status: completed
  - id: frontend-qa
    content: Add basic frontend tests and polish error/loading UX for demos.
    status: completed
isProject: false
---

# Phase 3 Plan: Persistent Backend + Simple Frontend

## Current State Snapshot

- Backend APIs already exist in [backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py) for `events`, `assets`, context, plan, render, and regenerate.
- Runtime state is currently non-persistent (`InMemoryStore`) in [backend/app/store.py](/home/skyguy/foss/videowala/backend/app/store.py).
- Indexing and face outputs are stubs in [backend/app/services/indexing.py](/home/skyguy/foss/videowala/backend/app/services/indexing.py), even though model names are present in payload metadata.
- Infra already provisions `postgres`, `redis`, and `minio` in [infra/docker-compose.yml](/home/skyguy/foss/videowala/infra/docker-compose.yml), so the persistence path is already scaffolded.

## Objective A: Backend Persistence + Face Recognition APIs

### 1) Replace In-Memory Store With Persistent Data Layer

- Introduce database models and migrations for Event, Asset, AssetInsight, PlannerPlan, RenderJob, and AuditLog aligned with [docs/data-model.md](/home/skyguy/foss/videowala/docs/data-model.md).
- Add repository/data-access layer and keep service method contracts stable to minimize API churn.
- Persist render specs/plans so `/requests/render` and `/requests/feedback/regenerate` are restart-safe.

### 2) Add Persistent "Memory" for Retrieval and Planner Context

- Implement event memory read model composed from persisted `AssetInsight` records (caption/tags/faces now, OCR/ASR/embedding later).
- Add query utilities for:
  - recent insights by event
  - insights by type
  - optional person-focused filters
- Ensure planner context builder reads from DB, not transient process memory.

### 3) Implement Real Face Pipeline (Detection + Embeddings + Matching)

- Add a face service module that wraps InsightFace for:
  - face detection on image/video frame samples
  - embedding extraction
  - tenant/event-scoped matching against enrolled references
- Persist face artifacts into structured insight payloads (`face_detections`, `face_matches`) using existing schema enums in [backend/app/schemas.py](/home/skyguy/foss/videowala/backend/app/schemas.py).

### 4) Expose Face Management and Recognition APIs

- Add endpoints for person enrollment and reference management (tenant/event scoped):
  - create/list person
  - upload person reference image(s)
  - trigger/rebuild event face index
  - fetch face matches by person/event
- Keep existing request/plan/render endpoints backward compatible.

### 5) Privacy + Operational Hardening

- Add strict tenant scoping checks to all new face/person APIs.
- Add audit entries for person enrollment, recognition runs, and face-match retrieval.
- Ensure worker scratch cleanup runs after indexing/rendering paths.

### 6) Validation and Tests

- Add integration tests for persistence boundaries (restart-safe behavior).
- Add tests for face enrollment/matching APIs and cross-tenant rejection.
- Add contract tests for planner context after moving to DB-backed memory.

## Objective B: Simple Frontend for Current Backend APIs

### 1) Create Minimal Frontend App Shell

- Add a lightweight React/Next.js app in a new `frontend/` folder.
- Keep auth mocked/simple for now (explicit tenant_id input) to match backend MVP.

### 2) Implement Core Screens (MVP Flow)

- Event screen: create event and view event list/details.
- Asset ingest screen: register media path + media type.
- Context screen: fetch and display grouped event context (`vlm_caption`, `vlm_tags`, `face_matches`).
- Plan/Render screen: submit content request and show returned plan/render job.
- Feedback/regenerate panel: include/exclude asset IDs and rerun.

### 3) API Integration Layer

- Add typed API client wrappers for existing endpoints in [backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py).
- Centralize request/response schemas in frontend types to reduce drift.
- Add clear loading/error states for each action.

### 4) Frontend UX Guardrails

- Validate required fields before API calls.
- Surface 403/404/500 backend errors with actionable text.
- Add simple JSON/insight viewers so debugging and demos are straightforward.

### 5) Frontend Testing

- Add basic component and API-client tests for the happy path and common failures.

## Delivery Sequence

```mermaid
flowchart LR
phaseA1[PersistenceLayer] --> phaseA2[DbBackedMemory]
phaseA2 --> phaseA3[FaceServiceAndApis]
phaseA3 --> phaseA4[BackendTests]
phaseA4 --> phaseB1[FrontendScaffold]
phaseB1 --> phaseB2[ApiIntegrationScreens]
phaseB2 --> phaseB3[FrontendTestsAndPolish]
```

## Definition of Done

- Backend state survives process restarts and planner/render flows still work.
- Face enrollment + matching APIs are available and tenant-scoped.
- Frontend can drive end-to-end flow: create event -> ingest -> inspect context -> plan/render -> regenerate.
- Existing MVP routes remain compatible for current scripts/tests.
