---
name: poc_profile_dashboard_upgrade
overview: Upgrade the MVP frontend into a profile-driven PoC app by replacing the current single-page controls with a tenant profile switcher, event workspace, and accurate per-event status cards powered by new backend summary endpoints.
todos:
  - id: backend-summary-endpoints
    content: Implement backend event summary + event renders endpoints with tenant checks and repository support.
    status: completed
  - id: backend-db-updates
    content: Add required DB/repository updates and migrations to support accurate PoC event dashboard summaries.
    status: completed
  - id: frontend-api-contract
    content: Extend frontend types and API client for summary and render listing responses.
    status: completed
  - id: frontend-poc-replacement
    content: Replace App.tsx with profile-based PoC dashboard and update styling for usable event/status workflow.
    status: completed
  - id: frontend-create-event-flow
    content: Add a first-class create-event flow in the PoC dashboard and auto-select newly created events.
    status: completed
  - id: tests-docs
    content: Add/update backend+frontend tests and API docs for new endpoints and PoC flow.
    status: completed
isProject: false
---

# PoC Profile Dashboard Upgrade

## Goals

- Replace the current MVP operator-style frontend with a user-facing PoC experience centered around `tenant_id` profiles.
- Let users switch profiles instantly and browse only their events.
- Include a clear in-app create-event flow so a user can create and enter a new event from the same workspace.
- Show clear event health/status signals: media present, faces saved, and renders list/status.
- Keep backend in GPU mode compatible (no DEV_MODE assumptions) and surface real backend states.

## Architecture Changes

- Add backend read endpoints that return event dashboard summaries and render lists per tenant/event.
- Refactor frontend API layer to consume new summary endpoints.
- Replace `App.tsx` with a PoC layout:
  - Profile switcher (tenant selector + add/select)
  - Event list panel
  - Event details and health indicators
  - Render jobs panel

```mermaid
flowchart LR
    profileSelector[ProfileSelector] --> tenantEventsApi[GET /events?tenant_id]
    eventSelection[SelectedEvent] --> eventSummaryApi[GET /events/{event_id}/summary]
    eventSelection --> eventRendersApi[GET /events/{event_id}/renders]
    eventSummaryApi --> eventHealthCards[Media Faces InsightsStatus]
    eventRendersApi --> renderList[RenderJobsList]
```

## Backend Plan

- In [/home/skyguy/foss/videowala/backend/app/repositories.py](/home/skyguy/foss/videowala/backend/app/repositories.py):
  - Add `RenderRepository.list_for_event(event_id)` with created-time ordering.
  - Add small aggregation helpers (or route-local aggregation) for:
    - total assets for event
    - presence of face references for event
    - presence/count of face match insights for event
- In [/home/skyguy/foss/videowala/backend/app/api/routes.py](/home/skyguy/foss/videowala/backend/app/api/routes.py):
  - Add `GET /events/{event_id}/summary?tenant_id=...` returning event dashboard status payload.
  - Add `GET /events/{event_id}/renders?tenant_id=...` returning render jobs for that event.
  - Reuse existing tenant scope checks.
- In [/home/skyguy/foss/videowala/backend/app/schemas.py](/home/skyguy/foss/videowala/backend/app/schemas.py) (if needed):
  - Add response models/types for summary and render list for stronger contract clarity.
- In [/home/skyguy/foss/videowala/backend/app/db.py](/home/skyguy/foss/videowala/backend/app/db.py):
  - Add/adjust DB migration(s) if new persisted fields/indexes are required for performant and accurate dashboard stats.
- In [/home/skyguy/foss/videowala/docs/api.md](/home/skyguy/foss/videowala/docs/api.md):
  - Document the two new PoC dashboard endpoints.

## Frontend Plan

- In [/home/skyguy/foss/videowala/frontend/src/types.ts](/home/skyguy/foss/videowala/frontend/src/types.ts):
  - Add `EventSummary` and `RenderJobListItem` types aligned with backend response.
- In [/home/skyguy/foss/videowala/frontend/src/api.ts](/home/skyguy/foss/videowala/frontend/src/api.ts):
  - Add client methods:
    - `getEventSummary(tenantId, eventId)`
    - `listEventRenders(tenantId, eventId)`
- Replace [/home/skyguy/foss/videowala/frontend/src/App.tsx](/home/skyguy/foss/videowala/frontend/src/App.tsx):
  - Build PoC-first screen with:
    - profile select/create quick switch
    - create-event form (title/type + optional metadata), inline in profile workspace
    - event list and selected event state
    - status cards (media presence/count, faces saved indicator, insights snapshot)
    - render list (status/progress, created time, open video for completed jobs)
  - Auto-select and load dashboard data for a newly created event.
  - Keep simple/no-auth flow as requested.
- In [/home/skyguy/foss/videowala/frontend/src/styles.css](/home/skyguy/foss/videowala/frontend/src/styles.css):
  - Move from raw MVP form styling to clean dashboard layout (cards, grid, status chips).

## Testing and Validation

- Backend:
  - Add/extend API tests for new endpoints and tenant scoping behavior.
- Frontend:
  - Update smoke tests to assert PoC shell renders and profile/event flow mounts.
  - Add API client tests for new methods and error handling.
- Manual validation:
  - Switch between at least two tenant profiles.
  - Verify event list isolation per tenant.
  - Confirm status cards change after ingest/faces/render pipeline activity.

## Rollout Notes

- Preserve existing core workflows by keeping API compatibility for current ingest/plan/render endpoints.
- Focus on functionality and correctness over auth/landing polish, matching PoC scope.
