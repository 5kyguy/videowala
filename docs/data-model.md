# Data model (current SQLite schema)

**Product context:** root [`README.md`](../README.md). This page is the **implemented** SQLite layer (`backend/app/db.py`).

## High-level ER diagram

```mermaid
erDiagram
  EVENTS ||--o{ ASSETS : has
  EVENTS ||--o{ ASSET_INSIGHTS : derives
  ASSETS ||--o{ ASSET_INSIGHTS : generates
  EVENTS ||--o{ PLANNER_PLANS : plans
  EVENTS ||--o{ RENDER_JOBS : renders
  RENDER_JOBS ||--|| RENDER_SPECS : config
  EVENTS ||--o{ PERSONS : includes
  PERSONS ||--o{ PERSON_REFERENCES : has
  EVENTS ||--o{ PERSON_REFERENCES : scoped_to
  EVENTS ||--o{ AUDIT_LOGS : audit

  EVENTS {
    text id PK
    text tenant_id
    text title
    text event_type
    text venue
    text date
    text created_at
  }

  ASSETS {
    text id PK
    text tenant_id
    text event_id FK
    text media_path
    text media_type
    text created_at
  }

  ASSET_INSIGHTS {
    text id PK
    text tenant_id
    text event_id FK
    text asset_id FK
    text insight_type
    text payload_json
    real confidence
    text created_at
  }

  PLANNER_PLANS {
    text id PK
    text tenant_id
    text event_id FK
    text output_type
    text rationale
    text actions_json
    text created_at
  }

  RENDER_JOBS {
    text id PK
    text tenant_id
    text event_id FK
    text plan_id
    text status
    text output_path
    text created_at
  }

  RENDER_SPECS {
    text render_job_id PK,FK
    text input_files_json
    int duration_seconds
    text scratch_dir
    int subtitles_enabled
    int overlays_enabled
  }

  AUDIT_LOGS {
    text id PK
    text tenant_id
    text event_id
    text action
    text payload_json
    text created_at
  }

  PERSONS {
    text id PK
    text tenant_id
    text event_id FK
    text display_name
    text created_at
  }

  PERSON_REFERENCES {
    text id PK
    text person_id FK
    text tenant_id
    text event_id FK
    text image_path
    text embedding_json
    text created_at
  }
```

## Notes and invariants

- **Tenant scope**: Most rows carry `tenant_id` for application-level checks; SQLite foreign keys enforce event/asset/person referential integrity.
- **Media**: `assets.media_path` is a path on disk; the backend does not currently ingest bytes into object storage.
- **Insights**: `asset_insights.payload_json` stores structured outputs for:
  - stage1: `vlm_caption`, `vlm_tags`, `face_detections`, `face_matches`
  - stage2 (optional): `ocr_text`, `asr_transcript`, `semantic_embedding`
- **Plans**: `planner_plans.actions_json` stores the ordered list of planner actions.
- **Renders**:
  - `render_jobs` is the user-visible job record.
  - `render_specs` stores deterministic inputs + feature flags for restart-safety.

## Out of scope (not persisted yet)

- Users/accounts (beyond `tenant_id` as a string).
- Upload sessions and object storage metadata.
- Fine-grained segments/timeline primitives (only “asset_ids → simple preview render” today).
