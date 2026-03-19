# Backend MVP

**What VideoWala is for:** see the repo root [`README.md`](../README.md).

FastAPI backend for Stage 1 of the MVP:

- media ingestion and event registration
- VLM + face indexing context
- planner LLM JSON tool-call plan generation
- deterministic render job execution
- include/exclude and regenerate loop

Stage 2 features (OCR, ASR, semantic embeddings) are intentionally deferred but schema hooks are preserved.

## Configuration

- Environment: copy `backend/.env.example` to `backend/.env` and adjust. The backend loads **only** `backend/.env` (not a repo-root file); shell exports still override.

## Persistence

- Runtime state is persisted in SQLite at `storage/videowala.db` by default.
- Schema migrations are applied automatically on startup.
- Configure DB path via `app/config.py` (`settings.db_path`).

## Face APIs

- `POST /persons`
- `GET /persons`
- `POST /persons/{person_id}/references`
- `POST /events/{event_id}/faces/reindex`
- `GET /events/{event_id}/faces/matches`
