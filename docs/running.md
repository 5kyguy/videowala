# Running locally

**What this project is:** root [`README.md`](../README.md). Below: commands only.

## Prereqs

- Python 3.11+ (recommended)
- Node 18+ (for the frontend)
- `ffmpeg` and `ffprobe` on PATH (rendering + metadata)
- **Postgres with `pgvector`** for semantic search and embedding upsert (startup migrates `asset_vectors`; search/embeddings degrade if DB is down)

## Backend

From `backend/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Configure Postgres DSN in `backend/app/config.py` (`settings.pg_dsn`) if not using the default.

Defaults:

- SQLite DB: `./storage/videowala.db` (repo root)
- Scratch: `./tmp/` (repo root)
- Renders: `./storage/<tenant>/<event>/renders/` (repo root)

Health check:

```bash
curl -s http://localhost:8000/health | jq .
```

## Frontend

From `frontend/` (package manager: **Yarn**):

```bash
yarn install
yarn dev
```

Configure backend URL (optional):

- Set `VITE_API_BASE_URL` (see `frontend/.env.example`)

## Demo media paths

The system currently registers **paths** (it does not upload bytes yet). The frontend defaults to ingesting from the repo’s top-level `media/` folder.

Make sure the path you register:

- exists on the backend machine
- is readable by the backend process

## Model behavior

Stub vs real inference is controlled by `settings.stage2_stub_models` in `backend/app/config.py` (tests use stubs).
