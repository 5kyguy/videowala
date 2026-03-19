# Running locally

**What this project is:** root [`README.md`](../README.md). Below: commands only.

## Prereqs

- Python 3.11+ (recommended)
- Node 18+ (for the frontend)
- `ffmpeg` and `ffprobe` on PATH (rendering + metadata)
- **GPU with recent NVIDIA CUDA or AMD ROCm build** for real model inference (VLM, OCR, ASR, embeddings)
- **Postgres with `pgvector`** for semantic search and embedding upsert (startup migrates `asset_vectors`; search/embeddings degrade if DB is down)

## Backend

From `backend/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Configure the backend via `backend/.env` (copy from `backend/.env.example`). There is no repo-root `.env` for the API.

Key settings (see `backend/app/config.py`):

- **Storage paths** (anchored at repo root):
  - SQLite DB: `./storage/videowala.db`
  - Scratch: `./tmp/`
  - Renders: `./storage/<tenant>/<event>/renders/`
- **Database**:
  - `PG_DSN` → Postgres DSN for pgvector-backed semantic search
- **Models (real mode, GPU preferred)**:
  - `VLM_MODEL_ID` → multimodal model (`HuggingFaceTB/SmolVLM2-2.2B-Instruct` by default)
  - `EMBEDDING_MODEL_ID` → text embedding model (`sentence-transformers/all-MiniLM-L6-v2` by default)
  - `OCR_ENGINE` → `"easyocr"` (default; uses GPU when available) or `"paddle"` (PaddleOCR with EasyOCR fallback)
- **Indexing UX**:
  - `INDEXING_PROGRESS` → `0` / `1` to force-disable/force-enable tqdm progress bars

Health check:

```bash
curl -s http://localhost:8000/health | jq .
```

## Frontend

From `frontend/` (use your preferred Node package manager; example with **npm**):

```bash
npm install
npm run dev
```

Configure backend URL (optional):

- Set `VITE_API_BASE_URL` (see `frontend/.env.example`)

## Demo media paths

The system currently registers **paths** (it does not upload bytes yet). The frontend defaults to ingesting from the repo’s top-level `media/` folder.

Make sure the path you register:

- exists on the backend machine
- is readable by the backend process

## Model behavior

Runtime behavior is controlled by `DEV_MODE` in `backend/.env` (loaded automatically on import):

- **Real mode (recommended for this project)**: omit `DEV_MODE` or set `DEV_MODE=false`
  - VLM, OCR, ASR, embeddings, and face pipelines run with real models (GPU preferred; some paths fall back to CPU if needed).
- **Stub mode**: `DEV_MODE=true`
  - Deterministic stubbed outputs for tests/CI only; not intended for normal local usage of this repo.
