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

To reach the API from **another machine** (browser on your laptop, another host), bind on all interfaces; uvicorn’s default is `127.0.0.1` only, so the public IP will refuse connections until you do:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

(`--reload` is optional.) Open the host firewall for `8000/tcp` if needed; ensure any cloud “security group” allows it too.

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
  - `EMBEDDING_MODEL_ID` → text embedding model (`BAAI/bge-m3` by default; must match `EMBEDDING_VECTOR_DIM`, usually `1024`)
  - `EMBEDDING_VECTOR_DIM` → pgvector column size (defaults to `1024` for BGE-M3; changing model may require Postgres `asset_vectors` rebuild — see startup migration)
  - `OCR_TRIGGER_TAGS` → comma-separated VLM tag names that trigger Paddle OCR after captioning (e.g. `text,signage,document,readable_text`)
- **OCR** uses **PaddleOCR** only; per-event `ocr_languages` on the event record selects the Paddle `lang` code (e.g. `en`, `hi`, `gu`).
- **Indexing UX**:
  - `INDEXING_PROGRESS` → `0` / `1` to force-disable/force-enable tqdm progress bars
- **Rendering:** uses `ffmpeg` / `ffprobe` on PATH. Output uses **center crop** to the requested orientation (landscape 16:9 or portrait 9:16), optional **transpose** when VLM tags indicate sideways capture, then **letterbox padding** only so concatenated clips share one canvas. No user-facing resolution, fps, filter preset, or crossfade settings.

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

### Frontend via ngrok (no custom domain)

1. Copy `frontend/.env.example` → `frontend/.env`.
2. Set **`VITE_DEV_PUBLIC_HOST`** to the hostname ngrok gives you (e.g. `abc123.ngrok-free.app`) — no `https://`.
3. Set **`VITE_API_BASE_URL=/api`** so the browser only talks to the **same** HTTPS origin as the UI. Vite proxies `/api` → FastAPI on `127.0.0.1:8000` (see `vite.config.ts`). This avoids **mixed content** errors (an `https://` ngrok page cannot call `http://your-vps:8000` directly).
4. Ensure the API is listening on the same machine as Vite (`uvicorn … --host 0.0.0.0 --port 8000`). The proxy uses `VITE_API_PROXY_TARGET` if you need a non-default target (defaults to `http://127.0.0.1:8000`).
5. **Alternative:** run a **second** ngrok tunnel to port `8000` and set **`VITE_API_BASE_URL`** to that tunnel’s `https://…` URL (no proxy).
6. From `frontend/`: `npm install` then `npm run dev` (port `5173`).
7. Start the tunnel: `ngrok http 5173` (or your ngrok UI equivalent).

Run **`yarn dev`** and **`ngrok`** in separate long-lived shells so they survive SSH disconnects—e.g. **`tmux`** (one pane or window each), **`screen`**, or **`systemd`** if you want them as services.

Restart `yarn dev` after changing `.env` (Vite reads it at startup).

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
