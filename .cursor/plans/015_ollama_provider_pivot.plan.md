---
name: Ollama provider pivot
overview: Refactor the backend model-loading services (VLM, planner sequencing, embeddings) to support `MODEL_PROVIDER=ollama` by calling Ollama’s HTTP API for each stage, while preserving the repo’s load/work/unload pattern via Ollama `keep_alive` and `release()` hooks.
todos:
  - id: plan-ollama-config
    content: Add `MODEL_PROVIDER` and `OLLAMA_*` settings to `backend/app/config.py`, including `ollama_base_url` and stage keep-alive.
    status: completed
  - id: plan-ollama-client
    content: Create `backend/app/services/ollama_client.py` to wrap Ollama `/api/generate`, `/api/embed`, and unload via `keep_alive=0`.
    status: completed
  - id: plan-vlm-ollama
    content: Update `backend/app/services/vlm.py` to call Ollama Vision (`images` base64) when `model_provider=ollama`, and unload in `release()`.
    status: completed
  - id: plan-planner-ollama
    content: Update `backend/app/services/plan_sequencer.py` to call Ollama LLM when `model_provider=ollama`, while preserving stub mode + existing JSON parsing/validation.
    status: completed
  - id: plan-emb-ollama
    content: Update `backend/app/services/embeddings.py` to call Ollama embeddings (`/api/embed`) when `model_provider=ollama`, with dimension check against `EMBEDDING_VECTOR_DIM`.
    status: completed
  - id: plan-concurrency-note
    content: Add a minimal lock or document `INDEX_WORKERS` constraint for Ollama unload safety.
    status: completed
  - id: plan-docs
    content: Update `docs/running.md` to describe Ollama provider usage and required Ollama models.
    status: completed
isProject: false
---

## High-level approach

- Keep the existing pipeline design (indexing stages load one model family, do the work, then release/unload before the next stage).
- Add an Ollama-backed provider path that replaces the current HuggingFace/transformers integration points:
  - `backend/app/services/vlm.py`: implement VLM caption+tags via Ollama Vision-capable model using `/api/generate` with an `images` array (base64 frames).
  - `backend/app/services/plan_sequencer.py`: implement segment reordering via Ollama LLM using `/api/generate` and reuse the existing JSON extraction/validation logic.
  - `backend/app/services/embeddings.py`: implement embedding vectors via Ollama embeddings using `/api/embed`.
- Introduce a small `OllamaClient` wrapper to centralize:
  - `/api/generate` calls (including `keep_alive`)
  - `/api/embed` calls (including `keep_alive`)
  - unloading via `/api/generate` with `keep_alive=0`

## Key implementation changes

### 1. Config: select provider + Ollama settings

- Update `backend/app/config.py` to add:
  - `model_provider: Literal["transformers","ollama"]` (default `transformers`)
  - `ollama_base_url` (default `http://localhost:11434`)
  - per-stage keep-alive duration (e.g. `ollama_keep_alive_stage` default `"5m"`)
  - Ollama model ids:
    - `OLLAMA_VLM_MODEL_ID`
    - `OLLAMA_EMBEDDING_MODEL_ID`
    - `OLLAMA_PLANNER_MODEL_ID`
- Keep current env keys `VLM_MODEL_ID`, `EMBEDDING_MODEL_ID`, `PLANNER_MODEL_ID` intact for `model_provider=transformers`.
- Add a runtime safety check in embeddings path: if the returned embedding vector length does not match `EMBEDDING_VECTOR_DIM`, raise a clear error.

### 2. New Ollama client

- Add a new module, e.g. `backend/app/services/ollama_client.py`:
  - `generate(model, prompt, images=None, format=None, options=None, keep_alive=None) -> str`
  - `embed(model, texts, dimensions, keep_alive=None) -> list[list[float]]`
  - `unload(model)`: POST `/api/generate` with `keep_alive=0` (no prompt) to force unload.
- Implement HTTP requests using stdlib `urllib` (no new dependency) or `requests` (requires dependency update). Prefer stdlib unless you want `requests`.

### 3. VLM service pivot (VLM caption/tags)

- Modify `backend/app/services/vlm.py`:
  - Keep `_build_vlm_prompt(...)` and `caption_and_tags(...)` API the same.
  - In `caption_and_tags(...)`, branch on `settings.model_provider`:
    - `transformers`: keep current behavior.
    - `ollama`: replace `_ensure_loaded()` + `Qwen2.5-VLForConditionalGeneration.generate(...)` with:
      - read each extracted frame image from disk (the service already has the frame paths in `image_paths`)
      - base64 encode the bytes
      - call Ollama `/api/generate` with:
        - `model=settings.ollama_vlm_model_id`
        - `prompt=<built VLM prompt>`
        - `images=[...base64...]`
        - `keep_alive=settings.ollama_keep_alive_stage`
        - `stream=false`
        - optional: `format="json"` (or a JSON schema) to reduce parsing errors
      - parse the returned text using existing `_parse_json_block` and fall back the same way as today.
  - Implement `release()` to call `OllamaClient.unload(settings.ollama_vlm_model_id)` when provider is Ollama.

### 4. Planner sequencing pivot (segment reordering)

- Modify `backend/app/services/plan_sequencer.py`:
  - Keep stub behavior: if `settings.stage2_stub_models` is True, use `continuity_heuristic_order` exactly as today (tests rely on it).
  - In `sequence_playback_order(...)`, branch on provider:
    - `transformers`: keep current `_ensure_loaded()` + generation logic.
    - `ollama`:
      - build the same user prompt via `_build_user_prompt(...)`.
      - call Ollama `/api/generate` with:
        - `model=settings.ollama_planner_model_id`
        - `prompt=<system+user combined prompt string>` (or pass as `system` + `prompt` if we adopt `/api/chat`; prefer `/api/generate` to match current code)
        - `keep_alive=settings.ollama_keep_alive_stage`
        - `stream=false`
        - `options.temperature=settings.planner_temperature` (and stop/num_predict mapped from `planner_max_new_tokens`)
      - reuse `_extract_json_object(decoded)` + `_validate_permutation(...)` + continuity enforcement already implemented in this file.
  - In `release()`/`finally`, unload the Ollama model (send `keep_alive=0`) to approximate the existing load/unload pattern.

### 5. Embedding service pivot

- Modify `backend/app/services/embeddings.py`:
  - Keep current `stage2_stub_models` behavior.
  - In `embed_text(...)`, branch on provider:
    - `transformers`: existing SentenceTransformer flow.
    - `ollama`:
      - call `/api/embed` with:
        - `model=settings.ollama_embedding_model_id`
        - `input=[normalized_text]` (keep batching simple to match current method)
        - `dimensions=settings.embedding_vector_dim`
        - `keep_alive=settings.ollama_keep_alive_stage`
      - return `EmbeddingResult(model=<ollama model id>, vector=<returned float list>)`
  - Implement `release()` to unload Ollama embeddings model.

## Concurrency / safety note

- Ollama unload (`keep_alive=0`) affects the model globally in the Ollama server.
- The current PoC assumes serial GPU-heavy work (`INDEX_WORKERS` default `1`) and uses singletons for model services.
- Plan includes adding a single process/thread lock around Ollama calls inside each Ollama-backed service (or document that `INDEX_WORKERS` must remain `1` under Ollama).

## Docs updates

- Update `docs/running.md` and/or `docs/pitch/model-stack.md` to document:
  - `MODEL_PROVIDER=ollama`
  - required `OLLAMA_*_MODEL_ID` variables
  - expectation that Ollama models should be available (pulled/created beforehand)
  - the “unload” is done via `keep_alive=0` per request stage, not by downloading models repeatedly.

## Minimal manual verification

- With `DEV_MODE=true`: confirm tests still use stub mode and do not call Ollama.
- With `DEV_MODE=false` and provider `ollama`: run one small event ingest (`POST /assets` with a couple images/videos), confirm indexing completes and JSON parsing does not fail.
- Trigger one `POST /requests/render` to confirm planner sequencing works with Ollama output format.

## Files to change / add

- Modify:
  - `backend/app/config.py`
  - `backend/app/services/vlm.py`
  - `backend/app/services/plan_sequencer.py`
  - `backend/app/services/embeddings.py`
  - `backend/requirements.txt` (only if we choose `requests`; otherwise no)
  - `docs/running.md` (and optionally `docs/pitch/model-stack.md`)
- Add:
  - `backend/app/services/ollama_client.py` (HTTP wrapper + unload helper)

