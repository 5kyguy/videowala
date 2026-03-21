---
name: Indexing quality implementation
overview: Implement event-aware VLM (same model), richer embeddings + BGE-M3 + pgvector migration, multilingual ASR, Paddle-only OCR with VLM gate and preprocessing, face match threshold 0.8, and VLM-based cull score (no person-match bonus).
todos:
  - id: schema-event-tags
    content: Extend Event schema + SQLite migration for predefined tag vocabulary and OCR/event locale hints
    status: pending
  - id: pgvector-dim
    content: Migrate asset_vectors to new embedding dimension (BGE-M3) and document reindex
    status: pending
  - id: vlm-prompt
    content: "VLM: event context, predefined tags + freeform, confidence, higher max_new_tokens; wire EventRepository in index_asset"
    status: pending
  - id: pipeline-order
    content: "Reorder index_asset: VLM before OCR; conditional OCR from VLM tags; update delete_for_asset insight order"
    status: pending
  - id: ocr-paddle
    content: "OCR: Paddle-only, remove EasyOCR; event-based lang; preprocessing; extract() signature for context"
    status: pending
  - id: asr-multilingual
    content: "ASR: auto language detection (Hindi, Gujarati, English) via faster-whisper"
    status: pending
  - id: embeddings-rich
    content: "Embeddings: structured combined text + BGE-M3 default + truncation policy"
    status: pending
  - id: faces-threshold
    content: Face match threshold 0.8
    status: pending
  - id: cull-vlm
    content: "Cull score: VLM signal; remove person/detections bonuses from formula"
    status: pending
isProject: false
---

# Implementation plan: indexing quality (PoC)

This plan matches the agreed product decisions below. It supersedes earlier exploratory notes.

## Summary of decisions


| Area       | Decision                                                                                                                                                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VLM        | Keep `HuggingFaceTB/SmolVLM2-2.2B-Instruct`. Add **event context** (title, type, venue, date). **Richer prompt**: predefined tag list + model may add tags; **more `max_new_tokens`**. Emit **confidence** for the chosen best caption. |
| Embeddings | Apply all three: **richer combined text** (caption, tags, ASR, OCR, matched person names), **stronger default model** (BGE-M3), **fix truncation** so structured fields are not silently dropped.                                       |
| ASR        | Same Whisper stack; enable **multilingual** handling (English, Hindi, Gujarati) via **automatic language detection** (no fixed `language=`).                                                                                            |
| OCR        | **Paddle PP-OCR only**; **remove EasyOCR**. **Language/variant** from event context. **Preprocessing** before inference. Run OCR **only if** VLM assigns a trigger tag (e.g. `text` / `signage` / configurable).                        |
| Faces      | Logic unchanged; **match threshold `0.8`**.                                                                                                                                                                                             |
| Cull score | **Incorporate VLM signal**; **no** person-match bonus (and remove detection-based bonus if it acted as proxy for “people”).                                                                                                             |


---

## 1. Event context and predefined tags (schema + API)

**Problem:** `[Event](backend/app/schemas.py)` has `title`, `event_type`, `venue`, `date` only—no tag vocabulary or locale hints.

**Add (SQLite + Pydantic):**

- `predefined_tags_json` — JSON array of strings (allowed labels for VLM; model may also add tags not in the list).
- Optional: `ocr_languages` or `primary_locale` — string or JSON for Paddle language selection (e.g. `en`, `hi`, `gu` / multilingual mode per Paddle API).

**Work:**

- Migration in `[backend/app/db.py](backend/app/db.py)` (`ALTER TABLE events` + defaults).
- Extend `Event`, `EventCreate`, PATCH/update routes if events are editable.
- `[index_asset](backend/app/services/indexing.py)`: `EventRepository.get(asset.event_id)` and pass structured context into VLM and OCR.

---

## 2. VLM (`[backend/app/services/vlm.py](backend/app/services/vlm.py)`)

**Prompt changes:**

- Inject event: title, type, venue, date, and bullet list of **predefined tags** (“assign zero or more from this list; you may add additional short tags”).
- Ask for JSON with at least: `caption`, `tags` (mix of predefined + new), and `**caption_confidence`** in `[0,1]` (model-estimated; see below).
- Increase `max_new_tokens` (e.g. 256–384; tune after smoke tests).

**Caption confidence:**

- **Primary:** If `generate` supports `output_scores=True`, derive a scalar from token-level probabilities for the generated span (e.g. mean log-prob → exp → [0,1]), attached to the frame that produced the **best** caption.
- **Fallback:** If scores are unavailable, omit or set heuristic from parse success only—document limitation in code comment.

**Dataclass:** Extend `VlmResult` with `caption_confidence: float | None` and optional `tags_assigned: dict` (predefined vs freeform) if useful for UI.

**Call site:** `caption_and_tags(..., event_context: dict | None, predefined_tags: list[str])`.

---

## 3. Pipeline order (`[backend/app/services/indexing.py](backend/app/services/indexing.py)`)

**Required:** OCR must run **after** VLM so tags can gate OCR.

**Proposed order:**

1. `ensure_asset_proxy`
2. Face detect + match (unchanged position vs proxy path)
3. ASR (video only) — independent of VLM
4. **VLM** (with event + predefined tags)
5. **OCR** only if `_should_run_ocr(vlm.tags)` (configurable tag set, e.g. contains `"text"` or `"signage"`—**exact list in config** `OCR_TRIGGER_TAGS`)
6. Segments + insights + **embedding**

**Reindex cleanup:** Ensure `InsightRepository.delete_for_asset` for full reindex still clears `ocr_text` / `vlm_`* consistently when types are listed (align with current delete list).

---

## 4. OCR (`[backend/app/services/ocr.py](backend/app/services/ocr.py)`)

- Remove EasyOCR code paths, env `OCR_ENGINE`, and fallbacks to EasyOCR.
- **PaddleOCR** only: lazy-init with **language** from event context (and document supported Paddle langs for hi/gu/en).
- **Preprocessing:** e.g. PIL/OpenCV resize/contrast/grayscale or minimal sharpen before `predict` (keep dependency footprint small—prefer PIL + numpy already in stack).
- `**extract` signature:** e.g. `extract(media_path, *, event: Event | None, run_ocr: bool)` or pass `ocr_context` dict; if `run_ocr` is False, return `[], "skipped-vlm-gate"`.
- Update `[config.py](backend/app/config.py)`: drop `ocr_engine`; add `OCR_TRIGGER_TAGS` (comma-separated or JSON).

**Tests:** Stub/stage2 paths must still work.

---

## 5. ASR (`[backend/app/services/asr.py](backend/app/services/asr.py)`)

- Call `transcribe(..., language=None)` (or omit) so faster-whisper **auto-detects** language per file—covers **English, Hindi, Gujarati** in one deployment.
- Optional: `task="transcribe"` explicitly.
- Log detected language from `info.language` when available for debugging.

---

## 6. Embeddings (`[backend/app/services/embeddings.py](backend/app/services/embeddings.py)` + vector store)

**Richer text (order matters):**

- Build a structured block, e.g. lines for `Caption:`, `Tags:`, `People:`, `ASR:`, `OCR:` — include matched person **display names** from face matches (when present).
- Default model: `**BAAI/bge-m3`** via `EMBEDDING_MODEL_ID` in `[config.py](backend/app/config.py)`.

**Truncation:**

- Keep a high character budget for the **concatenated** string; avoid cutting only the tail—prefer proportional caps per section or “caption + tags first, then ASR/OCR remainder”.

**pgvector:** `[vector_store.py](backend/app/vector_store.py)` hardcodes `vector(384)`. BGE-M3 dense dimension is **1024** (verify at runtime). Plan:

- Migration: alter column type / recreate table with `vector(1024)` and **reindex all assets** (document one-shot script or “reindex event” API).
- Update `migrate_pgvector()` for new installs.

---

## 7. Face threshold (`[backend/app/services/faces.py](backend/app/services/faces.py)`)

- Change default `threshold` in `match_faces` from `0.72` to `**0.8`**. Update any call sites that pass explicit threshold.

---

## 8. Cull score (`[backend/app/services/indexing.py](backend/app/services/indexing.py)` — `_base_cull_score`)

- **Remove** score bumps tied to **face detections** and **person matching** (user: no person-match bonus).
- **Add** VLM-based components, e.g.:
  - use `caption_confidence` (weight into score),
  - optional tag-based hints (e.g. “blur” / “dark” lowering score if such tags exist in predefined list),
- Keep neutral components: resolution, has_audio, ASR presence, OCR presence **only when OCR ran**—or drop OCR from base if gated rarely.

Define the exact formula in code with comments so PoC tuning is easy.

---

## 9. Frontend / API touchpoints

- Event create/edit: fields for **predefined tags** and **OCR/locale** hints.
- Display: show `caption_confidence` and tag source (predefined vs added) if exposed in insight payloads.

---

## 10. Files likely touched (checklist)


| File                                                                       | Changes                                  |
| -------------------------------------------------------------------------- | ---------------------------------------- |
| `[backend/app/db.py](backend/app/db.py)`                                   | `events` columns + migration             |
| `[backend/app/schemas.py](backend/app/schemas.py)`                         | Event fields                             |
| `[backend/app/api/routes.py](backend/app/api/routes.py)`                   | Event payloads                           |
| `[backend/app/services/vlm.py](backend/app/services/vlm.py)`               | Prompt, tokens, confidence, event + tags |
| `[backend/app/services/indexing.py](backend/app/services/indexing.py)`     | Order, cull, embedding text, Event load  |
| `[backend/app/services/ocr.py](backend/app/services/ocr.py)`               | Paddle-only, gate, preprocess, context   |
| `[backend/app/services/asr.py](backend/app/services/asr.py)`               | Auto language                            |
| `[backend/app/services/embeddings.py](backend/app/services/embeddings.py)` | Rich text                                |
| `[backend/app/config.py](backend/app/config.py)`                           | Defaults, OCR triggers, embedding id     |
| `[backend/app/vector_store.py](backend/app/vector_store.py)`               | Dimension migration                      |
| `[backend/app/services/faces.py](backend/app/services/faces.py)`           | Threshold 0.8                            |
| Tests under `[backend/tests/](backend/tests/)`                             | Update stubs and assertions              |


---

## Risk notes

- **VLM JSON:** Larger outputs need robust `_parse_json_block` and partial recovery.
- **OCR gate:** False negatives (no trigger tag → no OCR) miss text; tune trigger tag list and optionally allow “always OCR” per event flag (future).
- **BGE-M3 + pgvector:** One-time migration and full reindex mandatory.

