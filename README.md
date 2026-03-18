# VideoWala

Private, AI-assisted event media curation for photographers and families.

## The Problem

Photographers and families capture far more media than they can efficiently review, shortlist, and edit. A single wedding, birthday, or school function can produce thousands of photos and hours of video across multiple devices. The editing burden is high, repetitive, and time-sensitive.

Today, teams either:

- spend many manual hours reviewing and assembling footage
- use generic cloud AI tools that do not fit event workflows well
- avoid AI because privacy and client-trust concerns are too high

## The Opportunity

Build a privacy-first event media platform that helps users turn raw event coverage into highlight reels, chronological films, and person-focused edits faster than manual workflows.

The differentiator is not generative novelty. It is private, practical acceleration:

- understand what is in the photos and videos
- find key people, moments, and scenes
- rank the strongest media for a given brief
- assemble a first-cut output with deterministic tools

## Who It Is For

- professional photographers and studios
- videographers covering weddings and events
- families who want usable outputs from personal phone footage
- agencies serving schools, social events, or corporate gatherings

## Product Positioning

The product is not generative video editing. It is private media understanding plus deterministic assembly:

- source media remains unchanged
- media is never sent to third-party model APIs
- models are used for understanding, ranking, and retrieval
- final output is created through repeatable edit plans and media tooling

## Core Value Proposition

Upload event media, provide context, describe the desired output, and receive a high-quality first cut without sending sensitive media to third-party AI APIs.

## Example Inputs

- event name, date, venue, and event type
- important people and optional reference images
- a request such as:
  - "Create a 60-second Instagram reel focused on the couple"
  - "Create a chronological feature film of the wedding"
  - "Create a family montage focused on the birthday child"

## Example Outputs

- teaser/highlight reel
- chronological feature film
- person-focused montage
- media shortlist for a human editor

## Why Now

The technical ingredients are now good enough to make this credible:

- open-weight multimodal models can understand both images and videos
- open embedding models can support semantic retrieval locally
- face, OCR, and speech pipelines are mature enough for self-hosting
- `ffmpeg` remains a reliable deterministic backend for assembly

For the current prototype direction, `HuggingFaceTB/SmolVLM2-2.2B-Instruct` is the preferred multimodal model because it is sufficient for early media-understanding workflows without the larger deployment cost of Qwen-class models.

## Privacy Posture

- source media stays within our infrastructure
- media is never sent to third-party model APIs
- the system uses open-weight or self-hosted components only
- tenant isolation is a first-class product requirement

## Prototype Entry Point

The tested multimodal prototype is now exposed as `tools/vl_cli.py`, using `HuggingFaceTB/SmolVLM2-2.2B-Instruct` by default for image and video understanding.

## This repository (Stage 1 MVP)

The code here is a **single-process, synchronous** slice of the product above: FastAPI backend, SQLite, Vite + React UI (**Yarn**). **Indexing** (`POST /assets`) and **rendering** (`POST /requests/render`) run **inline** in the API—no background workers or queues yet. OCR, ASR, and semantic indexing/search are always on (Postgres + pgvector for vectors); see `docs/running.md`.

**Technical documentation:** start at [`docs/README.md`](docs/README.md) for links to architecture, data model, API, and how to run locally. Pitch and sizing notes live in [`docs/pitch/`](docs/pitch/).
