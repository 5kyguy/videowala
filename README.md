# VideoWala

Private, AI-assisted event media curation for photographers and families.

## Project status

**Development on this repository is discontinued.** The proof-of-concept does not meet our quality bar with the models we can run on current hardware, and we are not pursuing stronger models via upgraded hosting at this time. The code and docs remain here for reference.

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

## Why this seemed feasible (at the time)

On paper, the stack looked viable: open-weight multimodal models, local embeddings, mature face/OCR/speech tooling, and `ffmpeg` for assembly. In practice, **models that fit our hardware were not good enough for what we wanted**, and **stepping up to heavier models would require servers we are not willing to operate for this project right now**. That gap is why the PoC stops here.

## Privacy Posture

- source media stays within our infrastructure
- media is never sent to third-party model APIs
- the system uses open-weight or self-hosted components only
- tenant isolation is a first-class product requirement

## Prototype entry point (historical)

The multimodal CLI lives at `tools/vl_cli.py` (default: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`). It reflects the hardware-constrained model choice noted above.

## This repository (Stage 1 MVP, archived)

The code is a **single-process, synchronous** slice of the original product idea: FastAPI backend, SQLite, Vite + React UI (**Yarn**). **Indexing** (`POST /assets`) and **rendering** (`POST /requests/render`) run **inline** in the API—no background workers or queues. OCR, ASR, and semantic indexing/search are always on (Postgres + pgvector for vectors); see `docs/running.md`.

**Technical documentation:** start at [`docs/README.md`](docs/README.md) for architecture, data model, API, and local run instructions. Pitch and sizing notes live in [`docs/pitch/`](docs/pitch/).
