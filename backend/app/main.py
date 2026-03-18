from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import settings
from .db import migrate
from .vector_store import migrate_pgvector

app = FastAPI(title=settings.app_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    migrate()
    try:
        migrate_pgvector()
    except Exception:
        # pgvector optional at runtime; semantic search/embed upsert degrade gracefully.
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
