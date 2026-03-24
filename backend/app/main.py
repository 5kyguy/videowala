from __future__ import annotations

import warnings
import logging

# Quiet third-party noise during model inference (PaddleOCR / requests).
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*",
    category=UserWarning,
)
# Importing `requests` runs a version check that can warn before we could register a category filter;
# keep `chardet>=3.0.2,<6` in requirements (see comment there). This catches any similar future noise.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 \(.*\) or chardet \(.*\)/charset_normalizer \(.*\) doesn't match a supported version!",
)
try:
    from requests import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import settings
from .db import migrate
from .repositories import RenderRepository
from .vector_store import migrate_pgvector


class _QuietPollAccessFilter(logging.Filter):
    """Hide noisy GETs from the dashboard poll loop (summary + renders every few seconds)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True
        msg = record.getMessage()
        if "GET /events/" in msg and ("/summary?" in msg or "/renders?" in msg):
            return False
        return True


app = FastAPI(title=settings.app_name, version="0.1.0")
logger = logging.getLogger(__name__)

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
    logging.getLogger("uvicorn.access").addFilter(_QuietPollAccessFilter())
    migrate()
    recovered = RenderRepository.mark_incomplete_jobs_failed(
        "Render interrupted by backend restart/reload. Please retry."
    )
    if recovered:
        logger.warning("Marked %d stale render job(s) as failed on startup.", recovered)
    try:
        migrate_pgvector()
    except Exception:
        # pgvector optional at runtime; semantic search/embed upsert degrade gracefully.
        pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
