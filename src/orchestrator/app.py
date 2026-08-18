"""FastAPI orchestrator — job queue and API (Phase 6)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.logging_config import configure_logging, get_logger
from orchestrator.routes import health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: configure logging on startup."""
    configure_logging(json_logs=False)
    logger.info("orchestrator_start", service="viana-orchestrator", phase=0)
    yield
    logger.info("orchestrator_stop", service="viana-orchestrator")


app = FastAPI(
    title="ViAna Orchestrator",
    version="0.1.0",
    description="Job management API for ViAna moving-count engine.",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])


@app.get("/")
def root() -> dict[str, str | int]:
    """Return service identity for root probes."""
    return {"service": "viana-orchestrator", "version": "0.1.0", "phase": 0}
