"""FastAPI orchestrator — job queue and API (Phase 6)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.hub import hub
from orchestrator.logging_config import configure_logging, get_logger
from orchestrator.routes import health, jobs, prescan, profiles
from orchestrator.workers.pool import reset_pool
from orchestrator.ws import jobs as jobs_ws

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: logging, worker pool, telemetry loop binding."""
    configure_logging(json_logs=False)
    hub.bind_loop(asyncio.get_running_loop())
    reset_pool()
    logger.info("orchestrator_start", service="viana-orchestrator", phase=6)
    yield
    reset_pool()
    logger.info("orchestrator_stop", service="viana-orchestrator")


app = FastAPI(
    title="ViAna Orchestrator",
    version="0.1.0",
    description="Job management API for ViAna moving-count engine.",
    lifespan=lifespan,
)

# Browser UI (Next.js :3000) calls this API directly via NEXT_PUBLIC_API_URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(jobs.router)
app.include_router(prescan.router)
app.include_router(profiles.router)
app.include_router(jobs_ws.router)


@app.get("/")
def root() -> dict[str, str | int]:
    """Return service identity for root probes."""
    return {"service": "viana-orchestrator", "version": "0.1.0", "phase": 6}
