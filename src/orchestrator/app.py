"""FastAPI orchestrator — job queue and API (Phase 6)."""

from fastapi import FastAPI

from orchestrator.routes import health

app = FastAPI(
    title="ViAna Orchestrator",
    version="0.1.0",
    description="Job management API for ViAna moving-count engine.",
)

app.include_router(health.router, tags=["health"])


@app.get("/")
def root() -> dict:
    return {"service": "viana-orchestrator", "version": "0.1.0", "phase": 0}
