"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str | int]:
    """Liveness probe for container orchestration and UI health checks."""
    return {"status": "ok", "phase": 0}
