"""Shared HTTP errors for orchestrator stubs."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException

ENGINE_NOT_READY_DETAIL = (
    "Not implemented: GPU workers require Phase 5 (python -m viana run). "
    "CLI commands in docs/PROJECT_STATUS.md are still stubs."
)


def not_implemented() -> NoReturn:
    """Raise 501 until the engine CLI can actually run jobs."""
    raise HTTPException(status_code=501, detail=ENGINE_NOT_READY_DETAIL)
