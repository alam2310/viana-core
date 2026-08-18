"""HTTP errors for the orchestrator."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException


def not_found(detail: str) -> NoReturn:
    """Raise 404."""
    raise HTTPException(status_code=404, detail=detail)


def conflict(detail: str) -> NoReturn:
    """Raise 409 (checkpoint / busy)."""
    raise HTTPException(status_code=409, detail=detail)


def bad_request(detail: str) -> NoReturn:
    """Raise 400."""
    raise HTTPException(status_code=400, detail=detail)


def engine_failed(detail: str) -> NoReturn:
    """Raise 502 when the CLI exits non-zero without a usable payload."""
    raise HTTPException(status_code=502, detail=detail)
