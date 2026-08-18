"""Prescan utility routes. 501 until Phase 4 (`viana prescan`)."""

from __future__ import annotations

from fastapi import APIRouter

from orchestrator.errors import not_implemented
from orchestrator.logging_config import get_logger
from orchestrator.models import PrescanRequest

logger = get_logger(__name__)

router = APIRouter(tags=["utils"])


@router.post("/utils/prescan")
def post_prescan(body: PrescanRequest) -> None:
    """OCR + line proposal. Spawns `python -m viana prescan` when implemented."""
    logger.info("prescan_stub", project_id=body.project_id)
    not_implemented()
