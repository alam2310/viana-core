"""Calibration profile routes. 501 until storage is wired in Phase 6."""

from __future__ import annotations

from fastapi import APIRouter

from orchestrator.errors import not_implemented
from orchestrator.logging_config import get_logger
from orchestrator.models import CalibrationProfile

logger = get_logger(__name__)

router = APIRouter(tags=["profiles"])


@router.get("/projects/{project_id}/profiles", response_model=list[CalibrationProfile])
def list_profiles(project_id: str) -> list[CalibrationProfile]:
    """List calibration profiles for a project."""
    logger.info("profiles_list_stub", project_id=project_id)
    not_implemented()


@router.post("/projects/{project_id}/profiles", status_code=201, response_model=CalibrationProfile)
def create_profile(project_id: str, body: CalibrationProfile) -> CalibrationProfile:
    """Store a calibration profile under the project output directory."""
    logger.info(
        "profiles_create_stub",
        project_id=project_id,
        profile_id=body.profile_id,
    )
    not_implemented()
