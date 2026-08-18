"""Calibration profile routes — JSON under ``{output_dir}/profiles/``."""

from __future__ import annotations

from fastapi import APIRouter

from orchestrator.errors import bad_request
from orchestrator.logging_config import get_logger
from orchestrator.models import CalibrationProfile
from orchestrator.profiles_store import list_profiles as list_disk_profiles
from orchestrator.profiles_store import save_profile
from orchestrator.settings import project_dir
from viana.config.job import PROJECT_ID_PATTERN

logger = get_logger(__name__)

router = APIRouter(tags=["profiles"])


def _require_project_id(project_id: str) -> None:
    """Reject non-slug project ids."""
    if not PROJECT_ID_PATTERN.match(project_id):
        bad_request("project_id must match [a-z0-9][a-z0-9_-]*")


@router.get("/projects/{project_id}/profiles", response_model=list[CalibrationProfile])
def list_profiles(project_id: str) -> list[CalibrationProfile]:
    """List calibration profiles for a project."""
    _require_project_id(project_id)
    profiles = list_disk_profiles(project_dir(project_id))
    logger.info("profiles_list", project_id=project_id, count=len(profiles))
    return profiles


@router.post("/projects/{project_id}/profiles", status_code=201, response_model=CalibrationProfile)
def create_profile(project_id: str, body: CalibrationProfile) -> CalibrationProfile:
    """Store a calibration profile under the project output directory."""
    _require_project_id(project_id)
    saved = save_profile(project_dir(project_id), body)
    logger.info("profiles_create", project_id=project_id, profile_id=saved.profile_id)
    return saved
