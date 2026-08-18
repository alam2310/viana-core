"""Prescan utility routes — spawn ``python -m viana prescan``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from orchestrator.cli import run_viana
from orchestrator.errors import engine_failed, not_found
from orchestrator.logging_config import get_logger
from orchestrator.models import PrescanRequest
from orchestrator.settings import output_parent, project_dir

logger = get_logger(__name__)

router = APIRouter(tags=["utils"])

_PREVIEW_FILES: dict[str, Path] = {}


@router.post("/utils/prescan")
def post_prescan(body: PrescanRequest) -> dict[str, Any]:
    """OCR + line proposal. Spawns `python -m viana prescan`."""
    output_dir = project_dir(body.project_id)
    args = [
        "prescan",
        "--source",
        body.source_video_path,
        "--project-id",
        body.project_id,
        "--frame-offset",
        str(body.frame_offset_sec),
        "--output-dir",
        str(output_dir),
    ]
    logger.info("viana_prescan", project_id=body.project_id)
    result = run_viana(args, timeout=120.0)
    if result.returncode != 0:
        engine_failed(result.stderr.strip() or result.stdout.strip() or "prescan failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        engine_failed("prescan stdout was not a JSON object")
    prescan_id = payload.get("prescan_id")
    disk = payload.get("preview_url")
    if isinstance(prescan_id, str) and isinstance(disk, str) and disk:
        _PREVIEW_FILES[prescan_id] = Path(disk)
        payload["preview_url"] = f"/utils/prescan/{prescan_id}/preview.jpg"
    return payload


@router.get("/utils/prescan/{prescan_id}/preview.jpg")
def get_prescan_preview(prescan_id: str) -> FileResponse:
    """Serve the preview JPEG written by ``viana prescan``."""
    path = _PREVIEW_FILES.get(prescan_id)
    if path is None or not path.is_file():
        not_found(f"preview not found: {prescan_id}")
    resolved = path.resolve()
    if not resolved.is_relative_to(output_parent().resolve()):
        not_found(f"preview not found: {prescan_id}")
    return FileResponse(resolved, media_type="image/jpeg")
