"""Artifact HTTP serving — partial processed MP4 with range requests (G19)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from orchestrator.errors import not_found
from orchestrator.workers.pool import get_pool
from viana.io.paths import artifact_paths

router = APIRouter(tags=["artifacts"])


@router.get("/artifacts/{job_id}/partial.mp4")
def get_partial_processed_video(job_id: str) -> FileResponse:
    """Serve growing ``{stem}_processed.mp4`` for live monitor (HTTP Range supported)."""
    pool = get_pool()
    job = pool.get_job(job_id)
    if job.status not in {"PROCESSING", "PAUSED", "COMPLETED"}:
        not_found(f"processed video not available for status {job.status}")
    path = artifact_paths(job.output_dir, job.source_video_path.stem)["processed_video"]
    if not path.is_file():
        not_found(f"processed video not found: {path.name}")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
