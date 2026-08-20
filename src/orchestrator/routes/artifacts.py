"""Artifact HTTP serving — source and partial processed MP4 with range requests."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from orchestrator.errors import not_found
from orchestrator.workers.pool import PRESCAN_PHASE_STATUSES, get_pool
from viana.io.paths import artifact_paths

router = APIRouter(tags=["artifacts"])

_SOURCE_PHASE_STATUSES = PRESCAN_PHASE_STATUSES | {"READY"}


def _inline_mp4_response(path: Path) -> FileResponse:
    """Serve MP4 for ``<video>``: Range + inline disposition (not attachment download)."""
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        content_disposition_type="inline",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


@router.get("/artifacts/{job_id}/source.mp4")
def get_source_video(job_id: str) -> FileResponse:
    """Serve intake source MP4 for prescan review scrub (HTTP Range supported)."""
    pool = get_pool()
    job = pool.get_job(job_id)
    if job.status not in _SOURCE_PHASE_STATUSES:
        not_found(f"source video not available for status {job.status}")
    path = job.source_video_path
    if not path.is_file():
        not_found(f"source video not found: {path.name}")
    return _inline_mp4_response(path)


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
    return _inline_mp4_response(path)
