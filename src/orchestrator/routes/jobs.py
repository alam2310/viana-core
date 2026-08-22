"""Job lifecycle routes — spawn ``python -m viana`` via the worker pool."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import Response

from orchestrator.cli import run_viana
from orchestrator.errors import engine_failed
from orchestrator.logging_config import get_logger
from orchestrator.models import JobStatus, JobSubmitRequest, JobSubmitResponse
from orchestrator.workers.pool import get_pool
from viana.config.job import JobIntakeRequest, JobIntakeResponse, JobPrescanConfirmRequest
from viana.io.checkpoint import load_checkpoint
from viana.io.paths import resolve_artifact

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])


@router.post("/jobs/intake", status_code=201, response_model=JobIntakeResponse)
def post_jobs_intake(body: JobIntakeRequest) -> JobIntakeResponse:
    """Register video path(s) for backend prescan (`PRESCAN_PENDING`)."""
    return get_pool().intake(body)


@router.patch("/jobs/{job_id}/prescan", response_model=JobStatus)
def patch_job_prescan(job_id: str, body: JobPrescanConfirmRequest) -> JobStatus:
    """Confirm reviewed calibration and transition job to ``READY``."""
    return get_pool().confirm_prescan(job_id, body)


@router.post("/jobs/{job_id}/prescan/retry", response_model=JobStatus)
def retry_job_prescan(job_id: str) -> JobStatus:
    """Retry prescan after ``PRESCAN_FAILED`` → ``PRESCAN_PENDING``."""
    return get_pool().retry_prescan(job_id)


@router.get("/jobs/{job_id}/prescan/preview")
def get_job_prescan_preview(
    job_id: str,
    frame_offset_sec: float = Query(default=0.0, ge=0.0),
) -> dict[str, object]:
    """Re-run prescan at ``frame_offset_sec`` for scrub preview (G8)."""
    return get_pool().prescan_preview(job_id, frame_offset_sec)


@router.post("/jobs", status_code=201, response_model=JobSubmitResponse)
def post_job(body: JobSubmitRequest) -> JobSubmitResponse:
    """Accept a moving-count job. Backend assigns job_id and gpu_device."""
    return get_pool().submit(body)


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(project_id: str | None = Query(default=None)) -> list[JobStatus]:
    """List jobs, optionally filtered by project_id."""
    return get_pool().list_jobs(project_id=project_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    """Return job status for a single id."""
    pool = get_pool()
    return pool.to_status(pool.get_job(job_id))


@router.post("/jobs/{job_id}/resume", response_model=JobSubmitResponse)
def resume_job(job_id: str) -> JobSubmitResponse:
    """Explicit resume from checkpoint (never silent)."""
    return get_pool().resume(job_id)


@router.post("/jobs/{job_id}/start-fresh", response_model=JobSubmitResponse)
def start_fresh_job(job_id: str) -> JobSubmitResponse:
    """Delete checkpoint via engine start_fresh and restart."""
    return get_pool().start_fresh(job_id)


@router.delete("/jobs/{job_id}", status_code=204)
def cancel_job(job_id: str) -> Response:
    """Cancel a queued or running worker."""
    get_pool().cancel(job_id)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/aggregate")
def aggregate_job(job_id: str) -> dict[str, object]:
    """Rebuild `_15min.csv` from events (CLI `viana aggregate`)."""
    pool = get_pool()
    job = pool.get_job(job_id)
    ckpt = resolve_artifact(job.output_dir, job.source_video_path.stem, "checkpoint")
    partial = False
    if ckpt.is_file():
        checkpoint = load_checkpoint(ckpt)
        partial = not checkpoint.is_complete()
    args = [
        "aggregate",
        "--source",
        str(job.source_video_path),
        "--project-id",
        job.project_id,
        "--output-dir",
        str(job.output_dir),
    ]
    if partial:
        args.append("--partial")
    logger.info("viana_aggregate", job_id=job_id, args=args)
    result = run_viana(args, timeout=120.0)
    if result.returncode != 0:
        engine_failed(result.stderr.strip() or result.stdout.strip() or "aggregate failed")
    parsed: object = json.loads(result.stdout)
    if not isinstance(parsed, dict):
        engine_failed("aggregate stdout was not a JSON object")
    return parsed
