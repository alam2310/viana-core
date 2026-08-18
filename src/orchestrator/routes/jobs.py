"""Job lifecycle routes. Handlers are 501 until Phase 5 (`viana run`)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from orchestrator.errors import not_implemented
from orchestrator.logging_config import get_logger
from orchestrator.models import JobStatus, JobSubmitRequest, JobSubmitResponse

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])


@router.post("/jobs", status_code=201, response_model=JobSubmitResponse)
def post_job(body: JobSubmitRequest) -> JobSubmitResponse:
    """Accept a moving-count job. Backend will assign job_id and gpu_device."""
    logger.info(
        "job_submit_stub",
        project_id=body.project_id,
        resume=body.resume,
        start_fresh=body.start_fresh,
    )
    not_implemented()


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs(project_id: str | None = Query(default=None)) -> list[JobStatus]:
    """List jobs, optionally filtered by project_id."""
    logger.info("jobs_list_stub", project_id=project_id)
    not_implemented()


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    """Return job status for a single id."""
    logger.info("job_get_stub", job_id=job_id)
    not_implemented()


@router.post("/jobs/{job_id}/resume", response_model=JobSubmitResponse)
def resume_job(job_id: str) -> JobSubmitResponse:
    """Explicit resume from checkpoint (never silent)."""
    logger.info("job_resume_stub", job_id=job_id)
    not_implemented()


@router.post("/jobs/{job_id}/start-fresh", response_model=JobSubmitResponse)
def start_fresh_job(job_id: str) -> JobSubmitResponse:
    """Delete checkpoint and restart the job."""
    logger.info("job_start_fresh_stub", job_id=job_id)
    not_implemented()


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> None:
    """Cancel a queued or running worker."""
    logger.info("job_cancel_stub", job_id=job_id)
    not_implemented()


@router.post("/jobs/{job_id}/aggregate")
def aggregate_job(job_id: str) -> None:
    """Rebuild `_15min.csv` from events (CLI `viana aggregate`)."""
    logger.info("job_aggregate_stub", job_id=job_id)
    not_implemented()
