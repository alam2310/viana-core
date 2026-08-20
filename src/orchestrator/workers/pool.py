"""GPU worker pool: max two concurrent ``python -m viana`` processes."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import Popen, TimeoutExpired  # nosec B404
from typing import Any, Literal

from orchestrator.cli import run_viana, start_viana_process
from orchestrator.errors import bad_request, conflict, not_found
from orchestrator.hub import hub
from orchestrator.logging_config import get_logger
from orchestrator.models import (
    JobProgress,
    JobStatus,
    JobSubmitRequest,
    JobSubmitResponse,
)
from orchestrator.preview_registry import preview_http_url, register_preview, rewrite_preview_url
from orchestrator.settings import resolve_output_dir
from viana.config.job import (
    JobConfig,
    JobIntakeItem,
    JobIntakeRequest,
    JobIntakeResponse,
    JobMetadata,
    JobPrescanConfirmRequest,
    JobStatusLiteral,
    ProposedLines,
    ViAnaTaskParameters,
)
from viana.config.job import (
    JobSubmitRequest as EngineSubmit,
)
from viana.io.checkpoint import load_checkpoint, utc_now_iso
from viana.io.paths import artifact_paths

logger = get_logger(__name__)

MAX_CONCURRENT_GPU_JOBS = 2
MAX_CONCURRENT_PRESCAN_JOBS = 2
GPU_DEVICES: tuple[str, str] = ("cuda:0", "cuda:1")

CHECKPOINT_CONFLICT = (
    "checkpoint exists; POST /jobs/{id}/resume or set start_fresh=true (no silent resume)"
)

CommandKind = Literal["run", "resume"]

PRESCAN_PHASE_STATUSES: frozenset[JobStatusLiteral] = frozenset(
    {
        "PRESCAN_PENDING",
        "PRESCAN_RUNNING",
        "PRESCAN_FAILED",
        "AWAITING_REVIEW",
    }
)


@dataclass
class JobRecord:
    """In-memory job lifecycle record."""

    job_id: str
    status: JobStatusLiteral
    source_video_path: Path
    project_id: str
    output_dir: Path
    submit: EngineSubmit | None = None
    gpu_device: str | None = None
    queue_position: int = 0
    config_path: Path | None = None
    process: Popen[str] | None = None
    progress: JobProgress | None = None
    crossing_count: int = 0
    error_message: str | None = None
    command: CommandKind = "run"
    proposed_metadata: JobMetadata | None = None
    proposed_lines: ProposedLines | None = None
    proposed_preview_url: str | None = None
    confirmed_metadata: JobMetadata | None = None
    confirmed_task_parameters: ViAnaTaskParameters | None = None
    prescan_queue_position: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    video_duration_sec: float | None = None
    processing_started_monotonic: float | None = None
    processing_ended_monotonic: float | None = None


class WorkerPool:
    """Assigns at most two GPUs and spawns ``python -m viana`` workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: list[str] = []
        self._prescan_queue: list[str] = []
        self._threads: list[threading.Thread] = []

    def occupied_gpus(self) -> set[str]:
        """Return GPU ids currently running a PROCESSING job."""
        with self._lock:
            return {
                job.gpu_device
                for job in self._jobs.values()
                if job.status == "PROCESSING" and job.gpu_device is not None
            }

    def assign_gpu(self, occupied: set[str]) -> str | None:
        """Return the next free ``cuda:0`` or ``cuda:1``, or None if both busy."""
        for device in GPU_DEVICES:
            if device not in occupied:
                return device
        return None

    def shutdown(self) -> None:
        """Terminate running workers (app lifespan)."""
        with self._lock:
            running = [job for job in self._jobs.values() if job.process is not None]
        for job in running:
            proc = job.process
            if proc is not None and proc.poll() is None:
                proc.terminate()

    def list_jobs(self, project_id: str | None = None) -> list[JobStatus]:
        """Return job status payloads, optionally filtered."""
        with self._lock:
            records = list(self._jobs.values())
        if project_id is not None:
            records = [job for job in records if job.project_id == project_id]
        return [self.to_status(job) for job in records]

    def get_job(self, job_id: str) -> JobRecord:
        """Return a job or 404."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            not_found(f"job not found: {job_id}")
        return job

    def to_status(self, job: JobRecord) -> JobStatus:
        """Build JobStatus including disk checkpoint flag."""
        ckpt = artifact_paths(job.output_dir, job.source_video_path.stem)["checkpoint"]
        exists = ckpt.is_file()
        checkpoint_exists = exists and job.status in {"PAUSED", "FAILED"}
        return JobStatus(
            job_id=job.job_id,
            status=job.status,
            task_type="ViAna_Moving",
            source_video_path=str(job.source_video_path),
            project_id=job.project_id,
            output_dir=str(job.output_dir),
            checkpoint_exists=checkpoint_exists,
            gpu_device=job.gpu_device,
            queue_position=job.queue_position,
            progress=job.progress,
            error_message=job.error_message,
            proposed_metadata=job.proposed_metadata,
            proposed_lines=job.proposed_lines,
            proposed_preview_url=job.proposed_preview_url,
            confirmed_metadata=job.confirmed_metadata,
            confirmed_task_parameters=job.confirmed_task_parameters,
            created_at=job.created_at,
            video_duration_sec=job.video_duration_sec,
            processing_duration_sec=_processing_duration_sec(job),
        )

    def to_submit_response(self, job: JobRecord) -> JobSubmitResponse:
        """Build JobSubmitResponse (gpu_device always set for the contract)."""
        device = job.gpu_device or GPU_DEVICES[0]
        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status,
            gpu_device=device,
            queue_position=job.queue_position,
            output_dir=str(job.output_dir),
        )

    def intake(self, body: JobIntakeRequest) -> JobIntakeResponse:
        """Register one job per video path at ``PRESCAN_PENDING`` (Step 3 runs prescan)."""
        output_dir = resolve_output_dir(body.project_id, body.output_dir)
        items: list[JobIntakeItem] = []
        with self._lock:
            for path in body.source_video_paths:
                job_id = f"job_{uuid.uuid4().hex[:12]}"
                job = JobRecord(
                    job_id=job_id,
                    status="PRESCAN_PENDING",
                    source_video_path=path,
                    project_id=body.project_id,
                    output_dir=output_dir,
                    created_at=utc_now_iso(),
                )
                self._jobs[job_id] = job
                self._prescan_queue.append(job_id)
                self._refresh_prescan_queue_positions()
                items.append(
                    JobIntakeItem(
                        job_id=job_id,
                        source_video_path=str(path),
                        output_dir=str(output_dir),
                        queue_position=job.prescan_queue_position,
                    )
                )
        logger.info("jobs_intake", count=len(items), project_id=body.project_id)
        self._drain_prescan()
        return JobIntakeResponse(jobs=items)

    def confirm_prescan(self, job_id: str, body: JobPrescanConfirmRequest) -> JobStatus:
        """PATCH /jobs/{id}/prescan — persist confirmed calibration and enqueue GPU work."""
        job = self.get_job(job_id)
        if job.status not in {"AWAITING_REVIEW", "READY"}:
            bad_request(f"job status {job.status} cannot confirm prescan")
        confirmed_meta = JobMetadata(
            user_start_time=body.metadata.user_start_time,
            user_start_date=body.metadata.user_start_date,
            location=body.metadata.location,
        )
        job.confirmed_metadata = confirmed_meta
        job.confirmed_task_parameters = body.task_parameters
        job.submit = EngineSubmit(
            task_type="ViAna_Moving",
            source_video_path=job.source_video_path,
            project_id=job.project_id,
            output_dir=job.output_dir,
            metadata=confirmed_meta,
            task_parameters=body.task_parameters,
            calibration_profile_id=body.calibration_profile_id,
        )
        job.status = "READY"
        job.error_message = None
        with self._lock:
            if job.job_id in self._prescan_queue:
                self._prescan_queue.remove(job.job_id)
                self._refresh_prescan_queue_positions()
            if job.job_id not in self._queue:
                self._queue.append(job.job_id)
            self._refresh_queue_positions()
        status_response = self.to_status(job)
        self._drain()
        return status_response

    def retry_prescan(self, job_id: str) -> JobStatus:
        """POST /jobs/{id}/prescan/retry — re-queue a failed prescan."""
        job = self.get_job(job_id)
        if job.status != "PRESCAN_FAILED":
            bad_request(f"job status {job.status} cannot retry prescan")
        job.status = "PRESCAN_PENDING"
        job.error_message = None
        with self._lock:
            if job.job_id not in self._prescan_queue:
                self._prescan_queue.append(job.job_id)
            self._refresh_prescan_queue_positions()
        self._drain_prescan()
        return self.to_status(job)

    def prescan_preview(self, job_id: str, frame_offset_sec: float) -> dict[str, Any]:
        """Run prescan at ``frame_offset_sec`` for scrub preview (G8)."""
        job = self.get_job(job_id)
        if job.status not in PRESCAN_PHASE_STATUSES | {"READY"}:
            bad_request(f"job status {job.status} cannot preview prescan frame")
        if frame_offset_sec < 0:
            bad_request("frame_offset_sec must be >= 0")
        args = [
            "prescan",
            "--source",
            str(job.source_video_path),
            "--project-id",
            job.project_id,
            "--frame-offset",
            str(frame_offset_sec),
            "--output-dir",
            str(job.output_dir),
        ]
        logger.info("viana_prescan_preview", job_id=job_id, frame_offset_sec=frame_offset_sec)
        result = run_viana(args, timeout=120.0)
        if result.returncode != 0:
            bad_request(result.stderr.strip() or result.stdout.strip() or "prescan preview failed")
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            bad_request("prescan stdout was not a JSON object")
        return rewrite_preview_url(payload)

    def stub_awaiting_review(
        self,
        job_id: str,
        proposed_metadata: JobMetadata | None = None,
        proposed_lines: ProposedLines | None = None,
        proposed_preview_url: str | None = None,
    ) -> JobStatus:
        """Step 2 test helper — Step 3 prescan worker replaces this transition."""
        job = self.get_job(job_id)
        if job.status not in {"PRESCAN_PENDING", "PRESCAN_RUNNING", "PRESCAN_FAILED"}:
            bad_request(f"job status {job.status} cannot enter AWAITING_REVIEW")
        job.status = "AWAITING_REVIEW"
        job.proposed_metadata = proposed_metadata
        job.proposed_lines = proposed_lines
        job.proposed_preview_url = proposed_preview_url
        job.error_message = None
        with self._lock:
            if job.job_id in self._prescan_queue:
                self._prescan_queue.remove(job.job_id)
                self._refresh_prescan_queue_positions()
        return self.to_status(job)

    def submit(self, body: JobSubmitRequest) -> JobSubmitResponse:
        """Accept POST /jobs: assign ids, 409 on silent resume, enqueue or start."""
        output_dir = resolve_output_dir(body.project_id, body.output_dir)
        stem = body.source_video_path.stem
        ckpt_path = artifact_paths(output_dir, stem)["checkpoint"]
        if ckpt_path.is_file() and not body.resume and not body.start_fresh:
            conflict(CHECKPOINT_CONFLICT)
        if body.resume and not ckpt_path.is_file():
            not_found(f"checkpoint not found: {ckpt_path}")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        command: CommandKind = "resume" if body.resume else "run"
        submit_body = body.model_copy(update={"output_dir": output_dir})
        job = JobRecord(
            job_id=job_id,
            status="READY",
            source_video_path=body.source_video_path,
            project_id=body.project_id,
            output_dir=output_dir,
            submit=submit_body,
            command=command,
            confirmed_metadata=JobMetadata(
                user_start_time=body.metadata.user_start_time,
                user_start_date=body.metadata.user_start_date,
                location=body.metadata.location,
            ),
            confirmed_task_parameters=body.task_parameters,
            created_at=utc_now_iso(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._refresh_queue_positions()
        self._drain()
        with self._lock:
            current = self._jobs[job_id]
        logger.info(
            "job_submitted",
            job_id=job_id,
            status=current.status,
            gpu_device=current.gpu_device,
            command=command,
        )
        return self.to_submit_response(current)

    def resume(self, job_id: str) -> JobSubmitResponse:
        """Explicit resume: ``viana resume`` with resume=true."""
        job = self.get_job(job_id)
        if job.status == "PROCESSING":
            conflict("job is already processing")
        if job.submit is None:
            bad_request("job has no submit payload; confirm prescan first")
        ckpt_path = artifact_paths(job.output_dir, job.source_video_path.stem)["checkpoint"]
        if not ckpt_path.is_file():
            not_found(f"checkpoint not found: {ckpt_path}")
        job.submit = job.submit.model_copy(update={"resume": True, "start_fresh": False})
        job.command = "resume"
        job.status = "READY"
        job.error_message = None
        job.processing_ended_monotonic = None
        with self._lock:
            if job.job_id not in self._queue:
                self._queue.append(job.job_id)
            self._refresh_queue_positions()
        self._drain()
        return self.to_submit_response(self.get_job(job_id))

    def start_fresh(self, job_id: str) -> JobSubmitResponse:
        """Wipe checkpoint via engine start_fresh and ``viana run``."""
        job = self.get_job(job_id)
        if job.status == "PROCESSING":
            conflict("job is already processing")
        if job.submit is None:
            bad_request("job has no submit payload; confirm prescan first")
        job.submit = job.submit.model_copy(update={"resume": False, "start_fresh": True})
        job.command = "run"
        job.status = "READY"
        job.error_message = None
        job.processing_started_monotonic = None
        job.processing_ended_monotonic = None
        with self._lock:
            if job.job_id not in self._queue:
                self._queue.append(job.job_id)
            self._refresh_queue_positions()
        self._drain()
        return self.to_submit_response(self.get_job(job_id))

    def cancel(self, job_id: str) -> None:
        """Cancel queued or running work."""
        job = self.get_job(job_id)
        if job.status in PRESCAN_PHASE_STATUSES:
            with self._lock:
                if job.job_id in self._prescan_queue:
                    self._prescan_queue.remove(job.job_id)
                    self._refresh_prescan_queue_positions()
                job.status = "CANCELLED"
                job.queue_position = 0
            return
        with self._lock:
            if job.job_id in self._queue:
                self._queue.remove(job.job_id)
                job.status = "CANCELLED"
                job.queue_position = 0
                self._refresh_queue_positions()
                return
            proc = job.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return
        job.status = "CANCELLED"

    def wait_job(self, job_id: str, timeout: float = 5.0) -> JobRecord:
        """Block until a job leaves READY/PROCESSING (tests)."""
        import time

        job = self.get_job(job_id)
        end = time.time() + timeout
        while time.time() < end:
            if job.status not in {"READY", "PROCESSING"}:
                return job
            time.sleep(0.02)
        return job

    def wait_for_status(
        self, job_id: str, *statuses: JobStatusLiteral, timeout: float = 5.0
    ) -> JobRecord:
        """Block until job reaches one of ``statuses`` (tests)."""
        import time

        targets = set(statuses)
        job = self.get_job(job_id)
        end = time.time() + timeout
        while time.time() < end:
            if job.status in targets:
                return job
            time.sleep(0.02)
        return job

    def _refresh_prescan_queue_positions(self) -> None:
        """Set 1-based prescan queue index for PRESCAN_PENDING rows."""
        for index, job_id in enumerate(self._prescan_queue, start=1):
            job = self._jobs[job_id]
            job.prescan_queue_position = index

    def _refresh_queue_positions(self) -> None:
        """Set queue_position: 0 if running, 1-based index in the wait queue."""
        for index, job_id in enumerate(self._queue, start=1):
            job = self._jobs[job_id]
            job.queue_position = index
        for job in self._jobs.values():
            if job.status == "PROCESSING":
                job.queue_position = 0

    def _drain(self) -> None:
        """Start queued READY jobs while GPUs are free."""
        while True:
            with self._lock:
                occupied = {
                    j.gpu_device
                    for j in self._jobs.values()
                    if j.status == "PROCESSING" and j.gpu_device is not None
                }
                device = self.assign_gpu(occupied)
                if device is None or not self._queue:
                    return
                job_id = self._queue[0]
                job = self._jobs[job_id]
                if job.status != "READY":
                    return
                self._queue.pop(0)
                job.gpu_device = device
                job.status = "PROCESSING"
                if job.processing_started_monotonic is None:
                    job.processing_started_monotonic = time.monotonic()
                    job.processing_ended_monotonic = None
                job.queue_position = 0
                self._refresh_queue_positions()
            self._spawn(job)

    def _drain_prescan(self) -> None:
        """Start PRESCAN_PENDING jobs up to ``MAX_CONCURRENT_PRESCAN_JOBS``."""
        while True:
            with self._lock:
                running = sum(1 for job in self._jobs.values() if job.status == "PRESCAN_RUNNING")
                if running >= MAX_CONCURRENT_PRESCAN_JOBS:
                    return
                job_id: str | None = None
                for queued_id in self._prescan_queue:
                    queued = self._jobs.get(queued_id)
                    if queued is not None and queued.status == "PRESCAN_PENDING":
                        job_id = queued_id
                        queued.status = "PRESCAN_RUNNING"
                        break
                if job_id is None:
                    return
            threading.Thread(target=self._run_prescan, args=(job_id,), daemon=True).start()

    def _run_prescan(self, job_id: str) -> None:
        """Execute ``viana prescan`` for one intake job (CPU worker)."""
        job = self.get_job(job_id)
        try:
            args = [
                "prescan",
                "--source",
                str(job.source_video_path),
                "--project-id",
                job.project_id,
                "--output-dir",
                str(job.output_dir),
            ]
            logger.info("viana_prescan_worker", job_id=job_id)
            result = run_viana(args, timeout=300.0)
            if result.returncode != 0:
                job.status = "PRESCAN_FAILED"
                job.error_message = (
                    result.stderr.strip() or result.stdout.strip() or "prescan failed"
                )
                return
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict):
                job.status = "PRESCAN_FAILED"
                job.error_message = "prescan stdout was not a JSON object"
                return
            _apply_prescan_payload(job, payload)
            job.status = "AWAITING_REVIEW"
            job.error_message = None
        except TimeoutExpired:
            job.status = "PRESCAN_FAILED"
            job.error_message = "prescan timed out"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            job.status = "PRESCAN_FAILED"
            job.error_message = str(exc)
        finally:
            with self._lock:
                if job.job_id in self._prescan_queue:
                    self._prescan_queue.remove(job.job_id)
                    self._refresh_prescan_queue_positions()
            self._drain_prescan()

    def _auto_aggregate(self, job: JobRecord) -> None:
        """Fire ``viana aggregate`` after COMPLETED (G12)."""

        def run() -> None:
            args = [
                "aggregate",
                "--source",
                str(job.source_video_path),
                "--project-id",
                job.project_id,
                "--output-dir",
                str(job.output_dir),
            ]
            logger.info("viana_auto_aggregate", job_id=job.job_id, args=args)
            try:
                result = run_viana(args, timeout=120.0)
            except OSError as exc:
                logger.error("auto_aggregate_failed", job_id=job.job_id, error=str(exc))
                return
            if result.returncode != 0:
                logger.error(
                    "auto_aggregate_failed",
                    job_id=job.job_id,
                    detail=result.stderr.strip() or result.stdout.strip(),
                )

        threading.Thread(target=run, daemon=True).start()

    def _write_job_config(self, job: JobRecord) -> Path:
        """Persist JobConfig JSON for the CLI."""
        if job.submit is None:
            bad_request("internal: submit payload missing at spawn")
        job.output_dir.mkdir(parents=True, exist_ok=True)
        payload = job.submit.model_dump(mode="json")
        payload["job_id"] = job.job_id
        payload["gpu_device"] = job.gpu_device
        payload["output_dir"] = str(job.output_dir)
        if job.command == "resume":
            payload["resume"] = True
            payload["start_fresh"] = False
        config = JobConfig.model_validate(payload)
        path = job.output_dir / f"{job.job_id}.job.json"
        dumped = json.dumps(config.model_dump(mode="json"), indent=2) + "\n"
        path.write_text(dumped, encoding="utf-8")
        job.config_path = path
        return path

    def _spawn(self, job: JobRecord) -> None:
        """Write JobConfig and Popen ``python -m viana``."""
        if job.gpu_device is None:
            bad_request("internal: gpu_device missing at spawn")
        config_path = self._write_job_config(job)
        args = [job.command, "--config", str(config_path)]
        logger.info("viana_spawn", job_id=job.job_id, args=args, gpu_device=job.gpu_device)
        proc = start_viana_process(args)
        job.process = proc
        thread = threading.Thread(target=self._monitor, args=(job.job_id, proc), daemon=True)
        self._threads.append(thread)
        thread.start()

    def _monitor(self, job_id: str, proc: Popen[str]) -> None:
        """Pump stderr NDJSON to WebSocket; parse stdout RunResult on exit."""
        try:
            if proc.stderr is not None:
                for raw in proc.stderr:
                    line = raw.strip()
                    if not line:
                        continue
                    self._handle_telemetry_line(job_id, line)
            stdout = proc.stdout.read() if proc.stdout is not None else ""
            proc.wait()
            self._finalize(job_id, proc.returncode or 0, stdout)
        except (OSError, ValueError) as exc:
            logger.error("worker_monitor_failed", job_id=job_id, error=str(exc))
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = "FAILED"
                    job.error_message = str(exc)
                    job.process = None
                    _mark_processing_ended(job)
        finally:
            self._drain()

    def _handle_telemetry_line(self, job_id: str, line: str) -> None:
        """Parse one stderr line; ignore non-JSON engine logs."""
        try:
            payload: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or "telemetry_type" not in payload:
            return
        payload.setdefault("job_id", job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                payload.setdefault("status", job.status)
                data = payload.get("data")
                if payload.get("telemetry_type") == "MOVING_EVENT":
                    job.crossing_count += 1
                if isinstance(data, dict):
                    if payload.get("telemetry_type") == "PROGRESS":
                        _sync_progress(job, data)
                        payload["data"] = data
        hub.publish(payload)

    def _finalize(self, job_id: str, returncode: int, stdout: str) -> None:
        """Map RunResult / exit code onto the job state machine."""
        aggregate_target: JobRecord | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.process = None
            ckpt_path = artifact_paths(job.output_dir, job.source_video_path.stem)["checkpoint"]
            result = _parse_run_result(stdout)
            terminal = False
            if result is not None:
                status = result.get("status")
                if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                    job.status = status
                    err = result.get("error_message")
                    job.error_message = err if isinstance(err, str) else None
                    if status == "COMPLETED":
                        aggregate_target = job
                    terminal = True
            if not terminal:
                if returncode < 0:
                    if ckpt_path.is_file():
                        try:
                            checkpoint = load_checkpoint(ckpt_path)
                        except (OSError, ValueError):
                            checkpoint = None
                        if checkpoint is not None and not checkpoint.is_complete():
                            job.status = "PAUSED"
                            job.error_message = "worker cancelled"
                            _mark_processing_ended(job)
                            return
                    job.status = "CANCELLED"
                    _mark_processing_ended(job)
                    return
                if returncode != 0:
                    job.status = "FAILED"
                    job.error_message = stdout.strip() or f"viana exited {returncode}"
                    if ckpt_path.is_file():
                        try:
                            checkpoint = load_checkpoint(ckpt_path)
                        except (OSError, ValueError):
                            checkpoint = None
                        if checkpoint is not None and not checkpoint.is_complete():
                            job.status = "PAUSED"
                    _mark_processing_ended(job)
                    return
                job.status = "COMPLETED"
                aggregate_target = job
            _mark_processing_ended(job)
        if aggregate_target is not None:
            self._auto_aggregate(aggregate_target)


def _compute_eta_sec(current: int, total: int, fps: float | None) -> float | None:
    """Wall-clock ETA: ``(total_frames - current_frame) / processing_fps``.

    ``fps`` is engine *processing* throughput (frames per wall-clock second),
    never source ``video_meta.fps``. Units: frames / (frames/sec) = seconds.
    """
    if fps is None or fps <= 0:
        return None
    remaining = max(0, total - current)
    return round(remaining / fps, 1)


def _sync_progress(job: JobRecord, data: dict[str, Any]) -> None:
    """Update job progress and enrich telemetry data with ETA + crossings (G9)."""
    current = data.get("current_frame")
    total = data.get("total_frames")
    fps = data.get("processing_fps")
    crossing = data.get("crossing_count")
    if isinstance(crossing, int):
        job.crossing_count = crossing
    if isinstance(current, int) and isinstance(total, int):
        fps_val = fps if isinstance(fps, int | float) else None
        eta = _compute_eta_sec(current, total, float(fps_val) if fps_val is not None else None)
        if eta is not None:
            data["eta_sec"] = eta
        data["crossing_count"] = job.crossing_count
        job.progress = JobProgress(
            current_frame=current,
            total_frames=total,
            processing_fps=float(fps_val) if fps_val is not None else None,
            eta_sec=eta,
            crossing_count=job.crossing_count,
        )


def _processing_duration_sec(job: JobRecord) -> float | None:
    """Return elapsed processing wall-clock seconds when a GPU run has started."""
    start = job.processing_started_monotonic
    if start is None:
        return None
    end = (
        job.processing_ended_monotonic
        if job.processing_ended_monotonic is not None
        else time.monotonic()
    )
    return round(max(0.0, end - start), 2)


def _mark_processing_ended(job: JobRecord) -> None:
    """Freeze processing_duration_sec once a GPU run leaves PROCESSING."""
    if job.processing_started_monotonic is None:
        return
    if job.processing_ended_monotonic is None:
        job.processing_ended_monotonic = time.monotonic()


def _apply_prescan_payload(job: JobRecord, payload: dict[str, Any]) -> None:
    """Map ``PrescanResponse`` JSON onto job proposal fields."""
    ocr = payload.get("ocr")
    if isinstance(ocr, dict):
        job.proposed_metadata = JobMetadata(
            user_start_time=ocr.get("time"),
            user_start_date=ocr.get("date"),
            location=ocr.get("location"),
        )
    lines = payload.get("proposed_lines")
    if isinstance(lines, dict):
        job.proposed_lines = ProposedLines.model_validate(lines)
    prescan_id = payload.get("prescan_id")
    disk = payload.get("preview_url")
    if isinstance(prescan_id, str) and isinstance(disk, str) and disk:
        register_preview(prescan_id, Path(disk))
        job.proposed_preview_url = preview_http_url(prescan_id)
    meta = payload.get("video_meta")
    if isinstance(meta, dict):
        duration = meta.get("duration_sec")
        if isinstance(duration, int | float) and duration >= 0:
            job.video_duration_sec = float(duration)


def _parse_run_result(stdout: str) -> dict[str, Any] | None:
    """Extract a JSON object from CLI stdout."""
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


_pool: WorkerPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> WorkerPool:
    """Return the process-wide worker pool."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = WorkerPool()
        return _pool


def reset_pool() -> WorkerPool:
    """Replace the pool (tests / lifespan startup)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown()
        _pool = WorkerPool()
        return _pool
