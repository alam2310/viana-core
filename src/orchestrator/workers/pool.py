"""GPU worker pool: max two concurrent ``python -m viana`` processes."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen  # nosec B404
from typing import Any, Literal

from orchestrator.cli import start_viana_process
from orchestrator.errors import bad_request, conflict, not_found
from orchestrator.hub import hub
from orchestrator.logging_config import get_logger
from orchestrator.models import (
    JobProgress,
    JobStatus,
    JobStatusLiteral,
    JobSubmitRequest,
    JobSubmitResponse,
)
from orchestrator.settings import project_dir
from viana.config.job import JobConfig
from viana.io.checkpoint import load_checkpoint
from viana.io.paths import artifact_paths

logger = get_logger(__name__)

MAX_CONCURRENT_GPU_JOBS = 2
GPU_DEVICES: tuple[str, str] = ("cuda:0", "cuda:1")

CHECKPOINT_CONFLICT = (
    "checkpoint exists; POST /jobs/{id}/resume or set start_fresh=true (no silent resume)"
)

CommandKind = Literal["run", "resume"]


@dataclass
class JobRecord:
    """In-memory job lifecycle record."""

    job_id: str
    status: JobStatusLiteral
    source_video_path: Path
    project_id: str
    output_dir: Path
    submit: JobSubmitRequest
    gpu_device: str | None = None
    queue_position: int = 0
    config_path: Path | None = None
    process: Popen[str] | None = None
    progress: JobProgress | None = None
    error_message: str | None = None
    command: CommandKind = "run"


class WorkerPool:
    """Assigns at most two GPUs and spawns ``python -m viana`` workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: list[str] = []
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

    def submit(self, body: JobSubmitRequest) -> JobSubmitResponse:
        """Accept POST /jobs: assign ids, 409 on silent resume, enqueue or start."""
        output_dir = project_dir(body.project_id)
        stem = body.source_video_path.stem
        ckpt_path = artifact_paths(output_dir, stem)["checkpoint"]
        if ckpt_path.is_file() and not body.resume and not body.start_fresh:
            conflict(CHECKPOINT_CONFLICT)
        if body.resume and not ckpt_path.is_file():
            not_found(f"checkpoint not found: {ckpt_path}")

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        command: CommandKind = "resume" if body.resume else "run"
        job = JobRecord(
            job_id=job_id,
            status="PENDING",
            source_video_path=body.source_video_path,
            project_id=body.project_id,
            output_dir=output_dir,
            submit=body,
            command=command,
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
        ckpt_path = artifact_paths(job.output_dir, job.source_video_path.stem)["checkpoint"]
        if not ckpt_path.is_file():
            not_found(f"checkpoint not found: {ckpt_path}")
        job.submit = job.submit.model_copy(update={"resume": True, "start_fresh": False})
        job.command = "resume"
        job.status = "PENDING"
        job.error_message = None
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
        job.submit = job.submit.model_copy(update={"resume": False, "start_fresh": True})
        job.command = "run"
        job.status = "PENDING"
        job.error_message = None
        with self._lock:
            if job.job_id not in self._queue:
                self._queue.append(job.job_id)
            self._refresh_queue_positions()
        self._drain()
        return self.to_submit_response(self.get_job(job_id))

    def cancel(self, job_id: str) -> None:
        """Cancel queued or running work."""
        job = self.get_job(job_id)
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
        """Block until a job leaves PROCESSING/PENDING (tests)."""
        import time

        job = self.get_job(job_id)
        end = time.time() + timeout
        while time.time() < end:
            if job.status not in {"PENDING", "PROCESSING"}:
                return job
            time.sleep(0.02)
        return job

    def _refresh_queue_positions(self) -> None:
        """Set queue_position: 0 if running, 1-based index in the wait queue."""
        for index, job_id in enumerate(self._queue, start=1):
            job = self._jobs[job_id]
            job.queue_position = index
        for job in self._jobs.values():
            if job.status == "PROCESSING":
                job.queue_position = 0

    def _drain(self) -> None:
        """Start queued jobs while GPUs are free."""
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
                job_id = self._queue.pop(0)
                job = self._jobs[job_id]
                job.gpu_device = device
                job.status = "PROCESSING"
                job.queue_position = 0
                self._refresh_queue_positions()
            self._spawn(job)

    def _write_job_config(self, job: JobRecord) -> Path:
        """Persist JobConfig JSON for the CLI."""
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
                if payload.get("telemetry_type") == "PROGRESS" and isinstance(data, dict):
                    current = data.get("current_frame")
                    total = data.get("total_frames")
                    fps = data.get("processing_fps")
                    if isinstance(current, int) and isinstance(total, int):
                        fps_val = fps if isinstance(fps, int | float) else None
                        job.progress = JobProgress(
                            current_frame=current,
                            total_frames=total,
                            processing_fps=float(fps_val) if fps_val is not None else None,
                        )
        hub.publish(payload)

    def _finalize(self, job_id: str, returncode: int, stdout: str) -> None:
        """Map RunResult / exit code onto the job state machine."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.process = None
            ckpt_path = artifact_paths(job.output_dir, job.source_video_path.stem)["checkpoint"]
            result = _parse_run_result(stdout)
            if result is not None:
                status = result.get("status")
                if status in {"COMPLETED", "FAILED", "CANCELLED"}:
                    job.status = status
                    err = result.get("error_message")
                    job.error_message = err if isinstance(err, str) else None
                    return
            if returncode < 0:
                if ckpt_path.is_file():
                    try:
                        checkpoint = load_checkpoint(ckpt_path)
                    except (OSError, ValueError):
                        checkpoint = None
                    if checkpoint is not None and not checkpoint.is_complete():
                        job.status = "PAUSED"
                        job.error_message = "worker cancelled"
                        return
                job.status = "CANCELLED"
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
                return
            job.status = "COMPLETED"


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
