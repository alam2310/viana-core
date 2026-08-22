"""Concurrent ``_drain`` must not assign the same GPU twice."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.models import JobProgress
from orchestrator.workers.pool import JobRecord, WorkerPool, get_pool, reset_pool
from tests.orchestrator.test_job_routes import SOURCE, VALID_SUBMIT, HoldPopen
from tests.orchestrator.test_s62_pause import _write_checkpoint
from viana.config.job import JobMetadata, LineSegment, ViAnaTaskParameters


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))
    monkeypatch.setattr("orchestrator.workers.pool.GPU_DEVICES", ("cuda:0", "cuda:1"))

    def fake_run_viana(
        args: Any, timeout: float | None = None, **_kwargs: Any
    ) -> CompletedProcess[str]:
        if args and args[0] == "aggregate":
            return CompletedProcess(
                args=list(args),
                returncode=0,
                stdout='{"status":"ok","command":"aggregate","rows":0}',
                stderr="",
            )
        raise AssertionError(f"unexpected run_viana call: {args}")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run_viana)

    def fake_interrupt(proc: object) -> None:
        interrupt = getattr(proc, "interrupt", None)
        if callable(interrupt):
            interrupt()
        else:
            terminate = getattr(proc, "terminate", None)
            if callable(terminate):
                terminate()

    monkeypatch.setattr("orchestrator.workers.pool.interrupt_process_tree", fake_interrupt)
    reset_pool()
    with TestClient(app) as test_client:
        yield test_client
    reset_pool()


def _ready_job(job_id: str, output_dir: Path) -> JobRecord:
    meta = JobMetadata(
        user_start_time="09:00:00",
        user_start_date="15-03-2026",
        location="test",
    )
    lines = ViAnaTaskParameters(
        horizon_line=LineSegment(start=(0, 0), end=(10, 10)),
        counting_line=LineSegment(start=(0, 20), end=(10, 20)),
    )
    return JobRecord(
        job_id=job_id,
        status="READY",
        source_video_path=Path(SOURCE),
        project_id="nh48",
        output_dir=output_dir,
        confirmed_metadata=meta,
        confirmed_task_parameters=lines,
    )


def test_parallel_drain_never_double_books_same_gpu(tmp_path: Path) -> None:
    """Two threads calling ``_drain`` reserve at most one job per GPU."""
    reset_pool()
    pool = get_pool()
    output_dir = tmp_path / "nh48"
    output_dir.mkdir(parents=True)
    reserved: list[str] = []

    def fake_spawn(job: JobRecord) -> None:
        reserved.append(job.gpu_device or "")

    pool._spawn = fake_spawn  # type: ignore[method-assign]

    with pool._lock:
        for index in range(3):
            job_id = f"job_ready_{index}"
            pool._jobs[job_id] = _ready_job(job_id, output_dir)
            pool._queue.append(job_id)
        pool._refresh_queue_positions()
        pool._occupied_gpus.clear()

    barrier = threading.Barrier(2)

    def burst_drain() -> None:
        barrier.wait(timeout=2.0)
        pool._drain()

    threads = [threading.Thread(target=burst_drain) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(reserved) == 2
    assert len(set(reserved)) == 2
    assert set(reserved) == {"cuda:0", "cuda:1"}


def test_ready_clears_progress_and_api_omits_it() -> None:
    """READY rows must not carry prior run progress in memory or GET payloads."""
    pool = WorkerPool()
    job = JobRecord(
        job_id="job_prog",
        status="PAUSED",
        source_video_path=Path(SOURCE),
        project_id="nh48",
        output_dir=Path("/tmp/nh48"),
        progress=JobProgress(current_frame=50, total_frames=100, crossing_count=3),
    )
    pool._set_job_status(job, "READY")
    assert job.progress is None
    assert pool.to_status(job).progress is None


def test_ready_status_omits_stale_progress_after_resume(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume re-queue (READY, GPUs full) must not expose prior run progress."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()

    job_a = client.post("/jobs", json=VALID_SUBMIT).json()["job_id"]
    job_b = client.post("/jobs", json=VALID_SUBMIT).json()["job_id"]
    job_c = client.post("/jobs", json=VALID_SUBMIT).json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_a, "PROCESSING", timeout=5.0)
    pool.wait_for_status(job_b, "PROCESSING", timeout=5.0)
    assert pool.get_job(job_c).status == "READY"

    _write_checkpoint(client, job_a)
    client.post(f"/jobs/{job_a}/pause")
    pool.wait_for_status(job_a, "PAUSED", timeout=5.0)
    pool.wait_for_status(job_c, "PROCESSING", timeout=5.0)

    paused = client.get(f"/jobs/{job_a}").json()
    assert paused["status"] == "PAUSED"
    assert paused["progress"] is not None

    resumed = client.post(f"/jobs/{job_a}/resume")
    assert resumed.status_code == 200
    body = client.get(f"/jobs/{job_a}").json()
    assert body["status"] == "READY"
    assert body["progress"] is None
    assert body["gpu_device"] is None
