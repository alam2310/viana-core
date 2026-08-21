"""S27 / F022 — drain the next READY job after a GPU worker FAILED."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.workers.pool import JobRecord, get_pool, reset_pool
from tests.orchestrator.test_job_routes import (
    SOURCE,
    VALID_SUBMIT,
    HoldPopen,
    _HoldStderr,
    _run_result_json,
)


def _failed_run_result_json(job_id: str) -> str:
    payload = json.loads(_run_result_json(job_id))
    payload["status"] = "FAILED"
    payload["error_message"] = "engine boom"
    return json.dumps(payload)


class FailHoldPopen:
    """Occupies a GPU until fail(), then exits with RunResult status FAILED."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.stdin = None
        self.stdout = io.StringIO(_failed_run_result_json("job_fail"))
        self.stderr = _HoldStderr(self._done)
        self.returncode: int | None = None
        self.pid = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        finished = self._done.wait(timeout)
        if not finished and timeout is not None and self.returncode is None:
            raise TimeoutExpired("fail-hold", timeout)
        if self.returncode is None:
            self.returncode = 1
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._done.set()

    def kill(self) -> None:
        self.returncode = -9
        self._done.set()

    def fail(self) -> None:
        self.returncode = 1
        self._done.set()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))
    monkeypatch.setattr("orchestrator.workers.pool.GPU_DEVICES", ("cuda:0",))

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
    reset_pool()
    with TestClient(app) as test_client:
        yield test_client
    reset_pool()


def test_failed_job_starts_next_ready_fifo(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job A FAILED must free the GPU and auto-start job B with no operator action."""
    procs: list[FailHoldPopen | HoldPopen] = []

    def start(_args: object) -> FailHoldPopen | HoldPopen:
        proc: FailHoldPopen | HoldPopen
        if not procs:
            proc = FailHoldPopen()
        else:
            proc = HoldPopen()
        procs.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    first = client.post("/jobs", json=VALID_SUBMIT)
    second = client.post("/jobs", json=VALID_SUBMIT)
    assert first.status_code == 201
    assert second.status_code == 201
    job_a = first.json()["job_id"]
    job_b = second.json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_a, "PROCESSING", timeout=5.0)
    assert client.get(f"/jobs/{job_b}").json()["status"] == "READY"
    assert pool.occupied_gpus() == {"cuda:0"}

    assert isinstance(procs[0], FailHoldPopen)
    procs[0].fail()
    failed = pool.wait_for_status(job_a, "FAILED", timeout=5.0)
    assert failed.status == "FAILED"
    assert failed.process is None
    assert failed.gpu_device is None

    started = pool.wait_for_status(job_b, "PROCESSING", timeout=5.0)
    assert started.status == "PROCESSING"
    assert started.gpu_device == "cuda:0"
    assert pool.occupied_gpus() == {"cuda:0"}
    assert len(procs) == 2

    assert isinstance(procs[1], HoldPopen)
    procs[1].release()
    pool.wait_job(job_b, timeout=5.0)
    assert client.get(f"/jobs/{job_b}").json()["status"] == "COMPLETED"


def test_drain_skips_stale_queue_head(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-READY queue heads must not block the next READY job."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    first = client.post("/jobs", json=VALID_SUBMIT)
    second = client.post("/jobs", json=VALID_SUBMIT)
    assert first.status_code == 201
    assert second.status_code == 201
    job_a = first.json()["job_id"]
    job_b = second.json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_a, "PROCESSING", timeout=5.0)
    assert client.get(f"/jobs/{job_b}").json()["status"] == "READY"

    stale = JobRecord(
        job_id="job_stale_s27",
        status="FAILED",
        source_video_path=Path(SOURCE),
        project_id="nh48",
        output_dir=pool.get_job(job_a).output_dir,
    )
    with pool._lock:
        pool._jobs[stale.job_id] = stale
        pool._queue.insert(0, stale.job_id)

    holds[0].release()
    pool.wait_job(job_a, timeout=5.0)
    started = pool.wait_for_status(job_b, "PROCESSING", timeout=5.0)
    assert started.status == "PROCESSING"
    assert stale.job_id not in pool._queue
    holds[1].release()
    pool.wait_job(job_b, timeout=5.0)


def test_drain_skips_missing_queue_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A queue entry with no job record must not stall FIFO drain."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    first = client.post("/jobs", json=VALID_SUBMIT)
    second = client.post("/jobs", json=VALID_SUBMIT)
    job_a = first.json()["job_id"]
    job_b = second.json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_a, "PROCESSING", timeout=5.0)
    with pool._lock:
        pool._queue.insert(0, "job_ghost_s27")
    holds[0].release()
    pool.wait_for_status(job_b, "PROCESSING", timeout=5.0)
    holds[1].release()
    pool.wait_job(job_b, timeout=5.0)
