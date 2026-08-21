"""Step 6.3 — DELETE sets CANCELLED immediately without breaking S27 drain."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.workers.pool import get_pool, reset_pool
from tests.orchestrator.test_job_routes import (
    SOURCE,
    STEM,
    VALID_SUBMIT,
    HoldPopen,
)
from viana.io.checkpoint import Checkpoint, save_checkpoint
from viana.io.paths import artifact_paths


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


def test_delete_processing_is_cancelled_immediately(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET after DELETE must be CANCELLED without waiting for worker finalize."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    created = client.post("/jobs", json=VALID_SUBMIT)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_id, "PROCESSING", timeout=5.0)

    deleted = client.delete(f"/jobs/{job_id}")
    assert deleted.status_code == 204
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "CANCELLED"
    assert body["gpu_device"] is None
    job = pool.get_job(job_id)
    assert job.status == "CANCELLED"
    assert job.gpu_device is None
    assert pool.occupied_gpus() == set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        current = pool.get_job(job_id)
        if current.process is None and holds[0].returncode is not None:
            break
        time.sleep(0.02)
    assert holds[0].returncode == -15
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"


def test_delete_processing_does_not_become_completed_from_stdout(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker stdout COMPLETED must not overwrite a user cancel."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    job_id = client.post("/jobs", json=VALID_SUBMIT).json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_id, "PROCESSING", timeout=5.0)
    client.delete(f"/jobs/{job_id}")
    pool.wait_job(job_id, timeout=5.0)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"


def test_delete_processing_stays_cancelled_when_checkpoint_exists(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User cancel is CANCELLED even if a mid-run checkpoint would map SIGTERM to PAUSED."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    job_id = client.post("/jobs", json=VALID_SUBMIT).json()["job_id"]
    pool = get_pool()
    job = pool.wait_for_status(job_id, "PROCESSING", timeout=5.0)
    ckpt = artifact_paths(job.output_dir, STEM)["checkpoint"]
    save_checkpoint(
        ckpt,
        Checkpoint(
            job_id=job_id,
            project_id="nh48",
            source_video_path=Path(SOURCE),
            video_stem=STEM,
            current_frame=10,
            total_frames=100,
            saved_at="2026-03-15T10:00:00Z",
        ),
    )
    client.delete(f"/jobs/{job_id}")
    pool.wait_job(job_id, timeout=5.0)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"


def test_delete_processing_drains_next_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel of the GPU job must start the next FIFO READY job (S27 occupancy)."""
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
    assert client.get(f"/jobs/{job_b}").json()["status"] == "READY"

    client.delete(f"/jobs/{job_a}")
    assert client.get(f"/jobs/{job_a}").json()["status"] == "CANCELLED"
    started = pool.wait_for_status(job_b, "PROCESSING", timeout=5.0)
    assert started.status == "PROCESSING"
    assert started.gpu_device == "cuda:0"
    assert pool.occupied_gpus() == {"cuda:0"}
    assert len(holds) == 2

    holds[1].release()
    pool.wait_job(job_b, timeout=5.0)
    assert client.get(f"/jobs/{job_b}").json()["status"] == "COMPLETED"
    assert client.get(f"/jobs/{job_a}").json()["status"] == "CANCELLED"
