"""Step 6.2 — operator pause → PAUSED; resume; cancel must not fight pause."""

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


def _start_processing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, list[HoldPopen]]:
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        proc.stdout = proc.stdout  # type: ignore[misc]
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    created = client.post("/jobs", json=VALID_SUBMIT)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    get_pool().wait_for_status(job_id, "PROCESSING", timeout=5.0)
    return job_id, holds


def _write_checkpoint(client: TestClient, job_id: str) -> Path:
    output_dir = Path(client.get(f"/jobs/{job_id}").json()["output_dir"])
    ckpt = artifact_paths(output_dir, STEM)["checkpoint"]
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
    return ckpt


def test_pause_processing_becomes_paused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, holds = _start_processing(client, monkeypatch)
    _write_checkpoint(client, job_id)

    paused = client.post(f"/jobs/{job_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] in {"PROCESSING", "PAUSED"}

    pool = get_pool()
    pool.wait_for_status(job_id, "PAUSED", timeout=5.0)
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "PAUSED"
    assert body["checkpoint_exists"] is True
    assert body["error_message"] == "interrupted"
    assert body["gpu_device"] is None
    assert pool.occupied_gpus() == set()
    assert len(holds) == 1


def test_cancel_while_processing_stays_cancelled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, _holds = _start_processing(client, monkeypatch)
    _write_checkpoint(client, job_id)

    deleted = client.delete(f"/jobs/{job_id}")
    assert deleted.status_code == 204
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"

    pool = get_pool()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pool.get_job(job_id).status == "CANCELLED":
            break
        time.sleep(0.05)
    assert pool.get_job(job_id).status == "CANCELLED"


def test_pause_then_resume(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, holds = _start_processing(client, monkeypatch)
    _write_checkpoint(client, job_id)
    client.post(f"/jobs/{job_id}/pause")
    get_pool().wait_for_status(job_id, "PAUSED", timeout=5.0)

    resumed = client.post(f"/jobs/{job_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "PROCESSING"
    assert len(holds) == 2


def test_pause_rejected_when_not_processing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, holds = _start_processing(client, monkeypatch)
    _write_checkpoint(client, job_id)
    client.post(f"/jobs/{job_id}/pause")
    get_pool().wait_for_status(job_id, "PAUSED", timeout=5.0)

    again = client.post(f"/jobs/{job_id}/pause")
    assert again.status_code == 409


def test_resume_requires_paused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, _holds = _start_processing(client, monkeypatch)
    blocked = client.post(f"/jobs/{job_id}/resume")
    assert blocked.status_code == 409
