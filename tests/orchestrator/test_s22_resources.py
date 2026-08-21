"""S22 — orchestrator worker stdio and thread lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.workers.pool import get_pool, reset_pool
from tests.orchestrator.test_job_routes import SOURCE, VALID_SUBMIT, InstantPopen, _run_result_json
from viana.io.proc import open_fd_count


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))

    def fake_run_viana(
        args: Any, timeout: float | None = None, **kwargs: Any
    ) -> CompletedProcess[str]:
        on_spawn = kwargs.get("on_spawn")
        if on_spawn is not None:
            on_spawn(InstantPopen("{}", "", 0))
        if args and args[0] == "prescan":
            payload = (
                '{"prescan_id":"prescan_s22","video_meta":'
                '{"width":64,"height":64,"fps":15,"duration_sec":1,"frame_count":15},'
                '"ocr":{"time":null,"date":null,"location":null,"confidence":null},'
                '"proposed_lines":null,"preview_url":null,"profiles":[]}'
            )
            return CompletedProcess(args=list(args), returncode=0, stdout=payload, stderr="")
        if args and args[0] == "aggregate":
            return CompletedProcess(
                args=list(args),
                returncode=0,
                stdout='{"status":"ok","command":"aggregate","rows":0}',
                stderr="",
            )
        raise AssertionError(f"unexpected run_viana call: {args}")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run_viana)
    monkeypatch.setattr(
        "orchestrator.workers.pool.start_viana_process",
        lambda *_a, **_k: InstantPopen(_run_result_json("job_s22"), ""),
    )
    reset_pool()
    with TestClient(app) as test_client:
        yield test_client
    reset_pool()


def test_two_file_intake_does_not_grow_open_fds(client: TestClient) -> None:
    """Multi-file intake (S22 repro) must not accumulate orchestrator FDs."""
    baseline = open_fd_count()
    if baseline is None:
        pytest.skip("/proc/self/fd not available")
    counts: list[int] = []
    for _ in range(3):
        response = client.post(
            "/jobs/intake",
            json={
                "project_id": "nh48",
                "source_video_paths": [SOURCE, SOURCE.replace(".mp4", "-b.mp4")],
            },
        )
        assert response.status_code == 201
        assert len(response.json()["jobs"]) == 2
        pool = get_pool()
        for item in response.json()["jobs"]:
            pool.wait_for_status(item["job_id"], "AWAITING_REVIEW", timeout=5.0)
        fd_now = open_fd_count()
        assert fd_now is not None
        counts.append(fd_now)
    assert max(counts) <= baseline + 16
    assert counts[-1] <= counts[0] + 8


def test_prescan_cancel_keeps_cancelled_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel during PRESCAN_RUNNING must stay CANCELLED after the worker returns."""
    started = threading.Event()
    release = threading.Event()

    def slow_prescan(
        args: Any, timeout: float | None = None, **kwargs: Any
    ) -> CompletedProcess[str]:
        on_spawn = kwargs.get("on_spawn")
        if on_spawn is not None:
            on_spawn(InstantPopen("{}", "", 0))
        if args and args[0] != "prescan":
            return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")
        started.set()
        release.wait(timeout=5.0)
        return CompletedProcess(
            args=list(args),
            returncode=0,
            stdout=(
                '{"prescan_id":"prescan_x","video_meta":'
                '{"width":64,"height":64,"fps":15,"duration_sec":1,"frame_count":15},'
                '"ocr":{},"proposed_lines":null,"preview_url":null,"profiles":[]}'
            ),
            stderr="",
        )

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", slow_prescan)
    reset_pool()
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert intake.status_code == 201
    job_id = intake.json()["jobs"][0]["job_id"]
    assert started.wait(timeout=2.0)
    cancel = client.delete(f"/jobs/{job_id}")
    assert cancel.status_code == 204
    release.set()
    time.sleep(0.2)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"


def test_gpu_monitor_clears_process_after_complete(client: TestClient) -> None:
    """GPU worker must drop Popen after COMPLETED so pipes are not retained."""
    created = client.post("/jobs", json=VALID_SUBMIT)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    job = get_pool().wait_job(job_id, timeout=5.0)
    assert job.status == "COMPLETED"
    assert job.process is None
