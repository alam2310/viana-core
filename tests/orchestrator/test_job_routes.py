"""Phase 6 orchestrator tests — mocked ``python -m viana`` subprocesses."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.workers.pool import (
    CHECKPOINT_CONFLICT,
    GPU_DEVICES,
    MAX_CONCURRENT_GPU_JOBS,
    WorkerPool,
    get_pool,
    reset_pool,
)
from viana.io.checkpoint import Checkpoint, save_checkpoint
from viana.io.paths import artifact_paths, project_output_dir

VALID_SUBMIT = {
    "task_type": "ViAna_Moving",
    "source_video_path": "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
    "project_id": "nh48",
    "metadata": {
        "user_start_time": "09:00:00",
        "user_start_date": "15-03-2026",
        "location": "NH48 Km42",
    },
    "task_parameters": {
        "horizon_line": {"start": [120, 400], "end": [1800, 520]},
        "counting_line": {"start": [80, 900], "end": [1840, 780]},
        "confidence_threshold": 0.75,
        "use_heuristic_truck_split": True,
        "render_video": True,
        "telemetry_detail": False,
    },
    "calibration_profile_id": "morning_northbound",
    "resume": False,
    "start_fresh": False,
}

VALID_PROFILE = {
    "profile_id": "morning_northbound",
    "profile_name": "Morning northbound",
    "reference_resolution": [1920, 1080],
    "horizon_line": {"start": [120, 400], "end": [1800, 520]},
    "counting_line": {"start": [80, 900], "end": [1840, 780]},
    "source": "user_drawn",
}

STEM = "2026-03-15_09-00"
SOURCE = str(VALID_SUBMIT["source_video_path"])


class InstantPopen:
    """Process that already finished with RunResult stdout and telemetry stderr."""

    def __init__(self, stdout: str, stderr: str, returncode: int = 0) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.pid = 4242

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class HoldPopen:
    """Process that occupies a GPU until terminate() or release()."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.stdout = io.StringIO(_run_result_json("job_hold"))
        self.stderr = _HoldStderr(self._done)
        self.returncode: int | None = None
        self.pid = 4343

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._done.set()

    def kill(self) -> None:
        self.returncode = -9
        self._done.set()

    def release(self) -> None:
        """Let the fake worker finish successfully."""
        self.returncode = 0
        self._done.set()


class _HoldStderr:
    def __init__(self, done: threading.Event) -> None:
        self._done = done
        self._sent = False

    def __iter__(self) -> _HoldStderr:
        return self

    def __next__(self) -> str:
        if not self._sent:
            self._sent = True
            return _progress_line() + "\n"
        self._done.wait()
        raise StopIteration


def _run_result_json(job_id: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "job_id": job_id,
            "status": "COMPLETED",
            "source_video_path": SOURCE,
            "video_stem": STEM,
            "artifacts": {},
            "completed_at": "2026-03-15T10:00:00Z",
            "error_message": None,
        }
    )


def _progress_line() -> str:
    return json.dumps(
        {
            "job_id": "pending",
            "status": "PROCESSING",
            "telemetry_type": "PROGRESS",
            "data": {"current_frame": 10, "total_frames": 100, "processing_fps": 21.5},
        }
    )


def _instant_popen(*_args: object, **_kwargs: object) -> InstantPopen:
    return InstantPopen(_run_result_json("job_mock"), _progress_line() + "\n")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Isolated pool + output dir; default spawn finishes immediately."""
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))
    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", _instant_popen)
    reset_pool()
    with TestClient(app) as test_client:
        yield test_client
    reset_pool()


def test_health_still_ok(client: TestClient) -> None:
    """Health probe reports Phase 6."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": 6}


def test_cors_allows_ui_origin(client: TestClient) -> None:
    """Browser dashboard on :3000 must call the orchestrator on :8000."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_post_jobs_rejects_client_job_id(client: TestClient) -> None:
    """UI must not send job_id; extra fields are forbidden."""
    body = {**VALID_SUBMIT, "job_id": "job_from_ui"}
    response = client.post("/jobs", json=body)
    assert response.status_code == 422


def test_post_jobs_rejects_client_gpu_device(client: TestClient) -> None:
    """UI must not send gpu_device; backend assigns cuda:0 or cuda:1."""
    body = {**VALID_SUBMIT, "gpu_device": "cuda:0"}
    response = client.post("/jobs", json=body)
    assert response.status_code == 422


def test_post_jobs_assigns_backend_fields(client: TestClient) -> None:
    """POST /jobs returns JobSubmitResponse and shells viana (mocked)."""
    response = client.post("/jobs", json=VALID_SUBMIT)
    assert response.status_code == 201
    body = response.json()
    assert body["gpu_device"] in GPU_DEVICES
    assert body["job_id"].startswith("job_")
    assert "output_dir" in body
    job_id = body["job_id"]
    get_pool().wait_job(job_id)
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] == "COMPLETED"
    assert payload["project_id"] == "nh48"
    listed = client.get("/jobs?project_id=nh48")
    assert listed.status_code == 200
    assert any(item["job_id"] == job_id for item in listed.json())


def test_post_jobs_409_on_incomplete_checkpoint(client: TestClient, tmp_path: Path) -> None:
    """Plain submit must not silently resume."""
    output_dir = project_output_dir(tmp_path, "nh48")
    ckpt = artifact_paths(output_dir, STEM)["checkpoint"]
    save_checkpoint(
        ckpt,
        Checkpoint(
            job_id="job_old",
            project_id="nh48",
            source_video_path=Path(SOURCE),
            video_stem=STEM,
            current_frame=10,
            total_frames=100,
            saved_at="2026-03-15T10:00:00Z",
        ),
    )
    response = client.post("/jobs", json=VALID_SUBMIT)
    assert response.status_code == 409
    assert CHECKPOINT_CONFLICT in response.json()["detail"]


def test_start_fresh_allowed_with_checkpoint(client: TestClient, tmp_path: Path) -> None:
    """start_fresh=true bypasses 409 and spawns viana run."""
    output_dir = project_output_dir(tmp_path, "nh48")
    ckpt = artifact_paths(output_dir, STEM)["checkpoint"]
    save_checkpoint(
        ckpt,
        Checkpoint(
            job_id="job_old",
            project_id="nh48",
            source_video_path=Path(SOURCE),
            video_stem=STEM,
            current_frame=10,
            total_frames=100,
            saved_at="2026-03-15T10:00:00Z",
        ),
    )
    body = {**VALID_SUBMIT, "start_fresh": True}
    response = client.post("/jobs", json=body)
    assert response.status_code == 201


def test_queue_and_cancel_pending(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Third job waits; DELETE cancels it without a GPU slot."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    first = client.post("/jobs", json=VALID_SUBMIT)
    second = client.post("/jobs", json=VALID_SUBMIT)
    third = client.post("/jobs", json=VALID_SUBMIT)
    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201
    assert third.json()["status"] == "READY"
    assert third.json()["queue_position"] >= 1
    job_id = third.json()["job_id"]
    deleted = client.delete(f"/jobs/{job_id}")
    assert deleted.status_code == 204
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CANCELLED"
    for proc in holds:
        proc.release()


def test_prescan_rewrites_preview_url(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prescan CLI stdout preview path becomes an HTTP URL."""
    preview = tmp_path / "nh48" / "prescan" / "prescan_abc_preview.jpg"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"\xff\xd8\xff")
    stdout = json.dumps(
        {
            "prescan_id": "prescan_abc",
            "video_meta": {
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "duration_sec": 1.0,
                "frame_count": 25,
            },
            "ocr": {},
            "preview_url": str(preview),
            "profiles": [],
        }
    )

    def fake_run(args: Any, timeout: float | None = None) -> CompletedProcess[str]:
        return CompletedProcess(args=list(args), returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("orchestrator.routes.prescan.run_viana", fake_run)
    response = client.post(
        "/utils/prescan",
        json={
            "source_video_path": SOURCE,
            "project_id": "nh48",
            "frame_offset_sec": 0.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["preview_url"] == "/utils/prescan/prescan_abc/preview.jpg"
    image = client.get("/utils/prescan/prescan_abc/preview.jpg")
    assert image.status_code == 200


def test_profiles_roundtrip(client: TestClient) -> None:
    """GET/POST profiles write JSON under the project output dir."""
    empty = client.get("/projects/nh48/profiles")
    assert empty.status_code == 200
    assert empty.json() == []
    created = client.post("/projects/nh48/profiles", json=VALID_PROFILE)
    assert created.status_code == 201
    assert created.json()["profile_id"] == "morning_northbound"
    listed = client.get("/projects/nh48/profiles")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_aggregate_shells_cli(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST aggregate invokes viana aggregate."""
    submitted = client.post("/jobs", json=VALID_SUBMIT)
    job_id = submitted.json()["job_id"]
    get_pool().wait_job(job_id)

    def fake_run(args: Any, timeout: float | None = None) -> CompletedProcess[str]:
        assert args[0] == "aggregate"
        return CompletedProcess(
            args=list(args),
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "ok",
                    "command": "aggregate",
                    "rows": 0,
                    "events": "x_events.csv",
                    "aggregate_15min": "x_15min.csv",
                    "partial": False,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("orchestrator.routes.jobs.run_viana", fake_run)
    response = client.post(f"/jobs/{job_id}/aggregate")
    assert response.status_code == 200
    assert response.json()["command"] == "aggregate"


def test_ws_jobs_forwards_telemetry(client: TestClient) -> None:
    """WS payload matches telemetry.schema.json required fields."""
    with client.websocket_connect("/ws/jobs") as websocket:
        response = client.post("/jobs", json=VALID_SUBMIT)
        assert response.status_code == 201
        payload = websocket.receive_json()
    assert payload["telemetry_type"] == "PROGRESS"
    assert "job_id" in payload
    assert isinstance(payload["data"], dict)


def test_get_unknown_job_404(client: TestClient) -> None:
    """Missing jobs are 404, not 501."""
    assert client.get("/jobs/missing").status_code == 404


CONFIRM_BODY = {
    "metadata": {
        "user_start_time": "09:00:00",
        "user_start_date": "15-03-2026",
        "location": "NH48 Km42",
    },
    "task_parameters": VALID_SUBMIT["task_parameters"],
    "calibration_profile_id": "morning_northbound",
}


def test_post_jobs_intake_creates_prescan_pending(client: TestClient) -> None:
    """POST /jobs/intake registers videos at PRESCAN_PENDING."""
    response = client.post(
        "/jobs/intake",
        json={
            "project_id": "nh48",
            "source_video_paths": [SOURCE, "/data/projects/nh48/videos/2026-03-15_10-00.mp4"],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body["jobs"]) == 2
    assert body["jobs"][0]["status"] == "PRESCAN_PENDING"
    assert body["jobs"][0]["queue_position"] == 1
    job_id = body["jobs"][0]["job_id"]
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "PRESCAN_PENDING"


def test_post_jobs_intake_output_dir_override(client: TestClient, tmp_path: Path) -> None:
    """Intake accepts browsable output_dir override (G20)."""
    custom = tmp_path / "custom-out"
    response = client.post(
        "/jobs/intake",
        json={
            "project_id": "nh48",
            "source_video_paths": [SOURCE],
            "output_dir": str(custom),
        },
    )
    assert response.status_code == 201
    assert response.json()["jobs"][0]["output_dir"] == str(custom)


def test_patch_prescan_confirm_to_ready(client: TestClient) -> None:
    """PATCH /jobs/{id}/prescan validates metadata and moves job to READY."""
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    get_pool().stub_awaiting_review(job_id)
    response = client.patch(f"/jobs/{job_id}/prescan", json=CONFIRM_BODY)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "READY"
    assert payload["confirmed_metadata"]["user_start_time"] == "09:00:00"
    get_pool().wait_job(job_id)
    final = client.get(f"/jobs/{job_id}")
    assert final.json()["status"] == "COMPLETED"


def test_patch_prescan_rejects_invalid_metadata(client: TestClient) -> None:
    """Metadata must use HH:MM:SS and DD-MM-YYYY."""
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    get_pool().stub_awaiting_review(job_id)
    bad_time = client.patch(
        f"/jobs/{job_id}/prescan",
        json={
            **CONFIRM_BODY,
            "metadata": {
                "user_start_time": "9:00 AM",
                "user_start_date": "2026-03-15",
                "location": "NH48",
            },
        },
    )
    assert bad_time.status_code == 422


def test_patch_prescan_rejects_wrong_status(client: TestClient) -> None:
    """Cannot confirm prescan while still PRESCAN_PENDING."""
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    response = client.patch(f"/jobs/{job_id}/prescan", json=CONFIRM_BODY)
    assert response.status_code == 400


def test_worker_pool_max_two_gpus() -> None:
    """Queue design: at most two concurrent GPU devices."""
    assert MAX_CONCURRENT_GPU_JOBS == 2
    assert GPU_DEVICES == ("cuda:0", "cuda:1")
    pool = WorkerPool()
    first = pool.assign_gpu(set())
    second = pool.assign_gpu({first} if first else set())
    third = pool.assign_gpu(set(GPU_DEVICES))
    assert first == "cuda:0"
    assert second == "cuda:1"
    assert third is None
