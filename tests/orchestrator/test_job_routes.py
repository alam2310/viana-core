"""Phase 6 orchestrator tests — mocked ``python -m viana`` subprocesses."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.workers.pool import (
    CHECKPOINT_CONFLICT,
    GPU_DEVICES,
    MAX_CONCURRENT_GPU_JOBS,
    MAX_CONCURRENT_PRESCAN_JOBS,
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
        self.stdin = None
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.pid = 0

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
        self.stdin = None
        self.stdout = io.StringIO(_run_result_json("job_hold"))
        self.stderr = _HoldStderr(self._done)
        self.returncode: int | None = None
        self.pid = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        finished = self._done.wait(timeout)
        if not finished and timeout is not None and self.returncode is None:
            raise TimeoutExpired("hold", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self._done.set()

    def interrupt(self) -> None:
        """Simulate SIGINT (operator pause) — no RunResult on stdout."""
        self.returncode = -2
        self.stdout = io.StringIO("")
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

    def close(self) -> None:
        self._done.set()


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

    def fake_run_viana(
        args: Any, timeout: float | None = None, **_kwargs: Any
    ) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            return CompletedProcess(
                args=list(args),
                returncode=1,
                stdout="",
                stderr="prescan skipped in tests",
            )
        if args and args[0] == "aggregate":
            return CompletedProcess(
                args=list(args),
                returncode=0,
                stdout=json.dumps({"status": "ok", "command": "aggregate", "rows": 0}),
                stderr="",
            )
        raise AssertionError(f"unexpected run_viana call: {args}")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run_viana)
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
    _assert_iso_datetime(payload["created_at"])
    assert isinstance(payload["processing_duration_sec"], int | float)
    assert payload["processing_duration_sec"] >= 0
    listed = client.get("/jobs?project_id=nh48")
    assert listed.status_code == 200
    listed_row = next(item for item in listed.json() if item["job_id"] == job_id)
    assert listed_row["created_at"] == payload["created_at"]
    assert listed_row["processing_duration_sec"] == payload["processing_duration_sec"]


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


def test_post_jobs_409_on_legacy_flat_checkpoint(client: TestClient, tmp_path: Path) -> None:
    """Pre-S29 flat checkpoint must still 409 (no silent resume)."""
    from viana.io.paths import legacy_artifact_paths

    output_dir = project_output_dir(tmp_path, "nh48")
    ckpt = legacy_artifact_paths(output_dir, STEM)["checkpoint"]
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

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
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


def test_prescan_preview_survives_registry_restart(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview JPEG resolves from disk when in-memory registry is empty (S01)."""
    import orchestrator.preview_registry as preview_registry

    preview = tmp_path / "nh48" / "prescan" / "prescan_restart_preview.jpg"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"\xff\xd8\xff")
    stdout = json.dumps(
        {
            "prescan_id": "prescan_restart",
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

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
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
    preview_registry._PREVIEW_FILES.clear()
    image = client.get("/utils/prescan/prescan_restart/preview.jpg")
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/jpeg")


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

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
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
    assert status.json()["status"] in {
        "PRESCAN_PENDING",
        "PRESCAN_RUNNING",
        "AWAITING_REVIEW",
        "PRESCAN_FAILED",
    }


def _write_prior_checkpoint(output_dir: Path, *, current: int = 100, total: int = 100) -> Path:
    """Write a checkpoint file (complete or incomplete) for S36 intake detection."""
    paths = artifact_paths(output_dir, STEM)
    save_checkpoint(
        paths["checkpoint"],
        Checkpoint(
            job_id="job_prior",
            project_id="nh48",
            source_video_path=Path(SOURCE),
            video_stem=STEM,
            current_frame=current,
            total_frames=total,
            saved_at="2026-03-15T10:00:00Z",
        ),
    )
    return paths["checkpoint"]


def test_s36_intake_checkpoint_exists_skips_prescan(client: TestClient, tmp_path: Path) -> None:
    """Prior checkpoint (complete or not) → CHECKPOINT_EXISTS before prescan (S36)."""
    output_dir = project_output_dir(tmp_path, "nh48")
    _write_prior_checkpoint(output_dir, current=10, total=100)

    response = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert response.status_code == 201
    item = response.json()["jobs"][0]
    assert item["status"] == "CHECKPOINT_EXISTS"
    assert item["queue_position"] == 0
    job_id = item["job_id"]
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["status"] == "CHECKPOINT_EXISTS"
    assert detail["checkpoint_exists"] is True
    # Must not enter the prescan worker.
    assert detail["status"] not in {"PRESCAN_RUNNING", "AWAITING_REVIEW", "PRESCAN_FAILED"}


def test_s36_intake_complete_checkpoint_same_as_incomplete(
    client: TestClient, tmp_path: Path
) -> None:
    """Complete and incomplete checkpoints both map to CHECKPOINT_EXISTS."""
    output_dir = project_output_dir(tmp_path, "nh48")
    _write_prior_checkpoint(output_dir, current=100, total=100)
    response = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert response.json()["jobs"][0]["status"] == "CHECKPOINT_EXISTS"


def test_s36_start_fresh_from_checkpoint_exists_enters_prescan(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart (Overwrite) from Partial wipes sidecars and re-queues prescan."""
    output_dir = project_output_dir(tmp_path, "nh48")
    ckpt = _write_prior_checkpoint(output_dir, current=50, total=100)
    events = artifact_paths(output_dir, STEM)["events"]
    events.write_text("track_id\n1\n", encoding="utf-8")
    assert ckpt.is_file()

    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "CHECKPOINT_EXISTS"

    preview = tmp_path / "nh48" / "prescan" / "prescan_abc_preview.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"\xff\xd8\xff")
    stdout = _prescan_stdout(preview)

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            return CompletedProcess(args=list(args), returncode=0, stdout=stdout, stderr="")
        return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run)

    response = client.post(f"/jobs/{job_id}/start-fresh")
    assert response.status_code == 200
    assert not ckpt.is_file()
    assert not events.is_file()
    get_pool().wait_for_status(job_id, "AWAITING_REVIEW", "PRESCAN_FAILED", timeout=5.0)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "AWAITING_REVIEW"


def test_s36_confirm_rejected_while_checkpoint_exists(client: TestClient, tmp_path: Path) -> None:
    """Partial jobs cannot skip to confirm/GPU without Restart (Overwrite)."""
    output_dir = project_output_dir(tmp_path, "nh48")
    _write_prior_checkpoint(output_dir)
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    response = client.patch(f"/jobs/{job_id}/prescan", json=CONFIRM_BODY)
    assert response.status_code == 400


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
    assert MAX_CONCURRENT_PRESCAN_JOBS == 2
    assert GPU_DEVICES == ("cuda:0", "cuda:1")
    pool = WorkerPool()
    first = pool.assign_gpu(set())
    second = pool.assign_gpu({first} if first else set())
    third = pool.assign_gpu(set(GPU_DEVICES))
    assert first == "cuda:0"
    assert second == "cuda:1"
    assert third is None


def _prescan_stdout(preview: Path) -> str:
    return json.dumps(
        {
            "prescan_id": "prescan_abc",
            "video_meta": {
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "duration_sec": 1.0,
                "frame_count": 25,
            },
            "ocr": {
                "time": "09:00:00",
                "date": "15-03-2026",
                "location": "NH48 Km42",
                "confidence": 0.82,
            },
            "proposed_lines": {
                "horizon_line": VALID_SUBMIT["task_parameters"]["horizon_line"],
                "counting_line": VALID_SUBMIT["task_parameters"]["counting_line"],
                "confidence": 0.75,
            },
            "preview_url": str(preview),
            "profiles": [],
        }
    )


def test_intake_prescan_worker_reaches_awaiting_review(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intake job runs prescan worker → AWAITING_REVIEW with proposals (G13)."""
    preview = tmp_path / "nh48" / "prescan" / "prescan_abc_preview.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"\xff\xd8\xff")
    stdout = _prescan_stdout(preview)

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            return CompletedProcess(args=list(args), returncode=0, stdout=stdout, stderr="")
        return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run)
    reset_pool()
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert intake.status_code == 201
    job_id = intake.json()["jobs"][0]["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_id, "AWAITING_REVIEW", "PRESCAN_FAILED", timeout=5.0)
    status = client.get(f"/jobs/{job_id}")
    payload = status.json()
    assert payload["status"] == "AWAITING_REVIEW"
    assert payload["proposed_metadata"]["user_start_time"] == "09:00:00"
    assert payload["proposed_preview_url"] == "/utils/prescan/prescan_abc/preview.jpg"
    assert payload["video_duration_sec"] == 1.0
    _assert_iso_datetime(payload["created_at"])
    assert payload["processing_duration_sec"] is None
    listed = client.get("/jobs?project_id=nh48").json()
    listed_row = next(item for item in listed if item["job_id"] == job_id)
    assert listed_row["created_at"] == payload["created_at"]
    assert listed_row["video_duration_sec"] == 1.0


def test_retry_prescan_from_failed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PRESCAN_FAILED → retry → AWAITING_REVIEW."""
    preview = tmp_path / "nh48" / "prescan" / "prescan_abc_preview.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"\xff\xd8\xff")
    stdout = _prescan_stdout(preview)
    calls = {"prescan": 0}

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            calls["prescan"] += 1
            if calls["prescan"] == 1:
                return CompletedProcess(
                    args=list(args), returncode=1, stdout="", stderr="ocr timeout"
                )
            return CompletedProcess(args=list(args), returncode=0, stdout=stdout, stderr="")
        return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run)
    reset_pool()
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    get_pool().wait_for_status(job_id, "PRESCAN_FAILED", timeout=5.0)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "PRESCAN_FAILED"
    retry = client.post(f"/jobs/{job_id}/prescan/retry")
    assert retry.status_code == 200
    get_pool().wait_for_status(job_id, "AWAITING_REVIEW", timeout=5.0)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "AWAITING_REVIEW"


def test_job_prescan_preview_at_offset(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /jobs/{id}/prescan/preview runs prescan at scrub offset (G8)."""
    preview = tmp_path / "nh48" / "prescan" / "prescan_scrub_preview.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"\xff\xd8\xff")
    captured: dict[str, float] = {}

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            captured["frame_offset"] = float(args[args.index("--frame-offset") + 1])
            payload = json.loads(_prescan_stdout(preview))
            payload["prescan_id"] = "prescan_scrub"
            payload["preview_url"] = str(preview)
            return CompletedProcess(
                args=list(args),
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )
        return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run)
    reset_pool()
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    get_pool().stub_awaiting_review(job_id)
    response = client.get(f"/jobs/{job_id}/prescan/preview?frame_offset_sec=15.0")
    assert response.status_code == 200
    assert captured["frame_offset"] == 15.0
    assert response.json()["preview_url"] == "/utils/prescan/prescan_scrub/preview.jpg"


def test_source_mp4_served_with_range(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /artifacts/{id}/source.mp4 streams intake video with HTTP Range (S02)."""
    source = tmp_path / "clip.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42" + b"\xab" * 100
    source.write_bytes(payload)
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [str(source)]},
    )
    job_id = intake.json()["jobs"][0]["job_id"]
    get_pool().stub_awaiting_review(job_id)
    response = client.get(f"/artifacts/{job_id}/source.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers.get("accept-ranges") == "bytes"
    assert response.headers.get("content-disposition", "").startswith("inline")
    assert response.content == payload
    ranged = client.get(
        f"/artifacts/{job_id}/source.mp4",
        headers={"Range": "bytes=0-7"},
    )
    assert ranged.status_code == 206
    assert ranged.content == payload[:8]
    get_pool().get_job(job_id).status = "PROCESSING"
    denied = client.get(f"/artifacts/{job_id}/source.mp4")
    assert denied.status_code == 404


def test_partial_processed_mp4_served(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /artifacts/{id}/partial.mp4 returns growing file (G19)."""
    output_dir = project_output_dir(tmp_path, "nh48")
    mp4 = artifact_paths(output_dir, STEM)["processed_video"]
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    submitted = client.post("/jobs", json=VALID_SUBMIT)
    job_id = submitted.json()["job_id"]
    pool = get_pool()
    job = pool.get_job(job_id)
    job.status = "PROCESSING"
    response = client.get(f"/artifacts/{job_id}/partial.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers.get("accept-ranges") == "bytes"
    assert response.headers.get("cache-control") == "no-store"
    disposition = response.headers.get("content-disposition", "")
    assert disposition.startswith("inline")
    ranged = client.get(
        f"/artifacts/{job_id}/partial.mp4",
        headers={"Range": "bytes=0-3"},
    )
    assert ranged.status_code == 206
    assert ranged.content == b"\x00\x00\x00\x18"
    assert ranged.headers.get("content-range", "").startswith("bytes 0-3/")


def test_auto_aggregate_on_completed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """COMPLETED job triggers background aggregate (G12)."""
    aggregate_calls: list[list[str]] = []

    def fake_run(args: Any, timeout: float | None = None, **_kwargs: Any) -> CompletedProcess[str]:
        if args and args[0] == "aggregate":
            aggregate_calls.append(list(args))
            return CompletedProcess(
                args=list(args),
                returncode=0,
                stdout=json.dumps({"status": "ok", "command": "aggregate", "rows": 0}),
                stderr="",
            )
        raise AssertionError(f"unexpected run_viana call: {args}")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run)
    submitted = client.post("/jobs", json=VALID_SUBMIT)
    job_id = submitted.json()["job_id"]
    get_pool().wait_job(job_id, timeout=5.0)
    import time

    deadline = time.time() + 2.0
    while time.time() < deadline and not aggregate_calls:
        time.sleep(0.05)
    assert aggregate_calls
    assert aggregate_calls[0][0] == "aggregate"


def test_progress_telemetry_includes_eta_and_crossings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WS PROGRESS carries eta_sec and crossing_count (G9)."""
    line = json.dumps(
        {
            "job_id": "pending",
            "status": "PROCESSING",
            "telemetry_type": "PROGRESS",
            "data": {
                "current_frame": 50,
                "total_frames": 100,
                "processing_fps": 25.0,
                "crossing_count": 3,
            },
        }
    )

    class ProgressPopen(InstantPopen):
        def __init__(self) -> None:
            super().__init__(_run_result_json("job_eta"), line + "\n")

    monkeypatch.setattr(
        "orchestrator.workers.pool.start_viana_process",
        lambda *_args, **_kwargs: ProgressPopen(),
    )
    reset_pool()
    with client.websocket_connect("/ws/jobs") as websocket:
        response = client.post("/jobs", json=VALID_SUBMIT)
        assert response.status_code == 201
        payload = websocket.receive_json()
    assert payload["telemetry_type"] == "PROGRESS"
    assert payload["data"]["eta_sec"] == 2.0
    assert payload["data"]["crossing_count"] == 3


def test_ws_forwards_moving_event_with_timestamp(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S14: MOVING_EVENT telemetry reaches WS with timestamp fields."""
    line = json.dumps(
        {
            "job_id": "pending",
            "status": "PROCESSING",
            "telemetry_type": "MOVING_EVENT",
            "data": {
                "track_id": 9,
                "class_name": "Car",
                "direction": "in",
                "frame_index": 42,
                "fps": 25.0,
                "video_pts_ms": 1680.0,
                "event_timestamp": "2026-03-15T09:00:01.680000Z",
                "event_timestamp_source": "ocr_anchor",
                "event_timestamp_confidence": 0.92,
            },
        }
    )

    class MovingEventPopen(InstantPopen):
        def __init__(self) -> None:
            super().__init__(_run_result_json("job_evt"), line + "\n")

    monkeypatch.setattr(
        "orchestrator.workers.pool.start_viana_process",
        lambda *_args, **_kwargs: MovingEventPopen(),
    )
    reset_pool()
    with client.websocket_connect("/ws/jobs") as websocket:
        response = client.post("/jobs", json=VALID_SUBMIT)
        assert response.status_code == 201
        payload = websocket.receive_json()
    assert payload["telemetry_type"] == "MOVING_EVENT"
    assert payload["data"]["event_timestamp"] == "2026-03-15T09:00:01.680000Z"
    assert payload["data"]["event_timestamp_source"] == "ocr_anchor"


def _assert_iso_datetime(value: object) -> None:
    """Require a JSON Schema date-time string (ISO-8601, UTC Z ok)."""
    assert isinstance(value, str) and value
    datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_intake_list_includes_created_at_before_prescan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S11: GET /jobs returns created_at as soon as intake registers the job."""

    def fail_prescan(
        args: Any, timeout: float | None = None, **_kwargs: Any
    ) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            return CompletedProcess(args=list(args), returncode=1, stdout="", stderr="held")
        return CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fail_prescan)
    reset_pool()
    intake = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert intake.status_code == 201
    job_id = intake.json()["jobs"][0]["job_id"]
    listed = client.get("/jobs?project_id=nh48")
    assert listed.status_code == 200
    row = next(item for item in listed.json() if item["job_id"] == job_id)
    _assert_iso_datetime(row["created_at"])
    assert row["video_duration_sec"] is None
    assert row["processing_duration_sec"] is None
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["created_at"] == row["created_at"]


def test_processing_duration_live_then_frozen(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S12: processing_duration_sec grows during PROCESSING and freezes after COMPLETED."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()
    response = client.post("/jobs", json=VALID_SUBMIT)
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    get_pool().wait_for_status(job_id, "PROCESSING", timeout=5.0)
    first = client.get(f"/jobs/{job_id}").json()
    _assert_iso_datetime(first["created_at"])
    assert isinstance(first["processing_duration_sec"], int | float)
    time.sleep(0.05)
    second = client.get(f"/jobs/{job_id}").json()
    assert second["processing_duration_sec"] >= first["processing_duration_sec"]
    holds[0].release()
    get_pool().wait_job(job_id, timeout=5.0)
    done = client.get(f"/jobs/{job_id}").json()
    assert done["status"] == "COMPLETED"
    frozen = done["processing_duration_sec"]
    assert isinstance(frozen, int | float)
    assert frozen >= 0
    time.sleep(0.05)
    again = client.get(f"/jobs/{job_id}").json()
    assert again["processing_duration_sec"] == frozen
    listed = next(item for item in client.get("/jobs").json() if item["job_id"] == job_id)
    assert listed["processing_duration_sec"] == frozen
    assert listed["created_at"] == done["created_at"]


def test_processing_duration_helper_freezes_after_end(tmp_path: Path) -> None:
    """processing_duration_sec uses a frozen end timestamp, not wall-clock after stop."""
    from orchestrator.workers.pool import JobRecord, _processing_duration_sec

    job = JobRecord(
        job_id="job_timing",
        status="COMPLETED",
        source_video_path=Path(SOURCE),
        project_id="nh48",
        output_dir=tmp_path,
    )
    job.processing_started_monotonic = 100.0
    job.processing_ended_monotonic = 101.25
    assert _processing_duration_sec(job) == 1.25


def test_s30_resume_and_start_fresh_keep_list_jobs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S30: resume/start-fresh mutates succeed and GET /jobs stays healthy."""
    holds: list[HoldPopen] = []

    def start(_args: object) -> HoldPopen:
        proc = HoldPopen()
        proc.stdout = io.StringIO("")  # no RunResult → signal exit can become PAUSED
        holds.append(proc)
        return proc

    monkeypatch.setattr("orchestrator.workers.pool.start_viana_process", start)
    reset_pool()

    created = client.post("/jobs", json=VALID_SUBMIT)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    pool = get_pool()
    pool.wait_for_status(job_id, "PROCESSING", timeout=5.0)
    assert client.get("/jobs?project_id=nh48").status_code == 200

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
    holds[0].terminate()
    pool.wait_for_status(job_id, "PAUSED", timeout=5.0)

    resumed = client.post(f"/jobs/{job_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "PROCESSING"
    listed = client.get("/jobs?project_id=nh48")
    assert listed.status_code == 200
    assert any(row["job_id"] == job_id for row in listed.json())

    holds[1].terminate()
    pool.wait_for_status(job_id, "PAUSED", timeout=5.0)

    fresh = client.post(f"/jobs/{job_id}/start-fresh")
    assert fresh.status_code == 200
    assert fresh.json()["status"] == "PROCESSING"
    assert client.get("/jobs?project_id=nh48").status_code == 200
    holds[2].release()
    pool.wait_for_status(job_id, "COMPLETED", "FAILED", "CANCELLED", timeout=5.0)
    assert client.get(f"/jobs/{job_id}").status_code == 200
