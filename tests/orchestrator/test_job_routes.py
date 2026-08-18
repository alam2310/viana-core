"""Phase 6 route scaffold — 501 until engine CLI is ready."""

from __future__ import annotations

from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.errors import ENGINE_NOT_READY_DETAIL
from orchestrator.workers.pool import GPU_DEVICES, MAX_CONCURRENT_GPU_JOBS, WorkerPool

client = TestClient(app)

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


def test_health_still_ok() -> None:
    """Existing health probe must keep working."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": 0}


def test_post_jobs_rejects_client_job_id() -> None:
    """UI must not send job_id; extra fields are forbidden."""
    body = {**VALID_SUBMIT, "job_id": "job_from_ui"}
    response = client.post("/jobs", json=body)
    assert response.status_code == 422


def test_post_jobs_rejects_client_gpu_device() -> None:
    """UI must not send gpu_device; backend assigns cuda:0 or cuda:1."""
    body = {**VALID_SUBMIT, "gpu_device": "cuda:0"}
    response = client.post("/jobs", json=body)
    assert response.status_code == 422


def test_post_jobs_valid_body_is_501() -> None:
    """Workers are blocked on Phase 5; keep PROJECT_STATUS ❌."""
    response = client.post("/jobs", json=VALID_SUBMIT)
    assert response.status_code == 501
    assert response.json()["detail"] == ENGINE_NOT_READY_DETAIL


def test_job_lifecycle_stubs_are_501() -> None:
    """All job lifecycle routes exist and return 501."""
    assert client.get("/jobs").status_code == 501
    assert client.get("/jobs?project_id=nh48").status_code == 501
    assert client.get("/jobs/job_abc").status_code == 501
    assert client.post("/jobs/job_abc/resume").status_code == 501
    assert client.post("/jobs/job_abc/start-fresh").status_code == 501
    assert client.delete("/jobs/job_abc").status_code == 501
    assert client.post("/jobs/job_abc/aggregate").status_code == 501


def test_prescan_and_profiles_are_501() -> None:
    """Prescan and profile routes validate then stub."""
    prescan = client.post(
        "/utils/prescan",
        json={
            "source_video_path": "/data/projects/nh48/videos/2026-03-15_09-00.mp4",
            "project_id": "nh48",
            "frame_offset_sec": 0.0,
        },
    )
    assert prescan.status_code == 501

    listed = client.get("/projects/nh48/profiles")
    assert listed.status_code == 501

    created = client.post("/projects/nh48/profiles", json=VALID_PROFILE)
    assert created.status_code == 501


def test_ws_jobs_sends_telemetry_schema_payload() -> None:
    """Stub WS payload matches telemetry.schema.json required fields."""
    with client.websocket_connect("/ws/jobs") as websocket:
        payload = websocket.receive_json()
    assert payload["telemetry_type"] == "LOG"
    assert "job_id" in payload
    assert isinstance(payload["data"], dict)


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
