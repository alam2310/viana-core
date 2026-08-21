"""Step 6.7 / S09 / F006 — intake path normalize and reject."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import app
from orchestrator.intake_paths import IntakePathError, resolve_intake_path, resolve_intake_paths
from orchestrator.workers.pool import get_pool, reset_pool
from tests.orchestrator.test_job_routes import SOURCE

ROOTS = ("/data", "/app/ViAna")
MAPS = (
    ("/home/mushaffa/Work/ViAna/data", "/data"),
    ("/home/mushaffa/Work/ViAna", "/app/ViAna"),
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))
    monkeypatch.setenv("VIANA_INTAKE_ROOTS", "/data:/app/ViAna")
    monkeypatch.setenv("VIANA_HOST_REPO_ROOT", "/home/mushaffa/Work/ViAna")
    monkeypatch.setenv("VIANA_HOST_DATA_ROOT", "/home/mushaffa/Work/ViAna/data")
    monkeypatch.delenv("VIANA_PATH_MAPS", raising=False)

    def fake_run_viana(
        args: Any, timeout: float | None = None, **_kwargs: Any
    ) -> CompletedProcess[str]:
        if args and args[0] == "prescan":
            return CompletedProcess(args=list(args), returncode=1, stdout="", stderr="skipped")
        raise AssertionError(f"unexpected run_viana call: {args}")

    monkeypatch.setattr("orchestrator.workers.pool.run_viana", fake_run_viana)
    reset_pool()
    with TestClient(app) as test_client:
        yield test_client
    reset_pool()


def test_keeps_container_data_path() -> None:
    """Paths already under /data are accepted unchanged."""
    assert resolve_intake_path(SOURCE, roots=ROOTS, maps=MAPS) == Path(SOURCE)


def test_maps_host_data_prefix_to_container() -> None:
    """Host ./data path rewrites to /data (Step 5 negative-path repro)."""
    host = "/home/mushaffa/Work/ViAna/data/raw/s67-host-clip.mp4"
    assert resolve_intake_path(host, roots=ROOTS, maps=MAPS) == Path("/data/raw/s67-host-clip.mp4")


def test_longest_host_prefix_wins() -> None:
    """data mount wins over repo mount when both prefixes match."""
    host = "/home/mushaffa/Work/ViAna/data/raw/clip.mp4"
    assert resolve_intake_path(host, roots=ROOTS, maps=MAPS) == Path("/data/raw/clip.mp4")


def test_maps_host_repo_file_to_app_viana() -> None:
    """Repo-relative host paths rewrite to /app/ViAna."""
    host = "/home/mushaffa/Work/ViAna/tests/viana/fixtures/clip.mp4"
    assert resolve_intake_path(host, roots=ROOTS, maps=MAPS) == Path(
        "/app/ViAna/tests/viana/fixtures/clip.mp4"
    )


def test_suffix_rewrite_without_host_maps() -> None:
    """Unconfigured host maps still rewrite .../data/rest onto /data/rest."""
    host = "/opt/elsewhere/data/raw/clip.mp4"
    assert resolve_intake_path(host, roots=ROOTS, maps=()) == Path("/data/raw/clip.mp4")


def test_relative_data_prefix() -> None:
    """Operators may send data/raw/clip.mp4 instead of /data/raw/clip.mp4."""
    assert resolve_intake_path("data/raw/clip.mp4", roots=ROOTS, maps=()) == Path(
        "/data/raw/clip.mp4"
    )


def test_rejects_unmapped_host_path() -> None:
    """Paths outside bind-mounts are rejected (not queued as PRESCAN_FAILED)."""
    with pytest.raises(IntakePathError, match="not readable inside the processing container"):
        resolve_intake_path("/mnt/unmounted-hdd/videos/clip.mp4", roots=ROOTS, maps=MAPS)


def test_existing_file_outside_roots_is_accepted(tmp_path: Path) -> None:
    """Host-mode / pytest: a file this process can read is kept as-is."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"ftyp")
    assert resolve_intake_path(clip, roots=ROOTS, maps=MAPS) == clip


def test_prefers_readable_mapped_file(tmp_path: Path) -> None:
    """When the container path exists, store that rather than the host path."""
    container_root = tmp_path / "data"
    clip = container_root / "raw" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"ftyp")
    host = "/host/videos/data/raw/clip.mp4"
    assert (
        resolve_intake_path(
            host,
            roots=(str(container_root),),
            maps=(("/host/videos/data", str(container_root)),),
        )
        == clip
    )


def test_batch_rejects_all_when_one_path_is_unreadable() -> None:
    """One bad path fails the whole intake batch."""
    with pytest.raises(IntakePathError, match="unmounted-hdd"):
        resolve_intake_paths(
            [SOURCE, "/mnt/unmounted-hdd/videos/clip.mp4"],
            roots=ROOTS,
            maps=MAPS,
        )


def test_extra_path_map_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """G21 extra bind-mount: VIANA_PATH_MAPS + VIANA_INTAKE_ROOTS."""
    monkeypatch.setenv("VIANA_INTAKE_ROOTS", "/data:/app/ViAna:/mnt/extra")
    monkeypatch.setenv("VIANA_HOST_REPO_ROOT", "")
    monkeypatch.setenv("VIANA_HOST_DATA_ROOT", "")
    monkeypatch.setenv("VIANA_PATH_MAPS", "/media/hdd/videos->/mnt/extra")
    assert resolve_intake_path("/media/hdd/videos/cam1.mp4") == Path("/mnt/extra/cam1.mp4")


def test_relative_host_data_root_joins_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose may pass VIANA_HOST_DATA_ROOT=./data; join against host repo."""
    monkeypatch.setenv("VIANA_HOST_REPO_ROOT", "/home/mushaffa/Work/ViAna")
    monkeypatch.setenv("VIANA_HOST_DATA_ROOT", "./data")
    monkeypatch.delenv("VIANA_PATH_MAPS", raising=False)
    from orchestrator.settings import intake_path_maps

    assert ("/home/mushaffa/Work/ViAna/data", "/data") in intake_path_maps()


def test_intake_http_normalizes_host_data_path(client: TestClient) -> None:
    """POST /jobs/intake stores the container path, not the host path."""
    response = client.post(
        "/jobs/intake",
        json={
            "project_id": "nh48",
            "source_video_paths": ["/home/mushaffa/Work/ViAna/data/raw/s67-host-clip.mp4"],
        },
    )
    assert response.status_code == 201
    stored = response.json()["jobs"][0]["source_video_path"]
    assert stored == "/data/raw/s67-host-clip.mp4"
    job_id = response.json()["jobs"][0]["job_id"]
    status = client.get(f"/jobs/{job_id}")
    assert status.json()["source_video_path"] == stored


def test_intake_http_rejects_unreadable_path(client: TestClient) -> None:
    """Unmapped host paths return 400 and do not create jobs."""
    response = client.post(
        "/jobs/intake",
        json={
            "project_id": "nh48",
            "source_video_paths": ["/mnt/unmounted-hdd/videos/clip.mp4"],
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "not readable inside the processing container" in detail
    assert "VIANA_INTAKE_ROOTS" in detail
    assert get_pool().list_jobs() == []


def test_intake_http_accepts_existing_container_path(client: TestClient) -> None:
    """Canonical /data paths used by existing tests still intake."""
    response = client.post(
        "/jobs/intake",
        json={"project_id": "nh48", "source_video_paths": [SOURCE]},
    )
    assert response.status_code == 201
    assert response.json()["jobs"][0]["source_video_path"] == SOURCE
