"""Tests for viana.io.paths."""

from __future__ import annotations

from pathlib import Path

from viana.io.paths import (
    artifact_paths,
    job_config_path,
    legacy_artifact_paths,
    prescan_dir,
    profiles_dir,
    project_output_dir,
    resolve_artifact,
    stem_meta_dir,
    wipe_run_sidecars,
)


def test_project_output_dir() -> None:
    """Test resolution of project output directory."""
    parent_dir = Path("/data/viana-outputs")
    project_id = "test_project_123"
    result = project_output_dir(parent_dir, project_id)
    assert result == Path("/data/viana-outputs/test_project_123")


def test_artifact_paths_keep_layout() -> None:
    """Deliverables stay flat; sidecars under ``_meta/{stem}/`` (ADR 003)."""
    output_dir = Path("/data/viana-outputs/test_project_123")
    video_stem = "clip_01"
    result = artifact_paths(output_dir, video_stem)
    meta = stem_meta_dir(output_dir, video_stem)

    expected = {
        "events": Path("/data/viana-outputs/test_project_123/clip_01_events.csv"),
        "events_report": Path("/data/viana-outputs/test_project_123/clip_01_events_report.csv"),
        "aggregate_15min": Path("/data/viana-outputs/test_project_123/clip_01_15min.csv"),
        "processed_video": Path("/data/viana-outputs/test_project_123/clip_01_processed.mp4"),
        "manifest": meta / "manifest.json",
        "time_map": meta / "time_map.json",
        "checkpoint": meta / "checkpoint.json",
        "run_result": meta / "run_result.json",
    }
    assert result == expected


def test_resolve_artifact_prefers_legacy_flat_checkpoint(tmp_path: Path) -> None:
    """PAUSED resume must find pre-S29 ``{stem}.checkpoint.json`` (6.2)."""
    stem = "clip"
    legacy = legacy_artifact_paths(tmp_path, stem)["checkpoint"]
    legacy.write_text("{}", encoding="utf-8")
    assert resolve_artifact(tmp_path, stem, "checkpoint") == legacy


def test_resolve_artifact_prefers_canonical_over_legacy(tmp_path: Path) -> None:
    stem = "clip"
    canonical = artifact_paths(tmp_path, stem)["checkpoint"]
    canonical.parent.mkdir(parents=True)
    canonical.write_text("new", encoding="utf-8")
    legacy = legacy_artifact_paths(tmp_path, stem)["checkpoint"]
    legacy.write_text("old", encoding="utf-8")
    assert resolve_artifact(tmp_path, stem, "checkpoint") == canonical


def test_wipe_run_sidecars_clears_canonical_and_legacy(tmp_path: Path) -> None:
    stem = "clip"
    paths = artifact_paths(tmp_path, stem)
    paths["events"].write_text("e", encoding="utf-8")
    paths["checkpoint"].parent.mkdir(parents=True)
    paths["checkpoint"].write_text("c", encoding="utf-8")
    legacy = legacy_artifact_paths(tmp_path, stem)["checkpoint"]
    legacy.write_text("old", encoding="utf-8")
    wipe_run_sidecars(tmp_path, stem)
    assert not paths["events"].is_file()
    assert not paths["checkpoint"].is_file()
    assert not legacy.is_file()


def test_job_config_path() -> None:
    out = Path("/data/viana-outputs/nh44")
    assert job_config_path(out, "job_abc") == out / "_meta" / "jobs" / "job_abc.job.json"


def test_profiles_dir() -> None:
    """Test generation of profiles directory."""
    output_dir = Path("/data/viana-outputs/test_project_123")
    result = profiles_dir(output_dir)
    assert result == Path("/data/viana-outputs/test_project_123/profiles")


def test_prescan_dir() -> None:
    """Test generation of prescan directory."""
    output_dir = Path("/data/viana-outputs/test_project_123")
    result = prescan_dir(output_dir)
    assert result == Path("/data/viana-outputs/test_project_123/prescan")
