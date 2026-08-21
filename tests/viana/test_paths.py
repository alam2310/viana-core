"""Tests for viana.io.paths."""

from __future__ import annotations

from pathlib import Path

from viana.io.paths import (
    artifact_paths,
    prescan_dir,
    profiles_dir,
    project_output_dir,
)


def test_project_output_dir() -> None:
    """Test resolution of project output directory."""
    parent_dir = Path("/data/viana-outputs")
    project_id = "test_project_123"
    result = project_output_dir(parent_dir, project_id)
    assert result == Path("/data/viana-outputs/test_project_123")


def test_artifact_paths() -> None:
    """Test building standard artifact paths for a video stem."""
    output_dir = Path("/data/viana-outputs/test_project_123")
    video_stem = "clip_01"
    result = artifact_paths(output_dir, video_stem)

    expected = {
        "events": Path("/data/viana-outputs/test_project_123/clip_01_events.csv"),
        "aggregate_15min": Path("/data/viana-outputs/test_project_123/clip_01_15min.csv"),
        "processed_video": Path("/data/viana-outputs/test_project_123/clip_01_processed.mp4"),
        "manifest": Path("/data/viana-outputs/test_project_123/clip_01.manifest.json"),
        "time_map": Path("/data/viana-outputs/test_project_123/clip_01.time_map.json"),
        "checkpoint": Path("/data/viana-outputs/test_project_123/clip_01.checkpoint.json"),
        "run_result": Path("/data/viana-outputs/test_project_123/clip_01.run_result.json"),
    }
    assert result == expected


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
