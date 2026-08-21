"""Tests for run_result JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viana.io.run_result import (
    RunResultArtifacts,
    completed_now,
    load_run_result,
    save_run_result,
)


def test_load_run_result_missing_file(tmp_path: Path) -> None:
    """load_run_result raises FileNotFoundError if the file does not exist."""
    with pytest.raises(FileNotFoundError, match="Run result not found"):
        load_run_result(tmp_path / "missing_run_result.json")


def test_load_run_result_not_dict(tmp_path: Path) -> None:
    """load_run_result raises ValueError if the JSON payload is not an object/dict."""
    path = tmp_path / "array.run_result.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_run_result(path)


def test_save_and_load_run_result(tmp_path: Path) -> None:
    """Round-trip run_result JSON."""
    path = tmp_path / "clip.run_result.json"
    artifacts = RunResultArtifacts(events="/data/events.csv")
    original = completed_now(
        job_id="job_1",
        source_video_path=Path("/data/v.mp4"),
        video_stem="v",
        artifacts=artifacts,
    )
    save_run_result(path, original)
    loaded = load_run_result(path)
    assert loaded.job_id == "job_1"
    assert loaded.status == "COMPLETED"
    assert loaded.video_stem == "v"
    assert loaded.artifacts.events == "/data/events.csv"
