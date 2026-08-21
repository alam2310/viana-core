"""Tests for run_result JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viana.io.run_result import RunResult, RunResultArtifacts, load_run_result, save_run_result


def test_load_run_result_missing_file(tmp_path: Path) -> None:
    """Missing run_result JSON raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Run result not found"):
        load_run_result(tmp_path / "missing.json")


def test_load_run_result_invalid_json(tmp_path: Path) -> None:
    """JSON that is not a dictionary raises ValueError."""
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_run_result(path)

    path.write_text('"COMPLETED"', encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_run_result(path)


def test_save_and_load_run_result(tmp_path: Path) -> None:
    """Round-trip run_result JSON."""
    path = tmp_path / "run_result.json"
    original = RunResult(
        schema_version=1,
        job_id="job_1",
        status="COMPLETED",
        source_video_path="/data/v.mp4",
        video_stem="v",
        artifacts=RunResultArtifacts(events="/data/v.csv"),
        completed_at="2026-03-15T09:07:12Z",
    )
    save_run_result(path, original)
    loaded = load_run_result(path)
    assert loaded.job_id == "job_1"
    assert loaded.status == "COMPLETED"
    assert loaded.artifacts.events == "/data/v.csv"
