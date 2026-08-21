from pathlib import Path

import pytest
from pydantic import ValidationError

from viana.io.run_result import (
    RunResult,
    RunResultArtifacts,
    completed_now,
    load_run_result,
    save_run_result,
)


def test_save_and_load_run_result(tmp_path: Path) -> None:
    """Test saving and loading a RunResult works correctly and creates parent dirs."""
    artifacts = RunResultArtifacts(events="events.csv", processed_video="out.mp4")
    result = RunResult(
        job_id="job_123",
        status="COMPLETED",
        source_video_path="/path/to/vid.mp4",
        video_stem="vid",
        artifacts=artifacts,
        completed_at="2023-10-10T10:10:10Z",
    )

    # Path inside a non-existent directory
    target_path = tmp_path / "subdir" / "result.json"

    save_run_result(target_path, result)

    assert target_path.exists()

    # Test load
    loaded = load_run_result(target_path)
    assert loaded == result
    assert loaded.job_id == "job_123"
    assert loaded.artifacts.events == "events.csv"
    assert loaded.artifacts.processed_video == "out.mp4"


def test_load_run_result_not_found(tmp_path: Path) -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    target_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Run result not found"):
        load_run_result(target_path)


def test_load_run_result_not_dict(tmp_path: Path) -> None:
    """Test loading a JSON array raises ValueError."""
    target_path = tmp_path / "array.json"
    target_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a JSON object"):
        load_run_result(target_path)


def test_load_run_result_invalid_schema(tmp_path: Path) -> None:
    """Test loading an invalid JSON object raises ValidationError."""
    target_path = tmp_path / "invalid.json"
    # Missing required fields like job_id
    target_path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_run_result(target_path)


def test_completed_now() -> None:
    """Test completed_now populates correctly."""
    artifacts = RunResultArtifacts()
    result = completed_now(
        job_id="job_456",
        source_video_path=Path("/some/video.mp4"),
        video_stem="video",
        artifacts=artifacts,
    )

    assert result.job_id == "job_456"
    assert result.status == "COMPLETED"
    assert result.source_video_path == "/some/video.mp4"
    assert result.video_stem == "video"
    assert result.error_message is None
    assert result.completed_at is not None  # Should be set


def test_completed_now_failed() -> None:
    """Test completed_now with custom status."""
    artifacts = RunResultArtifacts()
    result = completed_now(
        job_id="job_789",
        source_video_path=Path("/some/video.mp4"),
        video_stem="video",
        artifacts=artifacts,
        status="FAILED",
        error_message="Something went wrong",
    )

    assert result.status == "FAILED"
    assert result.error_message == "Something went wrong"


def test_run_result_forbid_extra() -> None:
    """Test that extra fields are forbidden in Pydantic models."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunResultArtifacts(extra_field="not allowed") # type: ignore

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunResult(
            job_id="job_123",
            status="COMPLETED",
            source_video_path="/path/to/vid.mp4",
            video_stem="vid",
            artifacts=RunResultArtifacts(),
            completed_at="2023-10-10T10:10:10Z",
            extra_field="not allowed", # type: ignore
        )
