"""Phase 0 tests — config and path helpers."""

import pytest
from pydantic import ValidationError

from viana.config.job import JobSubmitRequest, ViAnaTaskParameters, LineSegment
from viana.io.paths import artifact_paths, project_output_dir


def test_project_id_validation():
    with pytest.raises(ValidationError):
        JobSubmitRequest(
            source_video_path="/data/v.mp4",
            project_id="INVALID SPACE",
            task_parameters=ViAnaTaskParameters(
                horizon_line=LineSegment(start=(0, 0), end=(100, 100)),
                counting_line=LineSegment(start=(0, 200), end=(100, 200)),
            ),
        )


def test_resume_and_start_fresh_mutually_exclusive():
    with pytest.raises(ValidationError):
        JobSubmitRequest(
            source_video_path="/data/v.mp4",
            project_id="nh48",
            resume=True,
            start_fresh=True,
            task_parameters=ViAnaTaskParameters(
                horizon_line=LineSegment(start=(0, 0), end=(100, 100)),
                counting_line=LineSegment(start=(0, 200), end=(100, 200)),
            ),
        )


def test_artifact_paths():
    from pathlib import Path

    out = project_output_dir(Path("/data/viana-outputs"), "nh48")
    paths = artifact_paths(out, "recording_20260315")
    assert paths["events"].name == "recording_20260315_events.csv"
    assert paths["aggregate_15min"].name == "recording_20260315_15min.csv"
    assert paths["processed_video"].name == "recording_20260315_processed.mp4"
