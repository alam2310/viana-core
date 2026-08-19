"""Phase 1 — JobConfig validation synced with job_config.schema.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from viana.config.files import contracts_schemas_dir, repo_root
from viana.config.job import (
    JobConfig,
    JobSubmitRequest,
    LineSegment,
    ViAnaTaskParameters,
    load_job_config,
)


def _fixture_job_config() -> Path:
    return repo_root() / "packages" / "contracts" / "fixtures" / "job_config.json"


def _submit_kwargs() -> dict[str, object]:
    return {
        "source_video_path": "/data/v.mp4",
        "project_id": "nh48",
        "task_parameters": ViAnaTaskParameters(
            horizon_line=LineSegment(start=(0, 0), end=(100, 100)),
            counting_line=LineSegment(start=(0, 200), end=(100, 200)),
        ),
    }


def test_job_submit_fields_match_schema() -> None:
    """JobSubmitRequest fields must equal job_submit.schema.json properties."""
    schema = json.loads((contracts_schemas_dir() / "job_submit.schema.json").read_text())
    assert set(JobSubmitRequest.model_fields) == set(schema["properties"])


def test_job_config_fields_match_schema() -> None:
    """JobConfig fields must equal job_config.schema.json properties."""
    schema = json.loads((contracts_schemas_dir() / "job_config.schema.json").read_text())
    assert set(JobConfig.model_fields) == set(schema["properties"])


def test_load_job_config_fixture() -> None:
    """Load the committed JobConfig fixture."""
    job = load_job_config(_fixture_job_config())
    assert job.job_id == "job_mock_001"
    assert job.gpu_device == "cuda:0"
    assert job.project_id == "nh48"
    assert job.task_parameters.confidence_threshold == 0.75


def test_job_submit_rejects_backend_owned_fields() -> None:
    """UI payload must not include job_id (schema additionalProperties false)."""
    with pytest.raises(ValidationError):
        JobSubmitRequest.model_validate(
            {
                **_submit_kwargs(),
                "job_id": "job_should_not_be_here",
            }
        )


def test_job_config_rejects_unknown_gpu() -> None:
    """Only cuda:0 and cuda:1 are valid GPU slots."""
    payload = json.loads(_fixture_job_config().read_text())
    payload["gpu_device"] = "cuda:2"
    with pytest.raises(ValidationError, match="cuda"):
        JobConfig.model_validate(payload)


def test_geometry_within_frame() -> None:
    """Calibration lines must sit inside pixel bounds."""
    job = load_job_config(_fixture_job_config())
    job.validate_geometry(1920, 1080)
    with pytest.raises(ValueError, match=r"outside \d+x\d+ frame"):
        job.validate_geometry(100, 100)


def test_degenerate_line_rejected() -> None:
    """Zero-length counting lines are invalid."""
    with pytest.raises(ValidationError, match="endpoints must differ"):
        LineSegment(start=(10, 10), end=(10, 10))


def test_load_job_config_missing_file(tmp_path: Path) -> None:
    """Missing CLI config files fail closed."""
    with pytest.raises(FileNotFoundError):
        load_job_config(tmp_path / "missing.json")
