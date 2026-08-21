from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from viana.io.telemetry import TelemetryMessage, emit_telemetry_stderr


def test_emit_telemetry_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """It emits a JSON line to stderr."""
    msg = TelemetryMessage(
        job_id="job_123",
        telemetry_type="PROGRESS",
        data={"progress": 0.5},
        status="PROCESSING",
    )
    emit_telemetry_stderr(msg)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith("\n")

    parsed = json.loads(captured.err)
    assert parsed["job_id"] == "job_123"
    assert parsed["telemetry_type"] == "PROGRESS"
    assert parsed["data"]["progress"] == 0.5
    assert parsed["status"] == "PROCESSING"


def test_telemetry_message_defaults() -> None:
    """It uses correct default values."""
    msg = TelemetryMessage(job_id="job_1", telemetry_type="LOG")
    assert msg.data == {}
    assert msg.status is None


def test_telemetry_message_validation() -> None:
    """Test validation rules for TelemetryMessage."""
    # Extra fields are forbidden
    with pytest.raises(ValidationError):
        TelemetryMessage(
            job_id="job_1",
            telemetry_type="LOG",
            extra_field="invalid",  # type: ignore[call-arg]
        )

    # Missing required field
    with pytest.raises(ValidationError):
        TelemetryMessage(
            telemetry_type="LOG",  # type: ignore[call-arg]
        )

    # Invalid status
    with pytest.raises(ValidationError):
        TelemetryMessage(
            job_id="job_1",
            telemetry_type="LOG",
            status="INVALID_STATUS",  # type: ignore[arg-type]
        )

    # Invalid type
    with pytest.raises(ValidationError):
        TelemetryMessage(
            job_id="job_1",
            telemetry_type="INVALID_TYPE",  # type: ignore[arg-type]
        )
