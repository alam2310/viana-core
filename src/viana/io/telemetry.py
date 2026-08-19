"""WebSocket-shaped telemetry (``telemetry.schema.json``) emitted by the engine CLI."""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobRuntimeStatus = Literal[
    "PRESCAN_PENDING",
    "PRESCAN_RUNNING",
    "PRESCAN_FAILED",
    "AWAITING_REVIEW",
    "READY",
    "PROCESSING",
    "PAUSED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]
TelemetryType = Literal["PROGRESS", "MOVING_EVENT", "LOG"]


class TelemetryMessage(BaseModel):
    """One orchestrator/UI telemetry record."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    telemetry_type: TelemetryType
    data: dict[str, Any] = Field(default_factory=dict)
    status: JobRuntimeStatus | None = None


def emit_telemetry_stderr(message: TelemetryMessage) -> None:
    """Write one JSON line to stderr (stdout stays the final RunResult)."""
    sys.stderr.write(json.dumps(message.model_dump(mode="json"), separators=(",", ":")) + "\n")
    sys.stderr.flush()
