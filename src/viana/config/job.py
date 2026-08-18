"""Job configuration models (Pydantic). Phase 1: full validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class LineSegment(BaseModel):
    """Two pixel endpoints within frame bounds (validated at run time against video dimensions)."""

    start: tuple[int, int]
    end: tuple[int, int]


class JobMetadata(BaseModel):
    """Optional user-supplied metadata (OCR fallback / report headers)."""

    user_start_time: str | None = None
    user_start_date: str | None = None
    location: str | None = None


class ViAnaTaskParameters(BaseModel):
    """Per-job CV parameters for ViAna Moving Count."""

    horizon_line: LineSegment
    counting_line: LineSegment
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    use_heuristic_truck_split: bool = True
    render_video: bool = True
    telemetry_detail: bool = False


class JobSubmitRequest(BaseModel):
    """Payload from UI to orchestrator. job_id and gpu_device are assigned by backend."""

    task_type: Literal["ViAna_Moving"] = "ViAna_Moving"
    source_video_path: Path
    project_id: str
    metadata: JobMetadata = Field(default_factory=JobMetadata)
    task_parameters: ViAnaTaskParameters
    calibration_profile_id: str | None = None
    resume: bool = False
    start_fresh: bool = False

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        """Ensure project_id is a lowercase slug safe for directory names."""
        if not PROJECT_ID_PATTERN.match(value):
            raise ValueError("project_id must match [a-z0-9][a-z0-9_-]*")
        return value

    @model_validator(mode="after")
    def resume_xor_fresh(self) -> JobSubmitRequest:
        """Disallow ambiguous resume intent when both flags are set."""
        if self.resume and self.start_fresh:
            raise ValueError("resume and start_fresh are mutually exclusive")
        return self


class JobSubmitResponse(BaseModel):
    """Orchestrator response after accepting a job submit request."""

    job_id: str
    status: Literal["PENDING", "PROCESSING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"]
    gpu_device: str
    queue_position: int
    output_dir: Path
