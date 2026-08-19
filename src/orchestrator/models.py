"""HTTP models aligned with packages/contracts/schemas.

Request bodies reject extra fields so clients cannot send backend-owned
`job_id` or `gpu_device` on submit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from viana.config.job import (
    GPU_DEVICE_PATTERN,
    PROJECT_ID_PATTERN,
    JobMetadata,
    JobStatusLiteral,
    LineSegment,
    ProposedLines,
    ViAnaTaskParameters,
)
from viana.config.job import (
    JobSubmitRequest as EngineJobSubmitRequest,
)


class JobSubmitRequest(EngineJobSubmitRequest):
    """POST /jobs body. extra=forbid matches job_submit.schema.json."""

    model_config = ConfigDict(extra="forbid")


class JobSubmitResponse(BaseModel):
    """POST /jobs 201 body — job_submit_response.schema.json."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatusLiteral
    gpu_device: str
    queue_position: int
    output_dir: str

    @field_validator("gpu_device")
    @classmethod
    def validate_gpu_device(cls, value: str) -> str:
        """Restrict assignment to the two locked GPU slots."""
        if not GPU_DEVICE_PATTERN.match(value):
            raise ValueError("gpu_device must match cuda:0 or cuda:1")
        return value


class JobProgress(BaseModel):
    """Optional progress block on JobStatus."""

    current_frame: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    processing_fps: float | None = Field(default=None, ge=0)
    eta_sec: float | None = Field(default=None, ge=0)
    crossing_count: int | None = Field(default=None, ge=0)


class JobStatus(BaseModel):
    """GET /jobs/{id} body — job_status.schema.json."""

    job_id: str
    status: JobStatusLiteral
    task_type: Literal["ViAna_Moving"]
    source_video_path: str
    project_id: str
    output_dir: str
    checkpoint_exists: bool
    gpu_device: str | None = Field(default=None, pattern=r"^cuda:[01]$")
    queue_position: int | None = Field(default=None, ge=0)
    progress: JobProgress | None = None
    error_message: str | None = None
    proposed_metadata: JobMetadata | None = None
    proposed_lines: ProposedLines | None = None
    proposed_preview_url: str | None = None
    confirmed_metadata: JobMetadata | None = None
    confirmed_task_parameters: ViAnaTaskParameters | None = None
    created_at: str
    video_duration_sec: float | None = Field(default=None, ge=0)
    processing_duration_sec: float | None = Field(default=None, ge=0)


class PrescanRequest(BaseModel):
    """POST /utils/prescan body (openapi.yaml PrescanRequest)."""

    model_config = ConfigDict(extra="forbid")

    source_video_path: str = Field(min_length=1)
    project_id: str
    task_type: Literal["ViAna_Moving"] = "ViAna_Moving"
    frame_offset_sec: float = 0.0

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        """Ensure project_id is a lowercase slug safe for directory names."""
        if not PROJECT_ID_PATTERN.match(value):
            raise ValueError("project_id must match [a-z0-9][a-z0-9_-]*")
        return value


class TelemetryMessage(BaseModel):
    """WS /ws/jobs payload — telemetry.schema.json."""

    job_id: str
    telemetry_type: Literal["PROGRESS", "MOVING_EVENT", "LOG"]
    data: dict[str, object]
    status: JobStatusLiteral | None = None


class CalibrationProfile(BaseModel):
    """GET/POST profiles body — calibration_profile.schema.json."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    profile_name: str
    reference_resolution: tuple[int, int]
    horizon_line: LineSegment
    counting_line: LineSegment
    created_at: str | None = None
    source: Literal["user_drawn", "auto_proposed", "user_edited"] | None = None
