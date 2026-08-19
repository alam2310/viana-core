"""Job configuration models (Pydantic). Synced with packages/contracts/schemas."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
GPU_DEVICE_PATTERN = re.compile(r"^cuda:[01]$")
USER_TIME_PATTERN = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
USER_DATE_PATTERN = re.compile(r"^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-[0-9]{4}$")

JobStatusLiteral = Literal[
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


class LineSegment(BaseModel):
    """Two pixel endpoints within frame bounds (validated against video dimensions)."""

    model_config = ConfigDict(extra="forbid")

    start: tuple[int, int]
    end: tuple[int, int]

    @field_validator("start", "end")
    @classmethod
    def non_negative_pixels(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Reject negative pixel coordinates (JSON schema Point minimum 0)."""
        if value[0] < 0 or value[1] < 0:
            raise ValueError("line coordinates must be >= 0")
        return value

    @model_validator(mode="after")
    def endpoints_differ(self) -> LineSegment:
        """Reject a zero-length line."""
        if self.start == self.end:
            raise ValueError("line endpoints must differ")
        return self


class JobMetadata(BaseModel):
    """Optional user-supplied metadata (OCR fallback / report headers)."""

    model_config = ConfigDict(extra="forbid")

    user_start_time: str | None = None
    user_start_date: str | None = None
    location: str | None = None


class ConfirmedJobMetadata(BaseModel):
    """Mandatory metadata after prescan review (HH:MM:SS + DD-MM-YYYY + location)."""

    model_config = ConfigDict(extra="forbid")

    user_start_time: str = Field(min_length=1)
    user_start_date: str = Field(min_length=1)
    location: str = Field(min_length=1)

    @field_validator("user_start_time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Require 24-hour wall-clock time ``HH:MM:SS``."""
        if not USER_TIME_PATTERN.match(value):
            raise ValueError("user_start_time must match HH:MM:SS")
        return value

    @field_validator("user_start_date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Require Indian traffic convention date ``DD-MM-YYYY``."""
        if not USER_DATE_PATTERN.match(value):
            raise ValueError("user_start_date must match DD-MM-YYYY")
        return value


class ProposedLines(BaseModel):
    """Prescan line proposal with confidence score."""

    model_config = ConfigDict(extra="forbid")

    horizon_line: LineSegment
    counting_line: LineSegment
    confidence: float = Field(ge=0.0, le=1.0)


class ViAnaTaskParameters(BaseModel):
    """Per-job CV parameters for ViAna Moving Count."""

    model_config = ConfigDict(extra="forbid")

    horizon_line: LineSegment
    counting_line: LineSegment
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    use_heuristic_truck_split: bool = True
    render_video: bool = True
    telemetry_detail: bool = False


class JobSubmitRequest(BaseModel):
    """Payload from UI to orchestrator. job_id and gpu_device are assigned by backend."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["ViAna_Moving"] = "ViAna_Moving"
    source_video_path: Path
    project_id: str
    output_dir: Path | None = None
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


class JobIntakeRequest(BaseModel):
    """POST /jobs/intake — register videos for backend prescan."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["ViAna_Moving"] = "ViAna_Moving"
    project_id: str
    source_video_paths: list[Path] = Field(min_length=1)
    output_dir: Path | None = None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        """Ensure project_id is a lowercase slug safe for directory names."""
        if not PROJECT_ID_PATTERN.match(value):
            raise ValueError("project_id must match [a-z0-9][a-z0-9_-]*")
        return value


class JobPrescanConfirmRequest(BaseModel):
    """PATCH /jobs/{id}/prescan — operator confirms reviewed calibration."""

    model_config = ConfigDict(extra="forbid")

    metadata: ConfirmedJobMetadata
    task_parameters: ViAnaTaskParameters
    calibration_profile_id: str | None = None


class JobConfig(JobSubmitRequest):
    """Engine CLI config: submit payload plus backend-assigned runtime fields."""

    job_id: str = Field(min_length=1)
    gpu_device: str
    output_dir: Path

    @field_validator("gpu_device")
    @classmethod
    def validate_gpu_device(cls, value: str) -> str:
        """Restrict assignment to the two locked GPU slots."""
        if not GPU_DEVICE_PATTERN.match(value):
            raise ValueError("gpu_device must match cuda:0 or cuda:1")
        return value

    def validate_geometry(self, frame_width: int, frame_height: int) -> None:
        """Require both calibration lines inside the video frame (v2 vs legacy off-screen)."""
        if frame_width < 1 or frame_height < 1:
            raise ValueError("frame dimensions must be positive")
        params = self.task_parameters
        params.horizon_line.assert_within_frame(frame_width, frame_height, "horizon_line")
        params.counting_line.assert_within_frame(frame_width, frame_height, "counting_line")


class JobSubmitResponse(BaseModel):
    """Orchestrator response after accepting a job submit request."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatusLiteral
    gpu_device: str
    queue_position: int
    output_dir: Path

    @field_validator("gpu_device")
    @classmethod
    def validate_gpu_device(cls, value: str) -> str:
        """Restrict assignment to the two locked GPU slots."""
        if not GPU_DEVICE_PATTERN.match(value):
            raise ValueError("gpu_device must match cuda:0 or cuda:1")
        return value


class JobIntakeItem(BaseModel):
    """One job created by POST /jobs/intake."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["PRESCAN_PENDING"] = "PRESCAN_PENDING"
    source_video_path: str
    output_dir: str
    queue_position: int = Field(ge=0)


class JobIntakeResponse(BaseModel):
    """POST /jobs/intake response."""

    model_config = ConfigDict(extra="forbid")

    jobs: list[JobIntakeItem] = Field(min_length=1)


def load_job_config(path: Path) -> JobConfig:
    """Load and validate a JobConfig JSON file for ``viana run`` / ``viana resume``."""
    if not path.is_file():
        raise FileNotFoundError(f"JobConfig not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return JobConfig.model_validate(payload)
