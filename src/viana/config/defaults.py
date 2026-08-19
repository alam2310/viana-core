"""Engine defaults loaded from ``configs/engine_defaults.yaml``."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from viana.config.files import resolve_configs_dir
from viana.config.job import ViAnaTaskParameters
from viana.config.yaml_io import load_yaml_mapping


class ModelPaths(BaseModel):
    """Relative or absolute paths to production and pedestrian weights."""

    model_config = ConfigDict(extra="forbid")

    vehicle: Path
    pedestrian: Path


class DetectionDefaults(BaseModel):
    """Detector thresholds (job may override ``confidence_threshold``)."""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(ge=0.0, le=1.0)
    imgsz: int = Field(ge=1)
    nms_threshold: float = Field(ge=0.0, le=1.0)
    suppression_ioa: float = Field(ge=0.0, le=1.0)


class ClassificationDefaults(BaseModel):
    """Heuristic truck-split and class-lock parameters."""

    model_config = ConfigDict(extra="forbid")

    use_heuristic_truck_split: bool
    lock_frames: int = Field(ge=1)
    perspective_scale: float
    trailer_ratio: float
    lcv_max_area: float = Field(ge=0.0)
    mcv_max_area: float = Field(ge=0.0)


class OcrDefaults(BaseModel):
    """On-screen clock OCR and time-map recalibration."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(ge=0.0, le=1.0)
    recalibration_interval_sec: float = Field(ge=0.0)
    drift_threshold_sec: float = Field(ge=0.0)


class PipelineDefaults(BaseModel):
    """Checkpoint and telemetry cadence (frames)."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_interval_frames: int = Field(ge=1)
    telemetry_progress_frames: int = Field(ge=1)
    telemetry_detail_progress_frames: int = Field(ge=1)


class OutputDefaults(BaseModel):
    """Artifact root; jobs write under ``{parent_dir}/{project_id}/``."""

    model_config = ConfigDict(extra="forbid")

    parent_dir: Path


class PrescanDefaults(BaseModel):
    """Prescan sampler tuning (dark-frame skip, scrub preview)."""

    model_config = ConfigDict(extra="forbid")

    dark_frame_luminance_threshold: float = Field(ge=0.0, le=255.0)
    dark_frame_scan_sec: float = Field(ge=0.0)
    dark_frame_step_sec: float = Field(gt=0.0)
    osd_min_score: float = Field(ge=0.0)
    osd_probe_start_sec: float = Field(ge=0.0)


class EngineDefaults(BaseModel):
    """Validated engine defaults (overridable per job)."""

    model_config = ConfigDict(extra="forbid")

    models: ModelPaths
    detection: DetectionDefaults
    classification: ClassificationDefaults
    ocr: OcrDefaults
    pipeline: PipelineDefaults
    prescan: PrescanDefaults
    output: OutputDefaults

    def apply_task_overrides(self, params: ViAnaTaskParameters) -> EngineDefaults:
        """Return a copy with job-level detection/classification overrides applied."""
        updated = self.model_copy(deep=True)
        updated.detection.confidence_threshold = params.confidence_threshold
        updated.classification.use_heuristic_truck_split = params.use_heuristic_truck_split
        return updated


def load_engine_defaults(path: Path | None = None) -> EngineDefaults:
    """Load and validate ``engine_defaults.yaml``.

    Args:
        path: Explicit file path. When omitted, ``<configs>/engine_defaults.yaml``.
    """
    yaml_path = path if path is not None else resolve_configs_dir() / "engine_defaults.yaml"
    payload = load_yaml_mapping(yaml_path)
    return EngineDefaults.model_validate(payload)
