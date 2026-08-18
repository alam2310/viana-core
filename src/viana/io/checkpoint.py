"""Checkpoint JSON I/O for explicit resume (``checkpoint.schema.json``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from viana.io.paths import artifact_paths


class Checkpoint(BaseModel):
    """Engine resume state written to ``{stem}.checkpoint.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    project_id: str
    source_video_path: Path
    video_stem: str
    current_frame: int = Field(ge=0)
    total_frames: int = Field(ge=1)
    saved_at: str
    counted_track_ids: list[int] = Field(default_factory=list)
    events_rows_written: int = Field(default=0, ge=0)
    manifest_path: str | None = None

    @model_validator(mode="after")
    def frame_within_total(self) -> Checkpoint:
        """current_frame may equal total_frames when the last frame is done."""
        if self.current_frame > self.total_frames:
            raise ValueError("current_frame must be <= total_frames")
        return self

    def is_complete(self) -> bool:
        """True when the worker finished the last frame."""
        return self.current_frame >= self.total_frames


def utc_now_iso() -> str:
    """UTC timestamp with a trailing Z (JSON Schema date-time)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def checkpoint_path(output_dir: Path, video_stem: str) -> Path:
    """Return the standard checkpoint path for a video stem."""
    return artifact_paths(output_dir, video_stem)["checkpoint"]


def load_checkpoint(path: Path) -> Checkpoint:
    """Load and validate a checkpoint file. Does not resume a job."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return Checkpoint.model_validate(payload)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Write checkpoint JSON (overwrite). Resume remains an explicit CLI/API action."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
