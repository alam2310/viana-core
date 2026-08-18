"""Final job outcome JSON (``run_result.schema.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from viana.io.checkpoint import utc_now_iso


class RunResultArtifacts(BaseModel):
    """Optional artifact paths recorded when a file was written."""

    model_config = ConfigDict(extra="forbid")

    events: str | None = None
    aggregate_15min: str | None = None
    processed_video: str | None = None
    manifest: str | None = None
    time_map: str | None = None


class RunResult(BaseModel):
    """``{stem}.run_result.json`` — engine outcome for the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    source_video_path: str
    video_stem: str
    artifacts: RunResultArtifacts
    completed_at: str
    error_message: str | None = None


def save_run_result(path: Path, result: RunResult) -> None:
    """Write run_result JSON (overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def load_run_result(path: Path) -> RunResult:
    """Load and validate a run_result file."""
    if not path.is_file():
        raise FileNotFoundError(f"Run result not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return RunResult.model_validate(payload)


def completed_now(
    job_id: str,
    source_video_path: Path,
    video_stem: str,
    artifacts: RunResultArtifacts,
    *,
    status: Literal["COMPLETED", "FAILED", "CANCELLED"] = "COMPLETED",
    error_message: str | None = None,
) -> RunResult:
    """Build a RunResult stamped with the current UTC time."""
    return RunResult(
        job_id=job_id,
        status=status,
        source_video_path=str(source_video_path),
        video_stem=video_stem,
        artifacts=artifacts,
        completed_at=utc_now_iso(),
        error_message=error_message,
    )
