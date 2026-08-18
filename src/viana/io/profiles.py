"""Calibration profile JSON I/O (``calibration_profile.schema.json``)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from viana.config.job import LineSegment
from viana.domain.geometry import scale_line
from viana.io.checkpoint import utc_now_iso
from viana.io.paths import profiles_dir

PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ProfileSource = Literal["user_drawn", "auto_proposed", "user_edited"]


class CalibrationProfile(BaseModel):
    """Reusable horizon + counting lines for a project."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    profile_name: str
    reference_resolution: tuple[int, int]
    horizon_line: LineSegment
    counting_line: LineSegment
    created_at: str | None = None
    source: ProfileSource | None = None

    @field_validator("reference_resolution")
    @classmethod
    def resolution_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Reject non-positive reference pixels."""
        if value[0] < 1 or value[1] < 1:
            raise ValueError("reference_resolution must be at least 1x1")
        return value

    def assert_id(self) -> None:
        """Raise ValueError when ``profile_id`` is not a slug."""
        if not PROFILE_ID_PATTERN.match(self.profile_id):
            raise ValueError("profile_id must match [a-z0-9][a-z0-9_-]*")

    def scaled_to(self, width: int, height: int) -> CalibrationProfile:
        """Return a copy with lines mapped onto ``width`` × ``height``."""
        from_size = (self.reference_resolution[0], self.reference_resolution[1])
        to_size = (width, height)
        return self.model_copy(
            update={
                "horizon_line": scale_line(self.horizon_line, from_size, to_size),
                "counting_line": scale_line(self.counting_line, from_size, to_size),
                "reference_resolution": (width, height),
            }
        )


def profile_path(output_dir: Path, profile_id: str) -> Path:
    """Return ``{output_dir}/profiles/{profile_id}.json``."""
    return profiles_dir(output_dir) / f"{profile_id}.json"


def load_profile(path: Path) -> CalibrationProfile:
    """Load and validate one calibration profile file."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration profile not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    profile = CalibrationProfile.model_validate(payload)
    profile.assert_id()
    return profile


def save_profile(output_dir: Path, profile: CalibrationProfile) -> Path:
    """Write ``{output_dir}/profiles/{profile_id}.json`` (overwrite)."""
    profile.assert_id()
    if profile.created_at is None:
        profile = profile.model_copy(update={"created_at": utc_now_iso()})
    path = profile_path(output_dir, profile.profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return path


def list_profiles(output_dir: Path) -> list[CalibrationProfile]:
    """Load all ``*.json`` profiles under the project profiles directory."""
    directory = profiles_dir(output_dir)
    if not directory.is_dir():
        return []
    profiles: list[CalibrationProfile] = []
    for path in sorted(directory.glob("*.json")):
        profiles.append(load_profile(path))
    return profiles


def parse_created_at(value: str | None) -> datetime:
    """Parse ISO created_at; missing values sort as oldest."""
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
