"""Calibration profile JSON on disk (HTTP layer; does not import viana.domain)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.models import CalibrationProfile
from viana.io.paths import profiles_dir


def _now_iso() -> str:
    """UTC timestamp with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def profile_path(output_dir: Path, profile_id: str) -> Path:
    """Return ``{output_dir}/profiles/{profile_id}.json``."""
    return profiles_dir(output_dir) / f"{profile_id}.json"


def save_profile(output_dir: Path, profile: CalibrationProfile) -> CalibrationProfile:
    """Write a profile JSON file, filling created_at when missing."""
    stored = profile
    if stored.created_at is None:
        stored = stored.model_copy(update={"created_at": _now_iso()})
    path = profile_path(output_dir, stored.profile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return stored


def list_profiles(output_dir: Path) -> list[CalibrationProfile]:
    """Load all ``*.json`` profiles under the project profiles directory."""
    directory = profiles_dir(output_dir)
    if not directory.is_dir():
        return []
    profiles: list[CalibrationProfile] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles.append(CalibrationProfile.model_validate(payload))
    return profiles
