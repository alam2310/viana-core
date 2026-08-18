"""Orchestrator runtime settings (output root)."""

from __future__ import annotations

import os
from pathlib import Path

from viana.config.defaults import load_engine_defaults


def output_parent() -> Path:
    """Return artifact root: ``VIANA_OUTPUT_PARENT`` or engine defaults."""
    env = os.environ.get("VIANA_OUTPUT_PARENT")
    if env:
        return Path(env)
    return load_engine_defaults().output.parent_dir


def project_dir(project_id: str) -> Path:
    """Return ``{output_parent}/{project_id}/``."""
    from viana.io.paths import project_output_dir

    return project_output_dir(output_parent(), project_id)
