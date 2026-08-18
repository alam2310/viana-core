"""Load YAML mappings with ``yaml.safe_load``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and return a mapping.

    Args:
        path: Absolute or relative path to a YAML document.

    Returns:
        Top-level mapping from the file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document is empty or not a mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload
