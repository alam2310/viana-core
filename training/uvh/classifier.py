"""Map raw UVH-26 detector labels to ITVA taxonomy sub-classes."""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_TAXONOMY = Path(__file__).resolve().parent / "taxonomy" / "vehicle_taxonomy.json"


class VehicleClassifier:
    """Load vehicle_taxonomy.json and resolve raw labels to category / class / sub_class."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else _DEFAULT_TAXONOMY
        self.mapping = self._load_mapping(path)

    def _load_mapping(self, path: Path) -> dict[str, dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(f"Mapping file not found at {path}")
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def get_classification(self, raw_label: str) -> dict[str, str]:
        """Return the 3-level classification for a raw UVH label."""
        key = raw_label.lower().strip()
        fallback = {
            "category": "Unknown",
            "class_type": "Unknown",
            "sub_class": raw_label,
        }
        return self.mapping.get(key, fallback)
