"""Validate the latest ITVA training checkpoint against itva_phase1."""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_YAML, TRAINING_OUTPUT  # noqa: E402

EXPERIMENT_NAME = "itva_phase1_1280p"


def main() -> None:
    weights = TRAINING_OUTPUT / EXPERIMENT_NAME / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    model = YOLO(str(weights))
    model.val(data=str(DATA_YAML), split="train", imgsz=1088, batch=4)


if __name__ == "__main__":
    main()
