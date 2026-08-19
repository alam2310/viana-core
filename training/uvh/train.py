"""YOLO11 ITVA training entry (UVH-26 → itva_phase1 manifest)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_YAML, PRETRAINED_WEIGHTS, TRAINING_OUTPUT  # noqa: E402

EXPERIMENT_NAME = "itva_phase1_1280p"


def run_training() -> None:
    """Train ITVA medium detector at 1088p with dual-GPU DDP when available."""
    if not torch.cuda.is_available():
        print("CUDA not detected. Training requires a GPU.")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    devices: list[int] | int = list(range(gpu_count)) if gpu_count > 1 else 0
    print(f"Starting training on {gpu_count} GPU(s).")
    print(f"  Weights: {PRETRAINED_WEIGHTS}")
    print(f"  Data:    {DATA_YAML}")

    model = YOLO(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        project=str(TRAINING_OUTPUT),
        name=EXPERIMENT_NAME,
        exist_ok=True,
        device=devices,
        batch=12,
        workers=8,
        imgsz=1088,
        epochs=30,
        mosaic=1.0,
        mixup=0.15,
        scale=0.5,
        degrees=0.0,
        fliplr=0.5,
        box=7.5,
        val=True,
        save=True,
    )
    print(f"\nTraining complete. Best weights: {TRAINING_OUTPUT}/{EXPERIMENT_NAME}/weights/best.pt")


if __name__ == "__main__":
    run_training()
