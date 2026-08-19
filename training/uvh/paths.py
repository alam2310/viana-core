"""Shared paths for UVH retraining (repo root or /app/ViAna in container)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TAXONOMY_JSON = REPO_ROOT / "training/uvh/taxonomy/vehicle_taxonomy.json"
DATA_YAML = REPO_ROOT / "training/uvh/configs/itva_phase1.yaml"
UVH_RAW = REPO_ROOT / "data/datasets/uvh26"
YOLO_FORMAT = REPO_ROOT / "data/processed/yolo_format"
ITVA_PHASE1 = REPO_ROOT / "data/itva_phase1"
TRAINING_OUTPUT = REPO_ROOT / "data/outputs/training"
DEBUG_PRETRAIN = REPO_ROOT / "data/outputs/debug_pretrain"
PRETRAINED_WEIGHTS = REPO_ROOT / "models/pretrained/yolo11m.pt"
PRODUCTION_WEIGHTS = REPO_ROOT / "models/v1/itva_medium_1088p.pt"
