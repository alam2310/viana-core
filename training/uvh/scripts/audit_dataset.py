"""Audit YOLO label distribution and write sample images."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_auditor import DatasetAuditor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import REPO_ROOT, YOLO_FORMAT  # noqa: E402

DATA_YAML = YOLO_FORMAT / "data.yaml"
OUTPUT_DIR = REPO_ROOT / "data/outputs/audit_samples"


def main() -> None:
    if not DATA_YAML.is_file():
        print(f"data.yaml not found: {DATA_YAML}")
        print("Run convert_coco_to_yolo.py first.")
        return

    auditor = DatasetAuditor(str(DATA_YAML))
    counts, label_files = auditor.scan_labels(split="train")

    total = sum(counts.values())
    print("\nDATASET DISTRIBUTION")
    print(f"{'ID':<5} | {'Class':<20} | {'Count':<10} | {'Share %':<10}")
    print("-" * 55)
    for cls_id, name in enumerate(auditor.classes):
        count = counts.get(cls_id, 0)
        share = (count / total * 100) if total else 0
        print(f"{cls_id:<5} | {name:<20} | {count:<10} | {share:.1f}%")

    auditor.generate_visual_samples(label_files, str(OUTPUT_DIR), num_samples=10)


if __name__ == "__main__":
    main()
