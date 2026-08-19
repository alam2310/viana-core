"""Build balanced ITVA Phase 1 dataset from YOLO-format UVH labels."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_YAML, ITVA_PHASE1, TAXONOMY_JSON, YOLO_FORMAT  # noqa: E402

TARGET_IDS = {
    "Car": 0,
    "Jeep": 1,
    "Van": 2,
    "Mini Bus": 3,
    "MTW": 4,
    "Auto": 5,
    "City Bus": 6,
    "Truck": 7,
    "LCV": 8,
    "Cycle": 9,
    "Others": 10,
}

RAW_CLASS_NAMES = [
    "Hatchback",
    "Sedan",
    "SUV",
    "MUV",
    "Bus",
    "Truck",
    "Three-wheeler",
    "Two-wheeler",
    "LCV",
    "Mini-bus",
    "Tempo-traveller",
    "Bicycle",
    "Van",
    "Others",
]


class ITVADatasetBuilder:
    def __init__(self) -> None:
        self._load_taxonomy()
        self.stats = dict.fromkeys(TARGET_IDS, 0)

    def _load_taxonomy(self) -> None:
        if not TAXONOMY_JSON.is_file():
            raise FileNotFoundError(f"Taxonomy config missing: {TAXONOMY_JSON}")
        with TAXONOMY_JSON.open(encoding="utf-8") as handle:
            self.taxonomy = json.load(handle)
        print(f"Loaded taxonomy: {len(self.taxonomy)} keys.")

    def get_target_id(self, raw_class_name: str) -> tuple[int, str]:
        key = raw_class_name.lower().strip()
        if key not in self.taxonomy:
            raise ValueError(f"Raw class '{raw_class_name}' not found in taxonomy JSON")
        sub_class = self.taxonomy[key]["sub_class"]
        if sub_class not in TARGET_IDS:
            raise ValueError(f"Sub-class '{sub_class}' is not in TARGET_IDS map")
        return TARGET_IDS[sub_class], sub_class

    def build(self) -> None:
        input_labels = list((YOLO_FORMAT / "labels/train").glob("*.txt"))
        if not input_labels:
            raise FileNotFoundError(f"No labels in {YOLO_FORMAT}/labels/train")

        if ITVA_PHASE1.exists():
            print("Output directory exists. Cleaning up...")
            shutil.rmtree(ITVA_PHASE1)

        (ITVA_PHASE1 / "images/train").mkdir(parents=True)
        (ITVA_PHASE1 / "labels/train").mkdir(parents=True)

        manifest_lines: list[str] = []
        print(f"Processing {len(input_labels)} label files...")
        for label_file in tqdm(input_labels):
            new_lines: list[str] = []
            rare_multiplier = 1

            with label_file.open(encoding="utf-8") as handle:
                lines = handle.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                raw_id = int(parts[0])
                if raw_id >= len(RAW_CLASS_NAMES):
                    continue
                raw_name = RAW_CLASS_NAMES[raw_id]
                coords = parts[1:]
                target_id, sub_class_name = self.get_target_id(raw_name)

                if sub_class_name in ("Mini Bus", "Van"):
                    rare_multiplier = max(rare_multiplier, 20)
                elif sub_class_name == "LCV":
                    rare_multiplier = max(rare_multiplier, 5)

                new_lines.append(f"{target_id} {' '.join(coords)}\n")
                self.stats[sub_class_name] += 1

            if not new_lines:
                continue

            file_stem = label_file.stem
            new_label_path = ITVA_PHASE1 / "labels/train" / f"{file_stem}.txt"
            with new_label_path.open("w", encoding="utf-8") as handle:
                handle.writelines(new_lines)

            src_img_dir = YOLO_FORMAT / "images/train"
            found_img = None
            for ext in (".png", ".jpg", ".jpeg"):
                potential = src_img_dir / f"{file_stem}{ext}"
                if potential.is_file():
                    found_img = potential
                    break

            if found_img:
                dst_img = ITVA_PHASE1 / "images/train" / found_img.name
                if not dst_img.exists():
                    os.symlink(found_img, dst_img)
                final_path = str(dst_img.absolute())
                manifest_lines.extend([final_path] * rare_multiplier)

        with (ITVA_PHASE1 / "train.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(manifest_lines))

        print("\nBuild complete.")
        self._generate_yaml()

    def _generate_yaml(self) -> None:
        yaml_data = {
            "path": str(ITVA_PHASE1.resolve()),
            "train": "train.txt",
            "val": "train.txt",
            "names": {v: k for k, v in TARGET_IDS.items()},
        }
        with DATA_YAML.open("w", encoding="utf-8") as handle:
            yaml.dump(yaml_data, handle, sort_keys=False)
        print(f"YAML saved to: {DATA_YAML}")


if __name__ == "__main__":
    ITVADatasetBuilder().build()
