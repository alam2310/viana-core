"""YOLO label dataset auditor — class histograms and sample visualizations."""

from __future__ import annotations

import glob
import os
import random
from collections import Counter

import cv2
import yaml
from tqdm import tqdm


class DatasetAuditor:
    def __init__(self, data_yaml_path: str) -> None:
        self.yaml_path = data_yaml_path
        self.classes: list[str] = []
        self.dataset_root = os.path.dirname(data_yaml_path)
        self._load_yaml()

    def _load_yaml(self) -> None:
        with open(self.yaml_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        names = data["names"]
        self.classes = list(names.values()) if isinstance(names, dict) else names
        print(f"Loaded {len(self.classes)} classes from {self.yaml_path}")

    def scan_labels(self, split: str = "train") -> tuple[Counter[int], list[str]]:
        labels_path = os.path.join(self.dataset_root, "labels", split)
        if not os.path.exists(labels_path):
            labels_path = os.path.join(self.dataset_root, split, "labels")
        label_files = glob.glob(os.path.join(labels_path, "*.txt"))
        if not label_files:
            raise FileNotFoundError("No .txt label files found")

        class_counts: Counter[int] = Counter()
        for file in tqdm(label_files, desc="Counting classes"):
            with open(file, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1
        return class_counts, label_files

    def generate_visual_samples(
        self, label_files: list[str], output_dir: str, num_samples: int = 5
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        if not label_files:
            return

        samples = random.sample(label_files, min(num_samples, len(label_files)))
        label_dir = os.path.dirname(label_files[0])
        if "/labels/" in label_dir:
            img_dir = label_dir.replace("/labels/", "/images/")
        elif "labels" in label_dir:
            img_dir = label_dir.replace("labels", "images")
        else:
            print(f"Could not infer image directory from {label_dir}")
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        image_map = {
            os.path.splitext(name)[0]: name
            for name in os.listdir(img_dir)
            if os.path.splitext(name)[1].lower() in valid_exts
        }

        for label_file in samples:
            base_id = os.path.splitext(os.path.basename(label_file))[0]
            if base_id not in image_map:
                continue
            img_path = os.path.join(img_dir, image_map[base_id])
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            with open(label_file, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
                    cv2.putText(
                        img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
            save_path = os.path.join(output_dir, f"audit_{image_map[base_id]}")
            cv2.imwrite(save_path, img)
            print(f"Saved: {save_path}")
