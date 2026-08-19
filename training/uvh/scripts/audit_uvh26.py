"""Sample UVH-26 labels and export debug visualizations for rare classes."""

from __future__ import annotations

import glob
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DEBUG_PRETRAIN, UVH_RAW  # noqa: E402

INTEREST_CLASSES = ["Auto Rickshaw", "LCV", "Bus", "Three-wheeler"]


def load_yaml_classes(yaml_path: Path) -> dict[int, str]:
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names", {})
    if isinstance(names, dict):
        return names
    return {index: name for index, name in enumerate(names)}


def draw_yolo_box(img, class_name: str, x: float, y: float, w: float, h: float):
    dh, dw, _ = img.shape
    left = int((x - w / 2) * dw)
    right = int((x + w / 2) * dw)
    top = int((y - h / 2) * dh)
    bottom = int((y + h / 2) * dh)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(img, class_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main() -> None:
    yaml_path = UVH_RAW / "data.yaml"
    if not yaml_path.is_file():
        print(f"Expected {yaml_path} — download UVH-26 first.")
        return

    class_map = load_yaml_classes(yaml_path)
    DEBUG_PRETRAIN.mkdir(parents=True, exist_ok=True)
    label_files = glob.glob(str(UVH_RAW / "**/labels/**/*.txt"), recursive=True)
    hits: Counter[str] = Counter()

    for label_file in tqdm(label_files, desc="audit uvh26"):
        with open(label_file, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                name = class_map.get(cls_id, str(cls_id))
                if name not in INTEREST_CLASSES:
                    continue
                hits[name] += 1
                if random.random() > 0.02:
                    continue
                img_path = Path(label_file).with_suffix(".png")
                if not img_path.is_file():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                x, y, w, h = map(float, parts[1:5])
                draw_yolo_box(img, name, x, y, w, h)
                out = DEBUG_PRETRAIN / f"uvh26_{name}_{hits[name]}.png"
                cv2.imwrite(str(out), img)

    print("Interest class counts:", dict(hits))


if __name__ == "__main__":
    main()
