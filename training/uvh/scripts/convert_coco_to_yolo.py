"""Convert UVH-26 COCO annotations to YOLO format."""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DEBUG_PRETRAIN, REPO_ROOT, UVH_RAW, YOLO_FORMAT  # noqa: E402

TRAIN_JSON = UVH_RAW / "UVH-26-Train/UVH-26-MV-Train.json"
VAL_JSON = UVH_RAW / "UVH-26-Val/UVH-26-MV-Val.json"


def convert_coco_to_yolo(json_file: Path, split_name: str) -> tuple[list[str], Counter[str]]:
    if not json_file.is_file():
        print(f"Missing annotations: {json_file}")
        return [], Counter()

    with json_file.open(encoding="utf-8") as handle:
        data = json.load(handle)

    cats = sorted(data["categories"], key=lambda item: item["id"])
    cat_map = {cat["id"]: index for index, cat in enumerate(cats)}
    class_names = [cat["name"] for cat in cats]
    images = {img["id"]: img for img in data["images"]}

    search_root = UVH_RAW / f"UVH-26-{split_name.capitalize()}"
    found_files = {path.name: path for path in search_root.rglob("*.png")}

    class_counts: Counter[str] = Counter()
    missing = 0
    for ann in tqdm(data["annotations"], desc=f"convert {split_name}"):
        img_info = images.get(ann["image_id"])
        if not img_info:
            continue
        file_name = Path(img_info["file_name"]).name
        src_path = found_files.get(file_name)
        if src_path is None:
            missing += 1
            continue

        dst_img = YOLO_FORMAT / "images" / split_name / file_name
        dst_lbl = YOLO_FORMAT / "labels" / split_name / f"{Path(file_name).stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        img_w, img_h = img_info["width"], img_info["height"]
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        yolo_cls = cat_map[ann["category_id"]]
        class_counts[class_names[yolo_cls]] += 1

        with dst_lbl.open("a", encoding="utf-8") as handle:
            handle.write(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    if missing:
        print(f"Missing images in {split_name}: {missing}")
    return class_names, class_counts


def main() -> None:
    if YOLO_FORMAT.exists():
        shutil.rmtree(YOLO_FORMAT)
    for split in ("train", "val"):
        (YOLO_FORMAT / "images" / split).mkdir(parents=True)
        (YOLO_FORMAT / "labels" / split).mkdir(parents=True)
    DEBUG_PRETRAIN.mkdir(parents=True, exist_ok=True)

    classes_train, counts_train = convert_coco_to_yolo(TRAIN_JSON, "train")
    _classes_val, counts_val = convert_coco_to_yolo(VAL_JSON, "val")
    if not classes_train:
        return

    yaml_lines = [
        f"path: {YOLO_FORMAT}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(classes_train))
    (YOLO_FORMAT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Wrote {YOLO_FORMAT / 'data.yaml'}")

    total = sum(counts_train.values()) + sum(counts_val.values())
    print("\nDATASET AUDIT")
    for cls in classes_train:
        count = counts_train[cls] + counts_val.get(cls, 0)
        pct = (count / total * 100) if total else 0
        print(f"  {cls:<20} {count:>8}  ({pct:.1f}%)")

    train_imgs = list((YOLO_FORMAT / "images/train").glob("*.png"))
    if train_imgs:
        for img_path in random.sample(train_imgs, min(3, len(train_imgs))):
            img = cv2.imread(str(img_path))
            lbl_path = YOLO_FORMAT / "labels/train" / f"{img_path.stem}.txt"
            if lbl_path.is_file() and img is not None:
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    dh, dw, _ = img.shape
                    left = int((x - w / 2) * dw)
                    right = int((x + w / 2) * dw)
                    top = int((y - h / 2) * dh)
                    bottom = int((y + h / 2) * dh)
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        classes_train[cls_id],
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            out = DEBUG_PRETRAIN / f"debug_{img_path.name}"
            cv2.imwrite(str(out), img)
            print(f"Debug image: {out}")


if __name__ == "__main__":
    main()
