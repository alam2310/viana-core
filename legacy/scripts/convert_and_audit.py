import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import random
from collections import Counter

# --- CONFIGURATION ---
ROOT_DIR = Path("/root/Work/ViAna/data/raw/UVH-26")
TRAIN_JSON = ROOT_DIR / "UVH-26-Train/UVH-26-MV-Train.json"
VAL_JSON = ROOT_DIR / "UVH-26-Val/UVH-26-MV-Val.json"
OUTPUT_DIR = Path("/root/Work/ViAna/data/processed/yolo_format")
DEBUG_DIR = Path("/root/Work/ViAna/debug_pretrain")

# Create output directories
for split in ['train', 'val']:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def convert_coco_to_yolo(json_file, split_name):
    print(f"📂 Loading {split_name} annotations from {json_file}...")
    if not json_file.exists():
        print(f"❌ Error: {json_file} not found!")
        return [], {}

    with open(json_file, 'r') as f:
        data = json.load(f)

    # 1. Map Categories
    cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_map = {c['id']: i for i, c in enumerate(cats)}
    class_names = [c['name'] for c in cats]
    
    # 2. Index Images
    print(f"🔍 Indexing {split_name} images...")
    images = {img['id']: img for img in data['images']}
    
    # 3. Pre-scan for actual file paths (Optimization)
    # We scan the specific split folder to avoid searching the whole drive
    search_root = ROOT_DIR / f"UVH-26-{split_name.capitalize()}"
    print(f"   Scanning {search_root} for PNGs...")
    # Create a map of filename -> full_path
    found_files = {p.name: p for p in search_root.rglob("*.png")}
    print(f"   Found {len(found_files)} physical PNG files.")

    # 4. Process Annotations
    print(f"🔄 Converting annotations...")
    class_counts = Counter()
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns: img_anns[img_id] = []
        img_anns[img_id].append(ann)

    missing_count = 0
    
    for img_id, anns in tqdm(img_anns.items()):
        img_info = images.get(img_id)
        if not img_info: continue

        src_filename = img_info['file_name'] # e.g., "12345.png"
        
        # Find the file
        src_path = found_files.get(src_filename)
        
        if not src_path:
            missing_count += 1
            if missing_count <= 5: 
                print(f"⚠️ Missing: {src_filename}")
            continue
        
        # Link/Copy Image
        dst_img_path = OUTPUT_DIR / "images" / split_name / src_filename
        if not dst_img_path.exists():
            try:
                os.symlink(src_path, dst_img_path)
            except OSError:
                shutil.copy(src_path, dst_img_path)

        # Generate Label
        label_file = OUTPUT_DIR / "labels" / split_name / (Path(src_filename).stem + ".txt")
        img_w, img_h = img_info['width'], img_info['height']
        
        with open(label_file, 'w') as lf:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in cat_map: continue
                
                yolo_cls = cat_map[cat_id]
                class_counts[class_names[yolo_cls]] += 1
                
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                lf.write(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    if missing_count > 0:
        print(f"⚠️ Total missing images in {split_name}: {missing_count}")
        
    return class_names, class_counts

def main():
    print("🚀 Starting COCO (PNG) -> YOLO Conversion...")
    
    classes_train, counts_train = convert_coco_to_yolo(TRAIN_JSON, "train")
    classes_val, counts_val = convert_coco_to_yolo(VAL_JSON, "val")
    
    if not classes_train: return

    # Generate data.yaml
    yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val
names:
"""
    for i, name in enumerate(classes_train):
        yaml_content += f"  {i}: {name}\n"
        
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ data.yaml created at {OUTPUT_DIR / 'data.yaml'}")

    # Audit Report
    print("\n📊 --- FINAL DATASET AUDIT ---")
    total = sum(counts_train.values()) + sum(counts_val.values())
    print(f"{'Class':<20} | {'Count':<10} | {'%'}")
    print("-" * 40)
    for cls in classes_train:
        c = counts_train[cls] + counts_val.get(cls, 0)
        pct = (c / total * 100) if total else 0
        print(f"{cls:<20} | {c:<10} | {pct:.1f}%")

    # Debug Images
    print(f"\n📸 Generating Debug Images in {DEBUG_DIR}...")
    train_imgs = list((OUTPUT_DIR / "images/train").glob("*.png"))
    if train_imgs:
        samples = random.sample(train_imgs, min(3, len(train_imgs)))
        for img_path in samples:
            img = cv2.imread(str(img_path))
            lbl_path = OUTPUT_DIR / "labels/train" / (img_path.stem + ".txt")
            
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        
                        dh, dw, _ = img.shape
                        l = int((x - w/2) * dw)
                        r = int((x + w/2) * dw)
                        t = int((y - h/2) * dh)
                        b = int((y + h/2) * dh)
                        
                        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
                        cv2.putText(img, classes_train[cls_id], (l, t-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(DEBUG_DIR / f"debug_{img_path.name}"), img)
            print(f"   Saved: debug_{img_path.name}")

if __name__ == "__main__":
    main()
