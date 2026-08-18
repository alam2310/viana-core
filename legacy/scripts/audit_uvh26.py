import os
import yaml
import cv2
import glob
import random
import numpy as np
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
# Path matches your container mount point
DATASET_ROOT = "/root/Work/ViAna/data/raw/UVH-26" 
DEBUG_DIR = "/root/Work/ViAna/debug_pretrain"
INTEREST_CLASSES = ["Auto Rickshaw", "LCV", "Bus", "Three-wheeler"] 

def load_yaml_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, dict):
        return names 
    return {i: name for i, name in enumerate(names)}

def draw_yolo_box(img, class_name, x, y, w, h):
    dh, dw, _ = img.shape
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    
    color = (0, 255, 0)
    cv2.rectangle(img, (l, t), (r, b), color, 2)
    cv2.putText(img, class_name, (l, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def main():
    print(f"🔍 Starting Audit on: {DATASET_ROOT}")
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: data.yaml not found at {yaml_path}")
        print("   Did you run the download script yet?")
        return

    class_map = load_yaml_classes(yaml_path)
    print(f"✅ Found {len(class_map)} classes.")
    
    # Updated glob to be safer with different folder structures
    label_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.txt"), recursive=True)
    label_files = [f for f in label_files if not f.endswith("classes.txt")]
    
    print(f"📂 Scanning {len(label_files)} label files...")
    
    class_counts = Counter()
    
    for lf in tqdm(label_files):
        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_counts[int(parts[0])] += 1

    print("\n📊 --- DISTRIBUTION REPORT ---")
    print(f"{'Class Name':<20} | {'Count':<10} | {'% of Total':<10}")
    total_objects = sum(class_counts.values())
    
    for cls_name in INTEREST_CLASSES:
        found = False
        for cid, cname in class_map.items():
            if cname.lower() == cls_name.lower():
                count = class_counts[cid]
                pct = (count / total_objects) * 100 if total_objects > 0 else 0
                print(f"👉 {cname:<17} | {count:<10} | {pct:.2f}%")
                found = True
        if not found:
            print(f"⚠️ {cls_name:<17} | NOT FOUND in dataset labels")

    print(f"\n📸 Generating 5 debug images in {DEBUG_DIR}...")
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)
        
    sample_files = random.sample(label_files, min(5, len(label_files)))
    
    for lbl_path in sample_files:
        # Heuristic to find image: replace 'labels' with 'images' and ext with .jpg
        img_path = lbl_path.replace("labels", "images").replace(".txt", ".jpg")
        
        if not os.path.exists(img_path):
             img_path = img_path.replace(".jpg", ".png")
             
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None: continue
            
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cid = int(parts[0])
                    # Denormalize coords
                    x, y, w, h = map(float, parts[1:5])
                    cname = class_map.get(cid, str(cid))
                    img = draw_yolo_box(img, cname, x, y, w, h)
            
            base_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_{base_name}"), img)
            print(f"   Saved: debug_{base_name}")

if __name__ == "__main__":
    main()
