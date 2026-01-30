import os
import glob
import yaml
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

class DatasetAuditor:
    def __init__(self, data_yaml_path):
        self.yaml_path = data_yaml_path
        self.classes = []
        self.dataset_root = os.path.dirname(data_yaml_path)
        self._load_yaml()

    def _load_yaml(self):
        """Parses the data.yaml to get class names and paths."""
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data['names'], dict):
            self.classes = list(data['names'].values())
        else:
            self.classes = data['names']
            
        print(f"✅ Loaded {len(self.classes)} classes from {self.yaml_path}")

    def scan_labels(self, split='train'):
        """Scans the label directory and counts class instances."""
        labels_path = os.path.join(self.dataset_root, 'labels', split)
        
        if not os.path.exists(labels_path):
            print(f"⚠️ Path {labels_path} not found. Searching structure...")
            labels_path = os.path.join(self.dataset_root, split, 'labels')
            
        print(f"📂 Scanning labels in: {labels_path}")
        label_files = glob.glob(os.path.join(labels_path, "*.txt"))
        
        if not label_files:
            raise FileNotFoundError("❌ No .txt label files found!")

        class_counts = Counter()
        
        print("⏳ Counting classes...")
        for file in tqdm(label_files):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        
        return class_counts, label_files

    def generate_visual_samples(self, label_files, output_dir, num_samples=5):
        """Randomly selects images and draws bounding boxes (Robust Path Finding)."""
        os.makedirs(output_dir, exist_ok=True)
        if not label_files:
            print("❌ No labels provided to sample from.")
            return

        samples = random.sample(label_files, min(num_samples, len(label_files)))
        print(f"\n🎨 Generating {len(samples)} visual verification samples...")
        
        # --- ROBUST IMAGE FINDER ---
        # 1. Deduce Image Directory from Label Directory
        first_label = label_files[0]
        label_dir = os.path.dirname(first_label)
        
        # Replace '/labels/' with '/images/' to handle standard YOLO structure
        if '/labels/' in label_dir:
            img_dir = label_dir.replace('/labels/', '/images/')
        elif 'labels' in label_dir:
            img_dir = label_dir.replace('labels', 'images')
        else:
            print(f"❌ Could not infer image directory from {label_dir}")
            return

        print(f"   Looking for images in: {img_dir}")
        
        if not os.path.exists(img_dir):
            print(f"❌ Critical: Image directory does not exist: {img_dir}")
            return

        # 2. Map Image Files {filename_no_ext: full_filename}
        # This handles extension mismatches (.png vs .PNG vs .jpg)
        try:
            all_files = os.listdir(img_dir)
            # Filter for images only
            valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
            image_map = {
                os.path.splitext(f)[0]: f 
                for f in all_files 
                if os.path.splitext(f)[1].lower() in valid_exts
            }
            print(f"   DEBUG: Found {len(image_map)} images in directory.")
            if len(image_map) > 0:
                print(f"   DEBUG: First 3 files found: {list(image_map.values())[:3]}")
        except Exception as e:
            print(f"❌ Error reading image directory: {e}")
            return
        
        # --- DRAWING LOOP ---
        for label_file in samples:
            base_id = os.path.splitext(os.path.basename(label_file))[0]
            
            # Lookup valid image file
            if base_id not in image_map:
                print(f"⚠️ Image not found for ID: {base_id}")
                continue
                
            img_path = os.path.join(img_dir, image_map[base_id])
            
            # Load Image
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Failed to load image (corrupt?): {img_path}")
                continue

            h, w, _ = img.shape
            
            # Read Bounding Boxes
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # De-normalize
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Draw
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    label_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
                    cv2.putText(img, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save Result
            save_name = f"audit_{image_map[base_id]}"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img)
            print(f"   Saved: {save_path}")