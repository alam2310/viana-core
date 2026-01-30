import os
import json
import shutil
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path("/app/ViAna") 
INPUT_DATA_DIR = PROJECT_ROOT / "data/processed/yolo_format"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data/itva_phase1"
TAXONOMY_PATH = PROJECT_ROOT / "configs/vehicle_taxonomy.json"
OUTPUT_YAML_PATH = PROJECT_ROOT / "src/utils/itva_phase1.yaml"

TARGET_IDS = {
    'Car': 0, 'Jeep': 1, 'Van': 2, 'Mini Bus': 3,
    'MTW': 4, 'Auto': 5, 'City Bus': 6, 'Truck': 7,
    'LCV': 8, 'Cycle': 9, 'Others': 10
}

RAW_CLASS_NAMES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", 
    "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus", 
    "Tempo-traveller", "Bicycle", "Van", "Others"
]

class ITVADatasetBuilder:
    def __init__(self):
        self._load_taxonomy()
        self.stats = {k: 0 for k in TARGET_IDS.keys()}
        
    def _load_taxonomy(self):
        if not TAXONOMY_PATH.exists():
            raise FileNotFoundError(f"❌ Taxonomy config missing: {TAXONOMY_PATH}")
        with open(TAXONOMY_PATH, 'r') as f:
            self.taxonomy = json.load(f)
        print(f"✅ Loaded Taxonomy: {len(self.taxonomy)} keys.")

    def get_target_id(self, raw_class_name):
        key = raw_class_name.lower().strip()
        if key not in self.taxonomy:
            raise ValueError(f"❌ CRITICAL: Raw class '{raw_class_name}' not found in JSON config!")
        sub_class = self.taxonomy[key]['sub_class']
        if sub_class not in TARGET_IDS:
             raise ValueError(f"❌ Config Error: Sub-class '{sub_class}' is not in TARGET_IDS map.")
        return TARGET_IDS[sub_class], sub_class

    def build(self):
        print(f"🚀 Starting Dataset Build: {INPUT_DATA_DIR} -> {OUTPUT_DATA_DIR}")
        
        if OUTPUT_DATA_DIR.exists():
            print("⚠️ Output directory exists. Cleaning up...")
            shutil.rmtree(OUTPUT_DATA_DIR)
        
        (OUTPUT_DATA_DIR / "images/train").mkdir(parents=True)
        (OUTPUT_DATA_DIR / "labels/train").mkdir(parents=True)

        input_labels = list((INPUT_DATA_DIR / "labels/train").glob("*.txt"))
        if not input_labels:
            raise FileNotFoundError(f"❌ No labels found in {INPUT_DATA_DIR}/labels/train")

        manifest_lines = []
        print(f"⏳ Processing {len(input_labels)} files...")
        
        for label_file in tqdm(input_labels):
            new_lines = []
            has_rare_class = False
            rare_multiplier = 1
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                raw_id = int(parts[0])
                if raw_id >= len(RAW_CLASS_NAMES): continue
                
                raw_name = RAW_CLASS_NAMES[raw_id]
                coords = parts[1:]
                target_id, sub_class_name = self.get_target_id(raw_name)
                
                if sub_class_name in ['Mini Bus', 'Van']:
                    has_rare_class = True
                    rare_multiplier = max(rare_multiplier, 20)
                elif sub_class_name == 'LCV':
                    has_rare_class = True
                    rare_multiplier = max(rare_multiplier, 5)

                new_lines.append(f"{target_id} {' '.join(coords)}\n")
                self.stats[sub_class_name] += 1

            if not new_lines: continue
            
            file_stem = label_file.stem
            new_label_path = OUTPUT_DATA_DIR / "labels/train" / f"{file_stem}.txt"
            
            with open(new_label_path, 'w') as f:
                f.writelines(new_lines)
            
            # --- IMAGE HANDLING ---
            src_img_dir = INPUT_DATA_DIR / "images/train"
            found_img = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential = src_img_dir / f"{file_stem}{ext}"
                if potential.exists():
                    found_img = potential
                    break
            
            if found_img:
                dst_img = OUTPUT_DATA_DIR / "images/train" / found_img.name
                if not dst_img.exists():
                    os.symlink(found_img, dst_img)
                
                # [CRITICAL FIX] Use absolute string path, DO NOT resolve symlinks
                # This ensures YOLO looks in 'itva_phase1/labels' not 'data/raw/labels'
                final_path = str(dst_img.absolute())
                
                repeats = rare_multiplier if has_rare_class else 1
                for _ in range(repeats):
                    manifest_lines.append(final_path)

        with open(OUTPUT_DATA_DIR / "train.txt", 'w') as f:
            f.write('\n'.join(manifest_lines))
            
        print("\n✅ Build Complete.")
        self._generate_yaml()

    def _generate_yaml(self):
        yaml_data = {
            'path': str(OUTPUT_DATA_DIR.resolve()), 
            'train': 'train.txt',
            'val': 'train.txt',
            'names': {v: k for k, v in TARGET_IDS.items()}
        }
        with open(OUTPUT_YAML_PATH, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
        print(f"\n📝 YAML Configuration saved to: {OUTPUT_YAML_PATH}")

if __name__ == "__main__":
    builder = ITVADatasetBuilder()
    builder.build()