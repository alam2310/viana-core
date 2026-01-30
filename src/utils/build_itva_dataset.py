import os
import json
import shutil
import glob
import yaml
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
# Assumes script is running from project root inside container
PROJECT_ROOT = Path("/app/ViAna") 
INPUT_DATA_DIR = PROJECT_ROOT / "data/processed/yolo_format"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data/itva_phase1"  # Git-ignored data folder
TAXONOMY_PATH = PROJECT_ROOT / "configs/vehicle_taxonomy.json"
# Save the model config YAML in src/utils so it is tracked by Git
OUTPUT_YAML_PATH = PROJECT_ROOT / "src/utils/itva_phase1.yaml"

# The Integer ID Map (The "Model's Truth")
TARGET_IDS = {
    'Car': 0, 
    'Jeep': 1, 
    'Van': 2, 
    'Mini Bus': 3,
    'MTW': 4, 
    'Auto': 5, 
    'City Bus': 6, 
    'Truck': 7,
    'LCV': 8, 
    'Cycle': 9, 
    'Others': 10
}

# The UVH-26 Raw ID Map (needed to read the raw .txt files)
# ⚠️ UPDATE THIS LIST if your raw classes differ in order!
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
        """Looks up the raw name in JSON and returns the new Integer ID."""
        key = raw_class_name.lower().strip()
        
        # 1. Lookup in JSON
        if key not in self.taxonomy:
            raise ValueError(f"❌ CRITICAL: Raw class '{raw_class_name}' not found in JSON config! Run sync_taxonomy.py first.")
        
        # 2. Get Sub-Class String
        sub_class = self.taxonomy[key]['sub_class']
        
        # 3. Convert to Target Integer
        if sub_class not in TARGET_IDS:
             raise ValueError(f"❌ Config Error: Sub-class '{sub_class}' (from '{key}') is not in TARGET_IDS map.")
             
        return TARGET_IDS[sub_class], sub_class

    def build(self):
        print(f"🚀 Starting Dataset Build: {INPUT_DATA_DIR} -> {OUTPUT_DATA_DIR}")
        
        # Setup Output Directories
        if OUTPUT_DATA_DIR.exists():
            print("⚠️ Output directory exists. Cleaning up...")
            shutil.rmtree(OUTPUT_DATA_DIR)
        
        # Create standard YOLO structure
        (OUTPUT_DATA_DIR / "images/train").mkdir(parents=True)
        (OUTPUT_DATA_DIR / "labels/train").mkdir(parents=True)

        # Get Source Files
        input_labels = list((INPUT_DATA_DIR / "labels/train").glob("*.txt"))
        if not input_labels:
            raise FileNotFoundError(f"❌ No labels found in {INPUT_DATA_DIR}/labels/train")

        manifest_lines = []
        
        print(f"⏳ Processing {len(input_labels)} files...")
        
        for label_file in tqdm(input_labels):
            # A. Prepare New Label Content
            new_lines = []
            has_rare_class = False
            rare_multiplier = 1
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                # Parse Raw
                raw_id = int(parts[0])
                if raw_id >= len(RAW_CLASS_NAMES):
                    continue # Skip invalid raw IDs
                
                raw_name = RAW_CLASS_NAMES[raw_id]
                coords = parts[1:]
                
                # Map to Target
                target_id, sub_class_name = self.get_target_id(raw_name)
                
                # Check Oversampling Trigger
                if sub_class_name in ['Mini Bus', 'Van']:
                    has_rare_class = True
                    rare_multiplier = max(rare_multiplier, 20) # Max wins
                elif sub_class_name == 'LCV':
                    has_rare_class = True
                    rare_multiplier = max(rare_multiplier, 5)

                # Append Transformed Line
                new_lines.append(f"{target_id} {' '.join(coords)}\n")
                self.stats[sub_class_name] += 1

            # B. Write New Label File
            if not new_lines: continue # Skip empty files
            
            file_stem = label_file.stem
            new_label_path = OUTPUT_DATA_DIR / "labels/train" / f"{file_stem}.txt"
            
            with open(new_label_path, 'w') as f:
                f.writelines(new_lines)
            
            # C. Handle Image (Symlink to save space)
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
                
                # D. Add to Manifest (Oversampling Logic)
                # If rare class exists, repeat this image path N times in the list
                repeats = rare_multiplier if has_rare_class else 1
                for _ in range(repeats):
                    manifest_lines.append(str(dst_img.resolve()))
            else:
                 # Optional: print warning if needed, usually skipped to reduce noise
                 pass

        # Write Manifest
        with open(OUTPUT_DATA_DIR / "train.txt", 'w') as f:
            f.write('\n'.join(manifest_lines))
            
        print("\n✅ Build Complete.")
        print("📊 New Class Distribution:")
        for k, v in self.stats.items():
            print(f"  {k:<10}: {v}")

        self._generate_yaml()

    def _generate_yaml(self):
        """Generates the dataset.yaml file required by YOLO."""
        # Create the dictionary
        yaml_data = {
            'path': str(OUTPUT_DATA_DIR.resolve()), # Absolute path to data
            'train': 'train.txt',
            'val': 'train.txt', # Using train as val for Phase 1 check
            'names': {v: k for k, v in TARGET_IDS.items()} # Invert map: ID -> Name
        }
        
        # Save to src/utils/ for version control
        with open(OUTPUT_YAML_PATH, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False)
            
        print(f"\n📝 YAML Configuration saved to: {OUTPUT_YAML_PATH}")

if __name__ == "__main__":
    builder = ITVADatasetBuilder()
    builder.build()