import sys
import os

# Import the new auditor module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.dataset_auditor import DatasetAuditor

# ================= CONFIGURATION =================
# ⚠️ UPDATE THIS PATH to point to your actual data.yaml
DATA_YAML_PATH = "/app/ViAna/data/processed/yolo_format/data.yaml"  
OUTPUT_DIR = "/app/ViAna/data/outputs/audit_samples/"
# =================================================

def main():
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: data.yaml not found at {DATA_YAML_PATH}")
        print("   Please edit 'audit_dataset.py' and set the correct DATA_YAML_PATH.")
        return

    auditor = DatasetAuditor(DATA_YAML_PATH)
    
    # 1. Scan Distribution
    try:
        counts, label_files = auditor.scan_labels(split='train')
    except Exception as e:
        print(e)
        return

    # 2. Print Report
    print("\n" + "="*40)
    print("📊 DATASET DISTRIBUTION REPORT")
    print("="*40)
    print(f"{'ID':<5} | {'Class Name':<20} | {'Count':<10} | {'Share %':<10}")
    print("-" * 55)
    
    total_objects = sum(counts.values())
    
    # Sort by count descending
    for cls_id, count in counts.most_common():
        name = auditor.classes[cls_id]
        share = (count / total_objects) * 100
        
        # Highlight specific classes as requested
        marker = ""
        if name in ['Auto-rickshaw', 'LCV', 'Bus', 'Three-wheeler', 'Mini-bus']:
            marker = "⬅️ CHECK"
            
        print(f"{cls_id:<5} | {name:<20} | {count:<10} | {share:.1f}% {marker}")
        
    print("-" * 55)
    print(f"TOTAL OBJECTS: {total_objects}")
    print("="*40)

    # 3. Visual Verification
    auditor.generate_visual_samples(label_files, OUTPUT_DIR, num_samples=5)
    print("\n✅ Audit Complete.")

if __name__ == "__main__":
    main()