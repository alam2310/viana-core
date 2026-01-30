import sys
import os
import pandas as pd
from datetime import datetime

# --- 1. Import your new modular logic ---
# This tells Python to look inside the 'src' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.classifier import VehicleClassifier

# --- Placeholder for your actual Model Inference ---
# In reality, you will replace this with your actual YOLO/DeepStream call
# For now, we simulate detections to test the PIPELINE
def get_mock_model_detections(frame_number):
    """
    Simulates what your UVH-26 model returns.
    Replace this function with your actual model inference code later.
    """
    # Simulating a Tempo-Traveller in frame 10
    if frame_number == 10:
        return [("Tempo-traveller", 0.95, [100, 100, 200, 200])]
    # Simulating a Bus in frame 20
    elif frame_number == 20:
        return [("Bus", 0.88, [50, 50, 150, 150])]
    # Simulating a MUV in frame 30
    elif frame_number == 30:
        return [("MUV", 0.75, [300, 300, 400, 400])]
    return []

def process_video_audit(video_path, output_csv="audit_report.csv"):
    print(f"🚀 Starting Video Audit: {video_path}")
    
    # 2. Initialize the Classifier
    try:
        classifier = VehicleClassifier("configs/vehicle_taxonomy.json")
        print("✅ Classifier loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return

    audit_logs = []
    frame_count = 0
    
    # --- 3. Main Processing Loop ---
    # (Simulated loop for 50 frames)
    for frame_id in range(50):
        # A. GET RAW DETECTIONS (From your Model)
        raw_detections = get_mock_model_detections(frame_id)
        
        for label, conf, bbox in raw_detections:
            # B. APPLY CLASSIFICATION LOGIC
            mapped_info = classifier.get_classification(label)
            
            # C. LOG THE RESULT
            audit_entry = {
                "Frame": frame_id,
                "Raw_Label": label,  # What the model saw
                "Confidence": conf,
                "Category": mapped_info['category'],      # Level 1
                "Class": mapped_info['class_type'],       # Level 2
                "Sub_Class": mapped_info['sub_class'],    # Level 3 (Corrected)
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            audit_logs.append(audit_entry)
            
            print(f"   Frame {frame_id}: Detected '{label}' -> Mapped to '{mapped_info['sub_class']}'")

    # --- 4. Save the Audit Report ---
    if audit_logs:
        df = pd.DataFrame(audit_logs)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Audit Complete. Report saved to: {output_csv}")
        print("-" * 30)
        print(df[["Raw_Label", "Sub_Class", "Class"]].to_string()) # Preview
    else:
        print("\n⚠️ No detections found to audit.")

if __name__ == "__main__":
    # Create a dummy video file path for testing
    dummy_video = "data/raw_videos/sample_traffic.mp4"
    process_video_audit(dummy_video)