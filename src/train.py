import os
import sys
from ultralytics import YOLO
import torch

# --- CONFIGURATION ---
# Absolute paths ensure stability across different execution contexts (CLI vs Docker)
PROJECT_ROOT = "/app/ViAna"
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, "src/utils/itva_phase1.yaml")
EXPERIMENT_PROJECT = os.path.join(PROJECT_ROOT, "data/outputs/training")
EXPERIMENT_NAME = "itva_phase1_1280p"
MODEL_WEIGHTS = "yolo11l.pt"  # YOLO11-Large

def run_training():
    """
    Executes the High-Fidelity Training Protocol (Phase 1.3).
    - Resolution: 1280p (Critical for small object recall)
    - Hardware: Dual RTX 3060 (DDP Strategy)
    - Anti-Overfitting: Aggressive MixUp/Mosaic to handle 20x class duplicates.
    """
    
    # 1. Hardware Sanity Check
    if not torch.cuda.is_available():
        print("❌ CRITICAL: CUDA not detected. Training will fail or be impossibly slow.")
        sys.exit(1)
        
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Starting High-Fidelity Training on {gpu_count} GPU(s).")
    print(f"   Model: {MODEL_WEIGHTS}")
    print(f"   Config: {DATA_YAML_PATH}")
    print(f"   Target Resolution: 1280px")

    # 2. Load Model
    # YOLO11l provides the capacity needed for 14+ distinct vehicle classes
    model = YOLO(MODEL_WEIGHTS)

    # 3. Execute Training
    # We use a specific dictionary of arguments to strictly control the behavior
    results = model.train(
        # --- Data & Model ---
        data=DATA_YAML_PATH,
        project=EXPERIMENT_PROJECT,
        name=EXPERIMENT_NAME,
        exist_ok=True,         # Overwrite existing run of same name (useful for iterative debugging)
        
        # --- Compute Strategy ---
        device=[0, 1],         # DDP: Use both RTX 3060s
        batch=8,               # FIXED: Explicit batch size (4 per GPU) for 1280p resolution
        workers=8,             # High worker count to prevent DataLoader bottlenecks at 1280p
        
        # --- High-Resolution Config ---
        #imgsz=1280,            # Mandatory for resolving small 'Auto Rickshaws' in wide static views
        #epochs=100,            # Sufficient convergence horizon for Large models

        # --- OPTIMIZATION ---
        imgsz=1088,            # REDUCED: 1280 -> 1088 (28% faster, safe form OOM)
        epochs=50,             # REDUCED: 100 -> 50 (Sufficient for baseline)

        # --- Anti-Overfitting Hyperparameters (The "Anti-Memory" Strategy) ---
        # These are crucial because we artificially duplicated 'Mini Bus' 20x.
        # We must force the model to see these duplicates in radically different contexts.
        mosaic=1.0,            # 100% chance to combine 4 images (Breaks context memorization)
        mixup=0.15,            # 15% chance to transparency-blend images (Forces shape learning over pixel memorization)
        scale=0.5,             # +/- 50% scale variation (Simulates depth/distance)
        degrees=0.0,           # No rotation (CCTV cameras are static/horizon-locked)
        fliplr=0.5,            # 50% horizontal flip (Standard data doubling)
        
        # --- Precision Tuning ---
        box=7.5,               # Higher Box Loss Gain: Prioritizes accurate bounding box localization
        val=True,              # Validate during training
        save=True              # Save checkpoints
    )

    print(f"\n✅ Training Complete.")
    print(f"   Best weights saved to: {EXPERIMENT_PROJECT}/{EXPERIMENT_NAME}/weights/best.pt")

if __name__ == "__main__":
    # The __main__ block is MANDATORY for Distributed Data Parallel (DDP) 
    # to prevent recursive process spawning on Windows/Linux.
    run_training()