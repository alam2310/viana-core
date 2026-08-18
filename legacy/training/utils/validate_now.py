from ultralytics import YOLO
import sys

# Load the best model trained so far
model = YOLO('/app/ViAna/data/outputs/training/itva_phase1_1280p/weights/best.pt')

# Run validation on the dataset
# split='train' is used here because Phase 1 uses train set for validation
results = model.val(data='/app/ViAna/src/utils/itva_phase1.yaml', split='train', imgsz=1088, batch=4)
