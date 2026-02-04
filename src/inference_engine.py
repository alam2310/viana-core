import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --- CONFIGURATION ---
PEDESTRIAN_CLASS_ID = 11
SUPPRESSION_IOA_THRESHOLD = 0.3

# [NEW] Angled Horizon Line Configuration
# Coordinates are relative (0.0 to 1.0)
# Example: Left point is higher (0.35), Right point is lower (0.45) for a slanted road
HORIZON_POINT_LEFT = (0.0, 0.35)   # (x=0%, y=35%)
HORIZON_POINT_RIGHT = (1.0, 0.35)  # (x=100%, y=35%) - Keep same as Left for a flat line

# Hardware Allocation
DEVICE_A = 'cuda:0' # Vehicles
DEVICE_B = 'cuda:1' # Pedestrians

# Classes for Logic (Model A IDs)
ALL_VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Mapping ID to Name for Visualization
CLASS_NAMES = {
    0: 'Car', 1: 'Jeep', 2: 'Van', 3: 'MiniBus', 4: 'MTW', 
    5: 'Auto', 6: 'Bus', 7: 'Truck', 8: 'LCV', 9: 'Cycle', 
    10: 'Other', 11: 'Pedestrian'
}

def is_inside_vehicle(person_box, vehicle_boxes, threshold=0.3):
    """Calculates Intersection over Area (IoA) to suppress drivers/riders."""
    px1, py1, px2, py2 = person_box
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0: return False

    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vbox
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        
        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            if (intersection / person_area) > threshold:
                return True
    return False

def run_tracking_pipeline(video_path, model_a_path, model_b_path, output_path):
    print(f"🚀 Starting Engine: Dual-GPU + ByteTrack + Angled Horizon")
    
    # 1. Load Models
    print(f"🧠 Loading Model A (Vehicles) -> {DEVICE_A}")
    model_a = YOLO(model_a_path)
    model_a.to(DEVICE_A)
    
    print(f"🧠 Loading Model B (Pedestrians) -> {DEVICE_B}")
    model_b = YOLO(model_b_path)
    model_b.to(DEVICE_B)

    # 2. Setup Video & Tracker
    tracker = sv.ByteTrack(frame_rate=30) 
    
    # Annotators (Updated for Supervision v0.16+)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate Absolute Pixel Coordinates for Horizon Line
    x1_line = int(HORIZON_POINT_LEFT[0] * w)
    y1_line = int(HORIZON_POINT_LEFT[1] * h)
    x2_line = int(HORIZON_POINT_RIGHT[0] * w)
    y2_line = int(HORIZON_POINT_RIGHT[1] * h)
    
    # Calculate Slope (m) for y = mx + c
    # Added 1e-6 to avoid division by zero
    slope = (y2_line - y1_line) / (x2_line - x1_line + 1e-6)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        
        # --- A. INFERENCE (Dual GPU) ---
        res_a = model_a.predict(frame, imgsz=1088, conf=0.25, device=DEVICE_A, verbose=False)[0]
        res_b = model_b.predict(frame, imgsz=1088, conf=0.25, classes=[0], device=DEVICE_B, verbose=False)[0]

        # --- B. PRE-PROCESSING & MERGE ---
        final_boxes = []
        final_conf = []
        final_cls = []
        
        vehicle_boxes = [] # For IoA check

        # 1. Process Vehicles (Model A)
        if res_a.boxes:
            for box in res_a.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                final_boxes.append(xyxy)
                final_conf.append(conf)
                final_cls.append(cls)
                
                if cls in ALL_VEHICLE_CLASSES:
                    vehicle_boxes.append(xyxy)

        # 2. Process Pedestrians (Model B) with Universal Suppression
        if res_b.boxes:
            for box in res_b.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Check Logic: Is this person inside a vehicle?
                if not is_inside_vehicle(xyxy, vehicle_boxes, SUPPRESSION_IOA_THRESHOLD):
                    final_boxes.append(xyxy)
                    final_conf.append(conf)
                    final_cls.append(PEDESTRIAN_CLASS_ID) # Remap ID

        # Draw Horizon Line even if no detections
        cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 2)

        if len(final_boxes) == 0:
            out.write(frame)
            continue

        # Convert to Supervision Detections
        detections = sv.Detections(
            xyxy=np.array(final_boxes),
            confidence=np.array(final_conf),
            class_id=np.array(final_cls)
        )

        # --- D. ANGLED HORIZON FILTER ---
        # 1. Calculate centers of objects
        centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
        
        # 2. Calculate the "Cutoff Y" for every object's specific X-position
        # Equation: y = y1 + slope * (x - x1)
        y_cutoff_thresholds = y1_line + slope * (centers_x - x1_line)
        
        # 3. Filter: Keep objects strictly "Below" the line (Y value is higher than cutoff)
        mask = centers_y > y_cutoff_thresholds
        detections = detections[mask]

        # --- E. TRACKING ---
        detections = tracker.update_with_detections(detections)

        # --- VISUALIZATION ---
        labels = [
            f"#{tracker_id} {CLASS_NAMES.get(class_id, 'Unk')}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Redraw Line on top
        cv2.line(annotated_frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 2)
        cv2.putText(annotated_frame, "HORIZON", (x1_line + 10, y1_line - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(annotated_frame)

        if frame_count % 50 == 0:
            print(f"   ⏳ Tracked {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Tracking Complete. Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video")
    parser.add_argument("--model_a", type=str, default="/app/ViAna/models/v1/itva_medium_1088p.pt", help="Vehicle Model")
    parser.add_argument("--model_b", type=str, default="yolo11l.pt", help="Person Model (Large)")
    parser.add_argument("--out", type=str, default="output_tracking.mp4", help="Output path")
    
    args = parser.parse_args()
    run_tracking_pipeline(args.video, args.model_a, args.model_b, args.out)