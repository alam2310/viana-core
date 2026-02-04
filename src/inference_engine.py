import warnings
# [FIX] Suppress known harmless numpy warnings about subnormal floats
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --- CONFIGURATION ---
PEDESTRIAN_CLASS_ID = 11
SUPPRESSION_IOA_THRESHOLD = 0.3

# Horizon Configuration
HORIZON_POINT_LEFT = (0.0, 0.65)
HORIZON_POINT_RIGHT = (1.0, 0.55)

# --- LOGIC CONSTANTS (Action 2.2) ---
# 1. Geometric Logic (Trucks)
MCV_AREA_THRESHOLD = 35000       # Pixels (Adjust based on 1088p resolution)
TRAILER_ASPECT_RATIO = 2.5       # Width / Height

# 2. Color Logic (Taxis) - HSV Range for Yellow
TAXI_YELLOW_MIN = np.array([20, 100, 100])
TAXI_YELLOW_MAX = np.array([30, 255, 255])
TAXI_PIXEL_RATIO = 0.10          # 10% of car must be yellow

# Hardware Allocation
DEVICE_A = 'cuda:0' # Vehicles
DEVICE_B = 'cuda:1' # Pedestrians

# Classes for Suppression Logic
ALL_VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]

# Updated Class Names with Logic IDs
CLASS_NAMES = {
    0: 'Car', 1: 'Jeep', 2: 'Van', 3: 'MiniBus', 4: 'MTW', 
    5: 'Auto', 6: 'Bus', 7: 'Heavy Truck', 8: 'LCV', 9: 'Cycle', 
    10: 'Other', 11: 'Pedestrian',
    # New Logic Classes
    12: 'MCV', 13: 'Trailer', 14: 'Taxi'
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

def refine_truck(box):
    """
    Differentiates Truck vs MCV vs Trailer based on geometry.
    Returns: Class ID (7, 12, or 13)
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    
    # Safety check to avoid division by zero
    if h == 0: return 7 
    
    ratio = w / h

    # Rule 1: Long and flat? -> Trailer
    if ratio > TRAILER_ASPECT_RATIO:
        return 13 # Trailer ID
    
    # Rule 2: Small area? -> MCV
    # Note: This threshold depends heavily on camera distance/resolution (1088p)
    if area < MCV_AREA_THRESHOLD:
        return 12 # MCV ID

    # Default -> Heavy Truck
    return 7

def is_taxi(frame, box):
    """
    Checks if a car is yellow enough to be a Taxi.
    Returns: True/False
    """
    x1, y1, x2, y2 = box.astype(int)
    
    # Boundary Checks (Ensure crop is within image)
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)
    
    if x1 >= x2 or y1 >= y2: return False

    # Crop the car
    car_roi = frame[y1:y2, x1:x2]
    
    # Convert to HSV
    hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
    
    # Mask Yellow
    mask = cv2.inRange(hsv_roi, TAXI_YELLOW_MIN, TAXI_YELLOW_MAX)
    
    # Count non-zero pixels
    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = car_roi.shape[0] * car_roi.shape[1]
    
    if total_pixels == 0: return False
    
    if (yellow_pixels / total_pixels) > TAXI_PIXEL_RATIO:
        return True
        
    return False

def run_tracking_pipeline(video_path, model_a_path, model_b_path, output_path):
    print(f"🚀 Starting Engine: Logic Layer (MCV/Trailer/Taxi) Active")
    
    # 1. Load Models
    print(f"🧠 Loading Model A (Vehicles) -> {DEVICE_A}")
    model_a = YOLO(model_a_path)
    model_a.to(DEVICE_A)
    
    print(f"🧠 Loading Model B (Pedestrians) -> {DEVICE_B}")
    model_b = YOLO(model_b_path)
    model_b.to(DEVICE_B)

    # 2. Setup Video & Tracker
    tracker = sv.ByteTrack(frame_rate=30) 
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Horizon Calc
    x1_line = int(HORIZON_POINT_LEFT[0] * w)
    y1_line = int(HORIZON_POINT_LEFT[1] * h)
    x2_line = int(HORIZON_POINT_RIGHT[0] * w)
    y2_line = int(HORIZON_POINT_RIGHT[1] * h)
    slope = (y2_line - y1_line) / (x2_line - x1_line + 1e-6)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        
        # --- A. INFERENCE ---
        res_a = model_a.predict(frame, imgsz=1088, conf=0.25, device=DEVICE_A, verbose=False)[0]
        res_b = model_b.predict(frame, imgsz=1088, conf=0.25, classes=[0], device=DEVICE_B, verbose=False)[0]

        # --- B. LOGIC & MERGE ---
        final_boxes = []
        final_conf = []
        final_cls = []
        
        vehicle_boxes = [] 

        # 1. Process Vehicles (Model A) + APPLY LOGIC
        if res_a.boxes:
            for box in res_a.boxes:
                xyxy = box.xyxy[0].cpu().numpy() # Keep float for precise IoU, but cast for logic
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # --- LOGIC LAYER START ---
                
                # Logic 1: Refine Truck -> MCV / Trailer
                if cls == 7: # Truck
                    cls = refine_truck(xyxy)
                
                # Logic 2: Refine Car -> Taxi
                elif cls == 0: # Car
                    if is_taxi(frame, xyxy):
                        cls = 14 # Taxi ID
                
                # --- LOGIC LAYER END ---

                final_boxes.append(xyxy)
                final_conf.append(conf)
                final_cls.append(cls)
                
                if cls in ALL_VEHICLE_CLASSES:
                    vehicle_boxes.append(xyxy)

        # 2. Process Pedestrians (Model B)
        if res_b.boxes:
            for box in res_b.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                if not is_inside_vehicle(xyxy, vehicle_boxes, SUPPRESSION_IOA_THRESHOLD):
                    final_boxes.append(xyxy)
                    final_conf.append(conf)
                    final_cls.append(PEDESTRIAN_CLASS_ID)

        # Draw Horizon
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

        # --- D. HORIZON FILTER ---
        centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
        y_cutoff_thresholds = y1_line + slope * (centers_x - x1_line)
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

        # Redraw Horizon
        cv2.line(annotated_frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 2)

        out.write(annotated_frame)

        if frame_count % 50 == 0:
            print(f"   ⏳ Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Logic Processing Complete. Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video")
    parser.add_argument("--model_a", type=str, default="/app/ViAna/models/v1/itva_medium_1088p.pt", help="Vehicle Model")
    parser.add_argument("--model_b", type=str, default="yolo11l.pt", help="Person Model")
    parser.add_argument("--out", type=str, default="output_logic.mp4", help="Output path")
    
    args = parser.parse_args()
    run_tracking_pipeline(args.video, args.model_a, args.model_b, args.out)