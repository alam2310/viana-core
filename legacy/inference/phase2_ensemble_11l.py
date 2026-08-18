import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
PEDESTRIAN_CLASS_ID = 11
SUPPRESSION_IOA_THRESHOLD = 0.3  # If 30% of person overlaps with vehicle, suppress

# Hardware Allocation
DEVICE_A = 'cuda:0' # Vehicles (Custom Medium)
DEVICE_B = 'cuda:1' # Pedestrians (Standard Large)

# Target Classes for Suppression (Model A IDs)
ALL_VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# --- COLOR PALETTE (BGR Format) ---
CLASS_COLORS = {
    0: (0, 165, 255),    # Car: Orange
    1: (0, 128, 0),      # Jeep: Dark Green
    2: (255, 191, 0),    # Van: Deep Sky Blue
    3: (255, 0, 255),    # Mini Bus: Magenta
    4: (255, 255, 0),    # MTW: Cyan
    5: (203, 192, 255),  # Auto: Pink
    6: (0, 255, 255),    # City Bus: Yellow
    7: (19, 69, 139),    # Truck: Brown
    8: (144, 238, 144),  # LCV: Light Green
    9: (127, 255, 212),  # Cycle: Aquamarine
    10: (128, 0, 128),   # Others: Purple
    11: (0, 255, 0)      # Pedestrian (Confirmed): Bright Green
}

COLOR_SUPPRESSED = (128, 128, 128)  # Gray (Suppressed)
DEFAULT_COLOR = (255, 255, 255)     # Fallback: White

def is_inside_vehicle(person_box, vehicle_boxes, threshold=0.3):
    """
    Calculates Intersection over Area (IoA).
    Returns True if the person is significantly overlapping with ANY vehicle.
    """
    px1, py1, px2, py2 = person_box
    person_area = (px2 - px1) * (py2 - py1)
    
    if person_area == 0: 
        return False

    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vbox
        
        # Calculate Intersection
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        
        # Check if there is an overlap
        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            ioa = intersection_area / person_area
            
            if ioa > threshold:
                return True
                
    return False

def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=10):
    """Utility to draw a dashed rectangle for suppressed items."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top and Bottom
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
        
    # Left and Right
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)

def run_ensemble(video_path, model_a_path, model_b_path, output_path):
    print(f"🚀 Starting Dual-GPU Ensemble Inference (Medium + Large)...")
    print(f"   • Model A (Vehicles)    -> {DEVICE_A}")
    print(f"   • Model B (Pedestrians) -> {DEVICE_B}")
    
    # 1. Load Models explicitly to devices
    print(f"🧠 Loading Model A: {model_a_path}")
    model_a = YOLO(model_a_path)
    model_a.to(DEVICE_A)
    
    print(f"🧠 Loading Model B: {model_b_path}")
    model_b = YOLO(model_b_path) # Auto-downloads yolo11l.pt if missing
    model_b.to(DEVICE_B)

    # 2. Setup Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    print(f"🎬 Processing {total_frames} frames at {w}x{h}...")

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        detections = []    # Valid objects to draw solid
        suppressed = []    # Drivers/Riders to draw dotted
        vehicle_boxes = [] # Store ALL vehicle coordinates for the logic check

        # --- PASS 1: VEHICLES (Model A on GPU 0) ---
        results_a = model_a.predict(frame, imgsz=1088, conf=0.25, device=DEVICE_A, verbose=False)[0]
        
        if results_a.boxes:
            for box in results_a.boxes:
                # Move tensor to CPU for merging
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append([x1, y1, x2, y2, conf, cls, 'A'])
                
                if cls in ALL_VEHICLE_CLASSES:
                    vehicle_boxes.append([x1, y1, x2, y2])

        # --- PASS 2: PEDESTRIANS (Model B on GPU 1) ---
        # High-Res 1088p with LARGE model
        results_b = model_b.predict(frame, imgsz=1088, conf=0.25, classes=[0], device=DEVICE_B, verbose=False)[0]

        if results_b.boxes:
            for box in results_b.boxes:
                # Move tensor to CPU for merging
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                # --- UNIVERSAL SUPPRESSION LOGIC (CPU) ---
                person_box = [x1, y1, x2, y2]
                
                if is_inside_vehicle(person_box, vehicle_boxes, threshold=SUPPRESSION_IOA_THRESHOLD):
                    suppressed.append([x1, y1, x2, y2, conf])
                else:
                    cls = PEDESTRIAN_CLASS_ID 
                    detections.append([x1, y1, x2, y2, conf, cls, 'B'])

        # --- VISUALIZATION ---
        for x1, y1, x2, y2, conf, cls, source in detections:
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            
            if source == 'B':
                label = f"Pedestrian {conf:.2f}"
            else:
                if cls in model_a.names:
                    label = f"{model_a.names[cls]} {conf:.2f}"
                else:
                    label = f"Class {cls}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for x1, y1, x2, y2, conf in suppressed:
            draw_dashed_rect(frame, (x1, y1), (x2, y2), COLOR_SUPPRESSED, thickness=1)

        out.write(frame)

        if frame_count % 50 == 0:
            print(f"   ⏳ Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\n✅ Dual-GPU (Medium + Large) Inference Complete. Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video")
    parser.add_argument("--model_a", type=str, default="/app/ViAna/models/v1/itva_medium_1088p.pt", help="Custom Vehicle Model")
    
    # [UPGRADE] Default changed to yolo11l.pt (Large)
    parser.add_argument("--model_b", type=str, default="yolo11l.pt", help="Large Person Model")
    
    parser.add_argument("--out", type=str, default="debug_dual_large.mp4", help="Output path")
    
    args = parser.parse_args()
    
    run_ensemble(args.video, args.model_a, args.model_b, args.out)