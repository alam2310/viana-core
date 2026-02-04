import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, Counter

# --- CONFIGURATION ---
PEDESTRIAN_CLASS_ID = 11
SUPPRESSION_IOA_THRESHOLD = 0.3

# Horizon (Red Line) 0.30
HORIZON_POINT_LEFT = (0.0, 0.6)
HORIZON_POINT_RIGHT = (1.0, -0.4)

# Counting (Green Line) 0.65
COUNTING_LINE_START = (0.0, 1.15) 
COUNTING_LINE_END = (1.0, -0.15)

# Logic Constants
MCV_AREA_THRESHOLD = 35000       
TRAILER_ASPECT_RATIO = 2.5
TAXI_YELLOW_MIN = np.array([20, 100, 100])
TAXI_YELLOW_MAX = np.array([30, 255, 255])
TAXI_PIXEL_RATIO = 0.10

# Hardware
DEVICE_A = 'cuda:0' 
DEVICE_B = 'cuda:1' 

# Classes
ALL_VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
CLASS_NAMES = {
    0: 'Car', 1: 'Jeep', 2: 'Van', 3: 'MiniBus', 4: 'MTW', 
    5: 'Auto', 6: 'Bus', 7: 'Heavy Truck', 8: 'LCV', 9: 'Cycle', 
    10: 'Other', 11: 'Pedestrian',
    12: 'MCV', 13: 'Trailer', 14: 'Taxi'
}

# --- STATE MANAGEMENT ---
track_history = defaultdict(lambda: {
    'max_area': 0, 
    'class_votes': Counter(), 
    'locked_class': None
})

counts_in = defaultdict(int)
counts_out = defaultdict(int)

def is_inside_vehicle(person_box, vehicle_boxes, threshold=0.3):
    px1, py1, px2, py2 = person_box
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0: return False
    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vbox
        ix1 = max(px1, vx1); iy1 = max(py1, vy1)
        ix2 = min(px2, vx2); iy2 = min(py2, vy2)
        if ix2 > ix1 and iy2 > iy1:
            if ((ix2 - ix1) * (iy2 - iy1) / person_area) > threshold: return True
    return False

def is_taxi(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    h_img, w_img = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w_img, x2); y2 = min(h_img, y2)
    if x1 >= x2 or y1 >= y2: return False
    car_roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, TAXI_YELLOW_MIN, TAXI_YELLOW_MAX)
    if cv2.countNonZero(mask) / (car_roi.size / 3) > TAXI_PIXEL_RATIO: return True
    return False

def print_traffic_report(current_frame, total_frames):
    print("\n" + "="*65)
    print(f"📊 TRAFFIC REPORT @ Frame {current_frame}/{total_frames}")
    print(f"{'CLASS NAME':<20} | {'IN COUNT':<10} | {'OUT COUNT':<10} | {'TOTAL':<10}")
    print("-" * 65)
    
    all_classes = set(counts_in.keys()) | set(counts_out.keys())
    sorted_stats = []
    
    for cls_id in all_classes:
        c_in = counts_in[cls_id]
        c_out = counts_out[cls_id]
        total = c_in + c_out
        name = CLASS_NAMES.get(cls_id, f"ID {cls_id}")
        sorted_stats.append((name, c_in, c_out, total))
    
    sorted_stats.sort(key=lambda x: x[3], reverse=True)
    
    for name, c_in, c_out, total in sorted_stats:
        print(f"{name:<20} | {c_in:<10} | {c_out:<10} | {total:<10}")
    print("="*65 + "\n")

def draw_counts_on_line(frame, line_start, line_end, in_counts, out_counts):
    center_x = int((line_start.x + line_end.x) / 2)
    center_y = int((line_start.y + line_end.y) / 2)
    
    DISPLAY_ORDER = sorted(CLASS_NAMES.keys())
    
    out_text = "OUT: "
    has_out = False
    for cls_id in DISPLAY_ORDER:
        if out_counts[cls_id] > 0:
            out_text += f"{CLASS_NAMES[cls_id]}:{out_counts[cls_id]} "
            has_out = True
    
    in_text = "IN: "
    has_in = False
    for cls_id in DISPLAY_ORDER:
        if in_counts[cls_id] > 0:
            in_text += f"{CLASS_NAMES[cls_id]}:{in_counts[cls_id]} "
            has_in = True

    font_scale = 0.6; thickness = 2
    
    if has_out:
        (tw, th), _ = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (center_x - tw//2 - 5, center_y - 30), (center_x + tw//2 + 5, center_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, out_text, (center_x - tw//2, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 255), thickness)

    if has_in:
        (tw, th), _ = cv2.getTextSize(in_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (center_x - tw//2 - 5, center_y + 10), (center_x + tw//2 + 5, center_y + 35), (0, 0, 0), -1)
        cv2.putText(frame, in_text, (center_x - tw//2, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

def run_engine(video_path, model_a_path, model_b_path, output_path):
    print(f"🚀 Starting Engine: Large Video Mode")
    
    model_a = YOLO(model_a_path); model_a.to(DEVICE_A)
    model_b = YOLO(model_b_path); model_b.to(DEVICE_B)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # [NEW] Get Total Frames

    tracker = sv.ByteTrack(frame_rate=30)
    
    line_start = sv.Point(int(COUNTING_LINE_START[0] * w), int(COUNTING_LINE_START[1] * h))
    line_end = sv.Point(int(COUNTING_LINE_END[0] * w), int(COUNTING_LINE_END[1] * h))
    line_zone = sv.LineZone(start=line_start, end=line_end)
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    x1_h = int(HORIZON_POINT_LEFT[0] * w); y1_h = int(HORIZON_POINT_LEFT[1] * h)
    x2_h = int(HORIZON_POINT_RIGHT[0] * w); y2_h = int(HORIZON_POINT_RIGHT[1] * h)
    horizon_slope = (y2_h - y1_h) / (x2_h - x1_h + 1e-6)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1

        res_a = model_a.predict(frame, imgsz=1088, conf=0.25, device=DEVICE_A, verbose=False)[0]
        res_b = model_b.predict(frame, imgsz=1088, conf=0.25, classes=[0], device=DEVICE_B, verbose=False)[0]

        final_boxes, final_conf, final_cls, vehicle_boxes = [], [], [], []

        if res_a.boxes:
            for box in res_a.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0 and is_taxi(frame, xyxy): cls = 14
                final_boxes.append(xyxy); final_conf.append(float(box.conf[0])); final_cls.append(cls)
                if cls in ALL_VEHICLE_CLASSES: vehicle_boxes.append(xyxy)

        if res_b.boxes:
            for box in res_b.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                if not is_inside_vehicle(xyxy, vehicle_boxes, SUPPRESSION_IOA_THRESHOLD):
                    final_boxes.append(xyxy); final_conf.append(float(box.conf[0])); final_cls.append(PEDESTRIAN_CLASS_ID)

        if not final_boxes:
            cv2.line(frame, (x1_h, y1_h), (x2_h, y2_h), (0, 0, 255), 2)
            cv2.line(frame, (line_start.x, line_start.y), (line_end.x, line_end.y), (0, 255, 0), 2)
            draw_counts_on_line(frame, line_start, line_end, counts_in, counts_out)
            out.write(frame); continue

        detections = sv.Detections(xyxy=np.array(final_boxes), confidence=np.array(final_conf), class_id=np.array(final_cls))

        centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
        cutoff = y1_h + horizon_slope * (centers_x - x1_h)
        detections = detections[centers_y > cutoff]

        detections = tracker.update_with_detections(detections)
        
        updated_class_ids = []
        for xyxy, tracker_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            
            w_box, h_box = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
            current_area = w_box * h_box
            track_history[tracker_id]['max_area'] = max(track_history[tracker_id]['max_area'], current_area)
            max_area = track_history[tracker_id]['max_area']

            if class_id in [7, 12]:
                ratio = w_box / h_box if h_box > 0 else 0
                if ratio > TRAILER_ASPECT_RATIO: 
                    class_id = 13
                    track_history[tracker_id]['locked_class'] = 13
                elif max_area < MCV_AREA_THRESHOLD: class_id = 12 
                else: class_id = 7 
            
            if track_history[tracker_id]['locked_class'] is not None:
                class_id = track_history[tracker_id]['locked_class']
            elif class_id == 14:
                track_history[tracker_id]['locked_class'] = 14

            track_history[tracker_id]['class_votes'][class_id] += 1
            if track_history[tracker_id]['locked_class'] is None:
                most_common = track_history[tracker_id]['class_votes'].most_common(1)[0][0]
                class_id = most_common

            updated_class_ids.append(class_id)
        
        detections.class_id = np.array(updated_class_ids)

        # --- COUNTING (SAFETY CHECKED) ---
        if len(detections) > 0:
            bottom_centers = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            proxy_xyxy = []
            for (x, y) in bottom_centers: proxy_xyxy.append([x-1, y-1, x+1, y+1])
            
            proxy_detections = sv.Detections(
                xyxy=np.array(proxy_xyxy), 
                confidence=detections.confidence, 
                class_id=detections.class_id, 
                tracker_id=detections.tracker_id
            )

            cross_in, cross_out = line_zone.trigger(detections=proxy_detections)
            
            for is_in, is_out, cls_id in zip(cross_in, cross_out, detections.class_id):
                if is_in: counts_out[cls_id] += 1
                if is_out: counts_in[cls_id] += 1
            
            # Draw Debug Anchors only if detections exist
            for center in bottom_centers:
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)

        labels = [f"#{t_id} {CLASS_NAMES.get(c_id, 'Unk')}" for c_id, t_id in zip(detections.class_id, detections.tracker_id)]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        cv2.line(frame, (x1_h, y1_h), (x2_h, y2_h), (0, 0, 255), 2)
        cv2.line(frame, (line_start.x, line_start.y), (line_end.x, line_end.y), (0, 255, 0), 2)
        draw_counts_on_line(frame, line_start, line_end, counts_in, counts_out)

        DISPLAY_ORDER = sorted(CLASS_NAMES.keys())
        dashboard_text = "TOTALS: "
        for cls_id in DISPLAY_ORDER:
            total = counts_in[cls_id] + counts_out[cls_id]
            if total > 0:
                name = CLASS_NAMES.get(cls_id, 'Unk')
                dashboard_text += f"{name}: {total} | "
        
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, dashboard_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        
        # [NEW] Log every 100 frames with Total count
        if frame_count % 100 == 0: 
            print(f"   ⏳ Processed {frame_count}/{total_frames} frames")
            
        # [NEW] Periodic ASCII Table every 1000 frames
        if frame_count % 1000 == 0:
            print_traffic_report(frame_count, total_frames)

    cap.release(); out.release()
    print_traffic_report(frame_count, total_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_a", default="/app/ViAna/models/v1/itva_medium_1088p.pt")
    parser.add_argument("--model_b", default="yolo11l.pt")
    parser.add_argument("--out", default="final_large_run.mp4")
    args = parser.parse_args()
    run_engine(args.video, args.model_a, args.model_b, args.out)