import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import cv2
import time
import datetime
import argparse
import subprocess
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import torch
from ultralytics import YOLO
import supervision as sv


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
@dataclass
class TrafficConfig:
    # Model Settings
    device_a: str = 'cuda:0'
    device_b: str = 'cuda:1'
    pedestrian_id: int = 11
    suppression_ioa: float = 0.3
    nms_threshold: float = 0.5
    
    # Logic Constants
    lock_frames: int = 15
    perspective_scale: float = 6.0
    trailer_ratio: float = 2.5
    lcv_max_area: int = 25000
    mcv_max_area: int = 65000
    commercial_trucks: List[int] = field(default_factory=lambda: [7, 8, 12])
    
    # Geometry (Normalized)
    horizon_left: Tuple[float, float] = (0.0, 0.2)
    horizon_right: Tuple[float, float] = (1.0, 0.2)
    counting_start: Tuple[float, float] = (0.0, 0.5)
    counting_end: Tuple[float, float] = (1.0, 0.5)
    
    # Taxonomy
    target_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'Car', 1: 'Jeep', 2: 'Van', 3: 'MiniBus', 4: 'MTW', 
        5: 'Auto', 6: 'Bus', 7: 'Heavy Truck', 8: 'LCV', 9: 'Cycle', 
        10: 'Other', 11: 'Pedestrian', 12: 'MCV', 13: 'Trailer', 14: 'Taxi'
    })


# ==============================================================================
# 2. BUSINESS LOGIC & STATE MANAGEMENT
# ==============================================================================
class ClassificationEngine:
    """Handles perspective math, trailer detection, and the N-frame voting lock."""
    def __init__(self, config: TrafficConfig, frame_height: int):
        self.config = config
        self.h = frame_height
        self.horizon_y = config.horizon_left[1] * frame_height
        
        # State: tracker_id -> {'votes': [], 'locked_class': int, 'max_ratio': float, 'max_norm_area': float}
        self.track_history = defaultdict(lambda: {
            'votes': [], 
            'locked_class': None,
            'max_ratio': 0.0,
            'max_norm_area': 0.0
        })

    def normalize_area(self, raw_area: float, box_cy: float) -> float:
        rel_y = (box_cy - self.horizon_y) / (self.h - self.horizon_y)
        rel_y = np.clip(rel_y, 0.001, 1.0)
        scale = 1 + (self.config.perspective_scale - 1) * (1 - rel_y)
        return raw_area * scale

    def process_vehicle(self, tracker_id: int, raw_class: int, xyxy: np.ndarray) -> Tuple[int, int]:
        """Returns (final_class_id, normalized_area)"""
        w_box, h_box = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        box_cy = (xyxy[1] + xyxy[3]) / 2
        
        raw_area = w_box * h_box
        norm_area = self.normalize_area(raw_area, box_cy)
        ratio = w_box / h_box if h_box > 0 else 0
        state = self.track_history[tracker_id]

        # Reset voting if object gets significantly closer
        if norm_area > state['max_norm_area'] * 1.5:
            state['votes'].clear()
            state['locked_class'] = None
            state['max_norm_area'] = norm_area

        # 1. Voting Lock
        if state['locked_class'] is not None:
            base_class = state['locked_class']
        else:
            state['votes'].append(raw_class)
            base_class = Counter(state['votes']).most_common(1)[0][0]
            if len(state['votes']) >= self.config.lock_frames:
                state['locked_class'] = base_class

        final_class = base_class

        # 2. Commercial Tiering & Geometry
        if base_class in self.config.commercial_trucks:
            state['max_ratio'] = max(state['max_ratio'], ratio)
            
            if state['max_ratio'] > self.config.trailer_ratio:
                final_class = 13
            else:
                if norm_area < self.config.lcv_max_area:
                    final_class = 8
                elif norm_area < self.config.mcv_max_area:
                    final_class = 12
                else:
                    final_class = 7

        return final_class, int(norm_area)


# ==============================================================================
# 3. I/O HANDLER
# ==============================================================================
class VideoStreamer:
    """Manages OpenCV capture and FFmpeg NVENC HEVC piping."""
    def __init__(self, input_path: str, output_path: str):
        self.cap = cv2.VideoCapture(input_path)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.w}x{self.h}', '-pix_fmt', 'bgr24', '-r', f'{self.fps}', '-i', '-',
            '-c:v', 'hevc_nvenc', '-pix_fmt', 'yuv420p', '-preset', 'p7',
            '-rc', 'vbr', '-cq', '38', '-b:v', '0', '-bf', '3', '-spatial-aq', '1', output_path
        ]
        self.pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.pipe.stdin.write(frame.tobytes())

    def close(self):
        self.cap.release()
        self.pipe.stdin.close()
        self.pipe.wait()


# ==============================================================================
# 4. PRESENTATION / VISUALIZER
# ==============================================================================
class Visualizer:
    """Handles all frame annotations and console reporting."""
    def __init__(self, config: TrafficConfig, width: int, height: int):
        self.config = config
        self.w, self.h = width, height
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        # Calculate line coordinates in pixels
        self.x1_h, self.y1_h = int(config.horizon_left[0] * width), int(config.horizon_left[1] * height)
        self.x2_h, self.y2_h = int(config.horizon_right[0] * width), int(config.horizon_right[1] * height)
        
        start_pt = sv.Point(int(config.counting_start[0] * width), int(config.counting_start[1] * height))
        end_pt = sv.Point(int(config.counting_end[0] * width), int(config.counting_end[1] * height))
        self.line_zone = sv.LineZone(start=start_pt, end=end_pt)
        self.p_start, self.p_end = start_pt, end_pt

    def draw_annotations(self, frame: np.ndarray, detections: sv.Detections, debug_info: dict) -> np.ndarray:
        labels = []
        for t_id, c_id in zip(detections.tracker_id, detections.class_id):
            info = debug_info.get(t_id, {})
            raw_id = info.get('raw', c_id)
            area = info.get('area', 0)
            
            tracked_name = self.config.class_names.get(c_id, 'Unk')
            raw_name = self.config.class_names.get(raw_id, 'Unk')
            
            text = f"#{t_id} {tracked_name} (Raw:{raw_name})"
            if c_id in self.config.commercial_trucks or raw_id in self.config.commercial_trucks:
                text += f" [{area//1000}k]"
            labels.append(text)

        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # Draw zones
        cv2.line(frame, (self.x1_h, self.y1_h), (self.x2_h, self.y2_h), (0, 0, 255), 2)
        cv2.line(frame, (self.p_start.x, self.p_start.y), (self.p_end.x, self.p_end.y), (0, 255, 0), 2)
        return frame

    def draw_dashboard(self, frame: np.ndarray, counts_in: dict, counts_out: dict):
        dashboard_text = "TOTALS: "
        for cls_id in sorted(self.config.class_names.keys()):
            total = counts_in.get(cls_id, 0) + counts_out.get(cls_id, 0)
            if total > 0:
                dashboard_text += f"{self.config.class_names[cls_id]}: {total} | "
        
        cv2.rectangle(frame, (0, 0), (self.w, 50), (0, 0, 0), -1)
        cv2.putText(frame, dashboard_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def print_report(self, current_frame: int, total_frames: int, counts_in: dict, counts_out: dict, start_time: float, fps: float):
        elapsed_sec = time.time() - start_time
        video_sec = current_frame / fps if fps > 0 else 0
        
        print("\n" + "="*65)
        print(f"📊 TRAFFIC REPORT @ Frame {current_frame}/{total_frames}")
        print(f"⏱️  Processing Time: {datetime.timedelta(seconds=int(elapsed_sec))} | 🎞️ Video Processed: {datetime.timedelta(seconds=int(video_sec))}")
        print(f"{'CLASS NAME':<20} | {'IN COUNT':<10} | {'OUT COUNT':<10} | {'TOTAL':<10}")
        print("-" * 65)
        
        all_classes = set(counts_in.keys()) | set(counts_out.keys())
        stats = []
        for cls_id in all_classes:
            c_in = counts_in.get(cls_id, 0)
            c_out = counts_out.get(cls_id, 0)
            if c_in + c_out > 0:
                stats.append((self.config.class_names.get(cls_id, f"ID {cls_id}"), c_in, c_out, c_in + c_out))
        
        for name, c_in, c_out, total in sorted(stats, key=lambda x: x[3], reverse=True):
            print(f"{name:<20} | {c_in:<10} | {c_out:<10} | {total:<10}")
        print("="*65 + "\n")


# ==============================================================================
# 5. ORCHESTRATOR PIPELINE
# ==============================================================================
class TrafficPipeline:
    """The main application controller linking ML, I/O, Logic, and UI."""
    def __init__(self, video_in: str, video_out: str, model_a_path: str, model_b_path: str):
        self.cfg = TrafficConfig()
        print("🚀 Initializing ViAna Pipeline Components...")
        
        self.stream = VideoStreamer(video_in, video_out)
        self.engine = ClassificationEngine(self.cfg, self.stream.h)
        self.viz = Visualizer(self.cfg, self.stream.w, self.stream.h)
        self.tracker = sv.ByteTrack(frame_rate=30)
        
        self.model_a = YOLO(model_a_path); self.model_a.to(self.cfg.device_a)
        self.model_b = YOLO(model_b_path); self.model_b.to(self.cfg.device_b)
        
        self.counts_in = defaultdict(int)
        self.counts_out = defaultdict(int)
        self.horizon_slope = (self.viz.y2_h - self.viz.y1_h) / (self.viz.x2_h - self.viz.x1_h + 1e-6)

    def is_inside_vehicle(self, person_box, vehicle_boxes) -> bool:
        px1, py1, px2, py2 = person_box
        p_area = (px2 - px1) * (py2 - py1)
        if p_area == 0: return False
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            ix1, iy1 = max(px1, vx1), max(py1, vy1)
            ix2, iy2 = min(px2, vx2), min(py2, vy2)
            if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1) / p_area) > self.cfg.suppression_ioa:
                return True
        return False

    def run(self):
        start_time = time.time()
        frame_count = 0

        while self.stream.cap.isOpened():
            success, frame = self.stream.read()
            if not success: break
            frame_count += 1
            
            # --- 1. Inference ---
            res_a = self.model_a.predict(frame, imgsz=1088, conf=0.25, device=self.cfg.device_a, verbose=False)[0]
            res_b = self.model_b.predict(frame, imgsz=1088, conf=0.25, classes=[0], device=self.cfg.device_b, verbose=False)[0]

            boxes, confs, clss, vehicles = [], [], [], []
            if res_a.boxes:
                for b in res_a.boxes:
                    xyxy, c = b.xyxy[0].cpu().numpy(), int(b.cls[0])
                    boxes.append(xyxy); confs.append(float(b.conf[0])); clss.append(c)
                    if c in self.cfg.target_classes: vehicles.append(xyxy)

            if res_b.boxes:
                for b in res_b.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    if not self.is_inside_vehicle(xyxy, vehicles):
                        boxes.append(xyxy); confs.append(float(b.conf[0])); clss.append(self.cfg.pedestrian_id)

            # Check Empty Frame & Handle Logs
            if not boxes:
                self.viz.draw_dashboard(frame, self.counts_in, self.counts_out)
                self.stream.write(frame)
                if frame_count % 100 == 0: print(f"   ⏳ Processed {frame_count}/{self.stream.total_frames}")
                if frame_count % 1000 == 0: self.viz.print_report(frame_count, self.stream.total_frames, self.counts_in, self.counts_out, start_time, self.stream.fps)
                continue

            # --- 2. Tracking & Horizon Filter ---
            dets = sv.Detections(xyxy=np.array(boxes), confidence=np.array(confs), class_id=np.array(clss))
            dets = dets.with_nms(threshold=self.cfg.nms_threshold, class_agnostic=True)
            
            centers_x = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2
            centers_y = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2
            cutoff = self.viz.y1_h + self.horizon_slope * (centers_x - self.viz.x1_h)
            dets = dets[centers_y > cutoff]
            
            dets = self.tracker.update_with_detections(dets)

            # --- 3. Business Logic (Classification & Sizing) ---
            updated_ids, debug_info = [], {}
            for xyxy, t_id, raw_c_id in zip(dets.xyxy, dets.tracker_id, dets.class_id):
                final_id, area = self.engine.process_vehicle(t_id, raw_c_id, xyxy)
                updated_ids.append(final_id)
                debug_info[t_id] = {'tracked': final_id, 'raw': raw_c_id, 'area': area}
                
            dets.class_id = np.array(updated_ids)

            # --- 4. Counting Events ---
            if len(dets) > 0:
                anchors = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                proxy = sv.Detections(
                    xyxy=np.array([[x-1, y-1, x+1, y+1] for x, y in anchors]), 
                    confidence=dets.confidence, class_id=dets.class_id, tracker_id=dets.tracker_id
                )
                cross_in, cross_out = self.viz.line_zone.trigger(detections=proxy)
                for is_in, is_out, c_id in zip(cross_in, cross_out, dets.class_id):
                    if is_in: self.counts_out[c_id] += 1
                    if is_out: self.counts_in[c_id] += 1
                for x, y in anchors: cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

            # --- 5. Rendering & I/O ---
            frame = self.viz.draw_annotations(frame, dets, debug_info)
            self.viz.draw_dashboard(frame, self.counts_in, self.counts_out)
            self.stream.write(frame)

            if frame_count % 100 == 0: print(f"   ⏳ Processed {frame_count}/{self.stream.total_frames}")
            if frame_count % 1000 == 0: self.viz.print_report(frame_count, self.stream.total_frames, self.counts_in, self.counts_out, start_time, self.stream.fps)

        # Cleanup
        self.stream.close()
        self.viz.print_report(frame_count, self.stream.total_frames, self.counts_in, self.counts_out, start_time, self.stream.fps)
        print(f"✅ Pipeline complete. Output saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_a", default="/app/ViAna/models/v1/itva_medium_1088p.pt")
    parser.add_argument("--model_b", default="yolo11l.pt")
    parser.add_argument("--out", default="final_pipeline_output.mp4")
    args = parser.parse_args()
    
    pipeline = TrafficPipeline(args.video, args.out, args.model_a, args.model_b)
    pipeline.run()