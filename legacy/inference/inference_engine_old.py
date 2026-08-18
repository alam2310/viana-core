import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import cv2
import time
import datetime
import argparse
import subprocess
import numpy as np
import re
import easyocr
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
from ultralytics import YOLO
import supervision as sv

# [FIX] Import the new standalone tracker replacing sv.ByteTrack
from trackers import ByteTrackTracker


# ==============================================================================
# 1. CONFIGURATION (Data Model)
# ==============================================================================
@dataclass
class TrafficConfig:
    device_a: str = 'cuda:0'
    device_b: str = 'cuda:1'
    pedestrian_id: int = 11
    suppression_ioa: float = 0.3
    nms_threshold: float = 0.5
    
    lock_frames: int = 15
    perspective_scale: float = 6.0
    trailer_ratio: float = 2.5
    lcv_max_area: int = 25000
    mcv_max_area: int = 65000
    commercial_trucks: List[int] = field(default_factory=lambda: [7, 8, 12])
    
    horizon_left: Tuple[float, float] = (0.0, 0.6)
    horizon_right: Tuple[float, float] = (1.0, -0.4)
    counting_start: Tuple[float, float] = (0.0, 1.15)
    counting_end: Tuple[float, float] = (1.0, -0.15)
    
    target_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'Car', 1: 'Jeep', 2: 'Van', 3: 'MiniBus', 4: 'MTW', 
        5: 'Auto', 6: 'Bus', 7: 'Heavy Truck', 8: 'LCV', 9: 'Cycle', 
        10: 'Other', 11: 'Pedestrian', 12: 'MCV', 13: 'Trailer', 14: 'Taxi'
    })


# ==============================================================================
# 2. TIME & METADATA ENGINE (Dynamic Recalibration)
# ==============================================================================
class TimeSyncEngine:
    """Handles OCR extraction and dynamic interval boundary calculations."""
    def __init__(self):
        print("👁️  Initializing EasyOCR Engine (this may take a moment)...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.target_video_msec = float('inf')
        
        self.time_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        self.date_pattern = re.compile(r'\d{2}[-/]\d{2}[-/]\d{2,4}')
        self.ignore_words = {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 
                             'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}

    def extract_metadata(self, frame: np.ndarray) -> dict:
        results = self.reader.readtext(frame)
        parsed_time, date_str, location_parts = None, "Unknown", []
        
        for bbox, text, prob in results:
            text_clean = text.strip()
            
            # Time Extraction
            time_match = self.time_pattern.search(text_clean)
            if time_match:
                try: parsed_time = datetime.datetime.strptime(time_match.group(), "%H:%M:%S").time()
                except ValueError: pass
                text_clean = self.time_pattern.sub('', text_clean).strip()
            
            # Date Extraction
            date_match = self.date_pattern.search(text_clean)
            if date_match:
                date_str = date_match.group()
                text_clean = self.date_pattern.sub('', text_clean).strip()
            
            # Location Extraction
            if text_clean and text_clean.lower() not in self.ignore_words:
                loc_cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', text_clean)
                if loc_cleaned: location_parts.append(loc_cleaned)
                    
        return {'time': parsed_time, 'date': date_str, 'location': " ".join(location_parts) if location_parts else "Unknown"}

    def _calculate_next_boundary_delta(self, current_time: datetime.time) -> float:
        """Returns the exact milliseconds required to hit the next :00, :15, :30, or :45 mark."""
        dt_now = datetime.datetime.combine(datetime.date.today(), current_time)
        minutes_to_add = 15 - (dt_now.minute % 15)
        next_boundary = (dt_now + datetime.timedelta(minutes=minutes_to_add)).replace(second=0, microsecond=0)
        return (next_boundary - dt_now).total_seconds() * 1000

    def initialize_anchor(self, frame: np.ndarray):
        print("\n⚓ Anchoring Video Time Context...")
        meta = self.extract_metadata(frame)
        print(f"📌 [ANCHOR] Date: {meta['date']} | Time: {meta['time']} | Location: {meta['location']}")
        
        if meta['time']:
            diff_ms = self._calculate_next_boundary_delta(meta['time'])
            self.target_video_msec = diff_ms
            print(f"🎯 First 15-Min Boundary set at +{diff_ms:.0f}ms (Video Time)")
        else:
            print("⚠️ [WARNING] Failed to parse start time. Interval reporting disabled.")

    def check_and_update_interval(self, current_video_msec: float, frame: np.ndarray) -> bool:
        """Checks if boundary reached. If yes, runs OCR and recalibrates the NEXT boundary dynamically."""
        if current_video_msec >= self.target_video_msec:
            print(f"\n⏰ [INTERVAL] Triggered at Video Time: {current_video_msec:.0f}ms")
            meta = self.extract_metadata(frame)
            print(f"   🔍 Sanity Check - OCR reads Time: {meta['time']} | Date: {meta['date']}")
            
            if meta['time']:
                diff_ms = self._calculate_next_boundary_delta(meta['time'])
                self.target_video_msec = current_video_msec + diff_ms
                print(f"   📐 Recalibrated next boundary to trigger at {self.target_video_msec:.0f}ms (Video Time)")
            else:
                print("   ⚠️ OCR failed to read timestamp. Blindly assuming 15 mins for next jump.")
                self.target_video_msec += 900000 
            return True
        return False


# ==============================================================================
# 3. AI DETECTION ENGINE
# ==============================================================================
class DetectionEngine:
    """Encapsulates Dual-GPU YOLO inference and bounding-box merging logic."""
    def __init__(self, cfg: TrafficConfig, model_a_path: str, model_b_path: str):
        self.cfg = cfg
        self.model_a = YOLO(model_a_path).to(cfg.device_a)
        self.model_b = YOLO(model_b_path).to(cfg.device_b)

    def _is_inside_vehicle(self, person_box: np.ndarray, vehicle_boxes: List[np.ndarray]) -> bool:
        px1, py1, px2, py2 = person_box
        p_area = (px2 - px1) * (py2 - py1)
        if p_area == 0: return False
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            ix1, iy1 = max(px1, vx1), max(py1, vy1)
            ix2, iy2 = min(px2, vx2), min(py2, vy2)
            if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1) / p_area) > self.cfg.suppression_ioa:
                return True
        return False

    def predict(self, frame: np.ndarray) -> Optional[sv.Detections]:
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
                if not self._is_inside_vehicle(xyxy, vehicles):
                    boxes.append(xyxy); confs.append(float(b.conf[0])); clss.append(self.cfg.pedestrian_id)

        if not boxes: return None
        
        dets = sv.Detections(xyxy=np.array(boxes), confidence=np.array(confs), class_id=np.array(clss))
        return dets.with_nms(threshold=self.cfg.nms_threshold, class_agnostic=True)


# ==============================================================================
# 4. BUSINESS LOGIC & STATE MANAGEMENT
# ==============================================================================
class ClassificationEngine:
    """Handles perspective math, trailer detection, and the N-frame voting lock."""
    def __init__(self, config: TrafficConfig, frame_height: int):
        self.config = config
        self.h = frame_height
        self.horizon_y = config.horizon_left[1] * frame_height
        self.track_history = defaultdict(lambda: {'votes': [], 'locked_class': None, 'max_ratio': 0.0, 'max_norm_area': 0.0})

    def process_vehicle(self, tracker_id: int, raw_class: int, xyxy: np.ndarray) -> Tuple[int, int]:
        w_box, h_box = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        box_cy = (xyxy[1] + xyxy[3]) / 2
        
        raw_area = w_box * h_box
        rel_y = np.clip((box_cy - self.horizon_y) / (self.h - self.horizon_y), 0.001, 1.0)
        norm_area = raw_area * (1 + (self.config.perspective_scale - 1) * (1 - rel_y))
        ratio = w_box / h_box if h_box > 0 else 0
        state = self.track_history[tracker_id]

        if norm_area > state['max_norm_area'] * 1.5:
            state['votes'].clear()
            state['locked_class'] = None
            state['max_norm_area'] = norm_area

        if state['locked_class'] is not None:
            base_class = state['locked_class']
        else:
            state['votes'].append(raw_class)
            base_class = Counter(state['votes']).most_common(1)[0][0]
            if len(state['votes']) >= self.config.lock_frames:
                state['locked_class'] = base_class

        final_class = base_class
        if base_class in self.config.commercial_trucks:
            state['max_ratio'] = max(state['max_ratio'], ratio)
            if state['max_ratio'] > self.config.trailer_ratio:
                final_class = 13
            else:
                final_class = 8 if norm_area < self.config.lcv_max_area else (12 if norm_area < self.config.mcv_max_area else 7)

        return final_class, int(norm_area)


# ==============================================================================
# 5. I/O HANDLER
# ==============================================================================
class VideoStreamer:
    """Manages OpenCV capture and FFmpeg NVENC HEVC piping."""
    def __init__(self, input_path: str, output_path: str):
        self.cap = cv2.VideoCapture(input_path)
        self.w, self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps, self.total_frames = self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.w}x{self.h}', '-pix_fmt', 'bgr24', '-r', f'{self.fps}', '-i', '-',
            '-c:v', 'hevc_nvenc', '-pix_fmt', 'yuv420p', '-preset', 'p7',
            '-rc', 'vbr', '-cq', '38', '-b:v', '0', '-bf', '3', '-spatial-aq', '1', output_path
        ]
        self.pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # [FIX] Re-added the missing I/O methods!
    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.pipe.stdin.write(frame.tobytes())

    def close(self):
        self.cap.release()
        self.pipe.stdin.close()
        self.pipe.wait()


# ==============================================================================
# 6. PRESENTATION / VISUALIZER
# ==============================================================================
class Visualizer:
    """Handles all frame annotations and console reporting."""
    def __init__(self, config: TrafficConfig, width: int, height: int):
        self.cfg = config
        self.w, self.h = width, height
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        self.y1_h, self.y2_h = int(config.horizon_left[1] * height), int(config.horizon_right[1] * height)
        self.x1_h, self.x2_h = int(config.horizon_left[0] * width), int(config.horizon_right[0] * width)
        self.horizon_slope = (self.y2_h - self.y1_h) / (self.x2_h - self.x1_h + 1e-6)
        
        self.line_zone = sv.LineZone(
            start=sv.Point(int(config.counting_start[0] * width), int(config.counting_start[1] * height)), 
            end=sv.Point(int(config.counting_end[0] * width), int(config.counting_end[1] * height))
        )

    def draw(self, frame: np.ndarray, dets: sv.Detections, debug_info: dict, counts_in: dict, counts_out: dict) -> np.ndarray:
        labels = [
            f"#{t} {self.cfg.class_names.get(c, 'Unk')} (Raw:{self.cfg.class_names.get(debug_info.get(t, {}).get('raw', c), 'Unk')})"
            + (f" [{debug_info.get(t, {}).get('area', 0)//1000}k]" if c in self.cfg.commercial_trucks else "")
            for t, c in zip(dets.tracker_id, dets.class_id)
        ]

        frame = self.label_annotator.annotate(scene=self.box_annotator.annotate(scene=frame, detections=dets), detections=dets, labels=labels)
        
        cv2.line(frame, (self.x1_h, self.y1_h), (self.x2_h, self.y2_h), (0, 0, 255), 2)
        cv2.line(frame, (self.line_zone.start.x, self.line_zone.start.y), (self.line_zone.end.x, self.line_zone.end.y), (0, 255, 0), 2)
        
        dash = "TOTALS: " + " | ".join(f"{self.cfg.class_names[c]}: {counts_in.get(c,0)+counts_out.get(c,0)}" 
                                       for c in sorted(self.cfg.class_names.keys()) if counts_in.get(c,0)+counts_out.get(c,0) > 0)
        cv2.rectangle(frame, (0, 0), (self.w, 50), (0, 0, 0), -1)
        cv2.putText(frame, dash, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame


# ==============================================================================
# 7. ORCHESTRATOR PIPELINE (The Facade)
# ==============================================================================
class TrafficPipeline:
    def __init__(self, video_in: str, video_out: str, model_a: str, model_b: str):
        print("🚀 Initializing ViAna Pipeline Components...")
        self.cfg = TrafficConfig()
        self.stream = VideoStreamer(video_in, video_out)
        self.detector = DetectionEngine(self.cfg, model_a, model_b)
        self.classifier = ClassificationEngine(self.cfg, self.stream.h)
        self.time_sync = TimeSyncEngine()
        self.viz = Visualizer(self.cfg, self.stream.w, self.stream.h)
        
        # [FIX] Utilize the new standalone tracker replacing sv.ByteTrack
        self.tracker = ByteTrackTracker(frame_rate=30)
        self.counts_in, self.counts_out = defaultdict(int), defaultdict(int)

    def run(self):
        success, frame = self.stream.read()
        if success: self.time_sync.initialize_anchor(frame)
        self.stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        start_time = time.time()
        frame_count = 0

        while self.stream.cap.isOpened():
            success, frame = self.stream.read()
            if not success: break
            frame_count += 1
            
            self.time_sync.check_and_update_interval(self.stream.cap.get(cv2.CAP_PROP_POS_MSEC), frame)
            dets = self.detector.predict(frame)
            
            if dets is None:
                self.stream.write(self.viz.draw(frame, sv.Detections.empty(), {}, self.counts_in, self.counts_out))
                continue

            centers_y = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2
            cutoff = self.viz.y1_h + self.viz.horizon_slope * (((dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2) - self.viz.x1_h)
            
            # [FIX] Update method renamed from `update_with_detections` to `update` for new ByteTrackTracker
            dets = self.tracker.update(dets[centers_y > cutoff])

            updated_ids, debug_info = [], {}
            for xyxy, t_id, raw_c_id in zip(dets.xyxy, dets.tracker_id, dets.class_id):
                final_id, area = self.classifier.process_vehicle(t_id, raw_c_id, xyxy)
                updated_ids.append(final_id)
                debug_info[t_id] = {'tracked': final_id, 'raw': raw_c_id, 'area': area}
            dets.class_id = np.array(updated_ids)

            if len(dets) > 0:
                anchors = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                proxy = sv.Detections(xyxy=np.array([[x-1, y-1, x+1, y+1] for x, y in anchors]), class_id=dets.class_id, tracker_id=dets.tracker_id)
                cross_in, cross_out = self.viz.line_zone.trigger(detections=proxy)
                for is_in, is_out, c_id in zip(cross_in, cross_out, dets.class_id):
                    if is_in: self.counts_out[c_id] += 1
                    if is_out: self.counts_in[c_id] += 1

            self.stream.write(self.viz.draw(frame, dets, debug_info, self.counts_in, self.counts_out))

            if frame_count % 1000 == 0: print(f"   ⏳ Processed {frame_count}/{self.stream.total_frames} frames")

        self.stream.close()
        print(f"✅ Pipeline complete. Output saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_a", default="/app/ViAna/models/v1/itva_medium_1088p.pt")
    parser.add_argument("--model_b", default="yolo11l.pt")
    parser.add_argument("--out", default="final_pipeline_output.mp4")
    args = parser.parse_args()
    
    TrafficPipeline(args.video, args.out, args.model_a, args.model_b).run()