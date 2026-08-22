#!/usr/bin/env python3
"""End-to-end process benchmark: baseline vs P1a vs P1b on a short clip."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from viana.config.defaults import load_engine_defaults
from viana.config.job import JobConfig, JobMetadata, ViAnaTaskParameters
from viana.io.events import read_events
from viana.stages.detect import PEDESTRIAN_ID
from viana.stages.lines import geometric_lines
from viana.stages.process import run_moving_count
from viana.stages.ultralytics_detect import UltralyticsDualDetector, resolve_weights_path

VIDEO = Path("/data/raw/test_video.mp4")
OUT_ROOT = Path("/data/viana-outputs/bench-detect-modes")
DEVICE = "cuda:1"
CONF = 0.75


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    veh_imgsz: int
    ped_imgsz: int
    ped_enabled: bool


MODES = (
    Mode("baseline", "Baseline (veh@1088 + ped@1088)", 1088, 1088, True),
    Mode("p1a", "P1a (veh@1088 + ped@640)", 1088, 640, True),
    Mode("p1b", "P1b (vehicle-only @1088)", 1088, 1088, False),
)


def make_detector(mode: Mode, defaults) -> Callable:
    """Build a frame detector for the given benchmark mode."""
    veh_w = resolve_weights_path(defaults.models.vehicle)
    ped_w = resolve_weights_path(defaults.models.pedestrian)

    if mode.name == "baseline":
        dual = UltralyticsDualDetector(
            veh_w,
            ped_w,
            device=DEVICE,
            detection=defaults.detection.model_copy(
                update={"confidence_threshold": CONF, "imgsz": 1088}
            ),
        )
        return dual.detect

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics required") from exc

    vehicle = YOLO(str(veh_w))
    pedestrian = YOLO(str(ped_w)) if mode.ped_enabled else None

    def detect(frame):
        if frame.image is None:
            return [], []
        veh_results = vehicle.predict(
            frame.image,
            device=DEVICE,
            imgsz=mode.veh_imgsz,
            conf=CONF,
            verbose=False,
        )
        vehicles = _boxes(veh_results)
        if not mode.ped_enabled:
            return vehicles, []
        ped_results = pedestrian.predict(  # type: ignore[union-attr]
            frame.image,
            device=DEVICE,
            imgsz=mode.ped_imgsz,
            conf=CONF,
            classes=[0],
            verbose=False,
        )
        people = _boxes(ped_results)
        return vehicles, people

    return detect


def _boxes(results) -> list:
    from viana.domain.boxes import Detection

    out: list[Detection] = []
    if not results:
        return out
    boxes = results[0].boxes
    if boxes is None:
        return out
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        out.append(
            Detection(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=float(box.conf[0]),
                class_id=int(box.cls[0]),
            )
        )
    return out


def run_mode(mode: Mode, defaults) -> dict[str, Any]:
    """Run full process loop for one mode; return metrics."""
    out_dir = OUT_ROOT / mode.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    props = geometric_lines(848, 478)
    job = JobConfig(
        source_video_path=VIDEO,
        project_id="bench-detect",
        metadata=JobMetadata(
            user_start_time="08:38:31",
            user_start_date="18-10-2024",
            location="bench",
        ),
        task_parameters=ViAnaTaskParameters(
            horizon_line=props.horizon_line,
            counting_line=props.counting_line,
            confidence_threshold=CONF,
            render_video=False,
            telemetry_detail=False,
        ),
        job_id=f"bench_{mode.name}",
        gpu_device=DEVICE,
        output_dir=out_dir,
        start_fresh=True,
    )

    stage_sec: dict[str, float] | None = None
    events_rows = 0

    def emit(msg) -> None:
        nonlocal stage_sec, events_rows
        if msg.telemetry_type == "LOG" and msg.data.get("message") == "process_complete":
            stage_sec = msg.data.get("stage_sec")
            events_rows = int(msg.data.get("events_rows", 0))

    detector = make_detector(mode, defaults)
    t0 = time.perf_counter()
    run_moving_count(job, resume=False, detector=detector, emit=emit)
    wall = time.perf_counter() - t0

    events_path = out_dir / "test_video_events.csv"
    rows = list(read_events(events_path)) if events_path.is_file() else []
    ped_events = sum(1 for r in rows if r.class_id == PEDESTRIAN_ID)
    veh_events = len(rows) - ped_events

    # 168 frames @ 15fps nominal
    frames = 168
    fps = frames / wall if wall > 0 else 0.0
    detect_sec = (stage_sec or {}).get("detect", 0.0)
    render_sec = (stage_sec or {}).get("render", 0.0)
    track_sec = (stage_sec or {}).get("track", 0.0)

    return {
        "mode": mode.name,
        "label": mode.label,
        "wall_sec": round(wall, 2),
        "avg_fps": round(fps, 2),
        "events_total": len(rows),
        "events_vehicle": veh_events,
        "events_pedestrian": ped_events,
        "stage_sec": stage_sec,
        "detect_pct": round(100 * detect_sec / wall, 1) if wall else None,
        "render_sec": render_sec,
        "track_sec": track_sec,
    }


def main() -> None:
    if not VIDEO.is_file():
        raise SystemExit(f"video missing: {VIDEO}")
    defaults = load_engine_defaults()
    print(f"clip={VIDEO} device={DEVICE} render_video=false")
    print(f"frames≈168 duration≈11.2s")
    print()

    results = []
    for mode in MODES:
        print(f"Running {mode.label}…", flush=True)
        results.append(run_mode(mode, defaults))

    print()
    print("| Mode | Wall (s) | Avg FPS | Events | Ped events | Detect % | Render (s) |")
    print("|------|----------|---------|--------|------------|----------|------------|")
    baseline_fps = results[0]["avg_fps"]
    for r in results:
        gain = ""
        if r["mode"] != "baseline" and baseline_fps:
            pct = 100 * (r["avg_fps"] / baseline_fps - 1)
            gain = f" ({pct:+.0f}%)"
        print(
            f"| {r['label']} | {r['wall_sec']} | {r['avg_fps']}{gain} | "
            f"{r['events_total']} | {r['events_pedestrian']} | {r['detect_pct']}% | {r['render_sec']} |"
        )

    print()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
