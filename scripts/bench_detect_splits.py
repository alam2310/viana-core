#!/usr/bin/env python3
"""Relative detect microbench (vehicle / pedestrian / imgsz / half)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = Path("/data/raw/test_video.mp4")
VEH = Path("/app/ViAna/models/v1/itva_medium_1088p.pt")
PED = Path("/app/ViAna/models/pretrained/yolo11l.pt")
PED_M = Path("/app/ViAna/models/pretrained/yolo11m.pt")
DEVICE = "cuda:0"
N = 20


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    frames: list = []
    while len(frames) < N:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    if not frames:
        raise SystemExit("no frames")
    print(f"frames={len(frames)} shape={frames[0].shape} device={DEVICE}")

    veh = YOLO(str(VEH))
    ped = YOLO(str(PED))

    def timed(label: str, fn, warmup: int = 2) -> float:
        for _ in range(warmup):
            fn(frames[0])
        if DEVICE.startswith("cuda"):
            import torch

            torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        for im in frames:
            fn(im)
        if DEVICE.startswith("cuda"):
            import torch

            torch.cuda.synchronize(DEVICE)
        dt = time.perf_counter() - t0
        fps = len(frames) / dt
        ms = 1000 * dt / len(frames)
        print(f"{label:44s} {ms:7.1f} ms/frame  {fps:6.1f} fps")
        return ms

    # Warm load
    _ = veh.predict(frames[0], device=DEVICE, imgsz=1088, conf=0.75, verbose=False)
    _ = ped.predict(frames[0], device=DEVICE, imgsz=1088, conf=0.75, classes=[0], verbose=False)

    timed(
        "vehicle only imgsz=1088",
        lambda im: veh.predict(im, device=DEVICE, imgsz=1088, conf=0.75, verbose=False),
    )
    timed(
        "pedestrian only imgsz=1088 (yolo11l)",
        lambda im: ped.predict(im, device=DEVICE, imgsz=1088, conf=0.75, classes=[0], verbose=False),
    )
    timed(
        "BOTH sequential (current default)",
        lambda im: (
            veh.predict(im, device=DEVICE, imgsz=1088, conf=0.75, verbose=False),
            ped.predict(im, device=DEVICE, imgsz=1088, conf=0.75, classes=[0], verbose=False),
        ),
    )
    timed(
        "vehicle imgsz=960",
        lambda im: veh.predict(im, device=DEVICE, imgsz=960, conf=0.75, verbose=False),
    )
    timed(
        "vehicle imgsz=640",
        lambda im: veh.predict(im, device=DEVICE, imgsz=640, conf=0.75, verbose=False),
    )
    timed(
        "ped imgsz=640 (yolo11l)",
        lambda im: ped.predict(im, device=DEVICE, imgsz=640, conf=0.75, classes=[0], verbose=False),
    )
    timed(
        "veh1088 + ped640",
        lambda im: (
            veh.predict(im, device=DEVICE, imgsz=1088, conf=0.75, verbose=False),
            ped.predict(im, device=DEVICE, imgsz=640, conf=0.75, classes=[0], verbose=False),
        ),
    )
    timed(
        "vehicle 1088 half=True",
        lambda im: veh.predict(
            im, device=DEVICE, imgsz=1088, conf=0.75, half=True, verbose=False
        ),
    )
    timed(
        "BOTH half=True imgsz=1088",
        lambda im: (
            veh.predict(im, device=DEVICE, imgsz=1088, conf=0.75, half=True, verbose=False),
            ped.predict(
                im, device=DEVICE, imgsz=1088, conf=0.75, classes=[0], half=True, verbose=False
            ),
        ),
    )

    if PED_M.is_file():
        pedm = YOLO(str(PED_M))
        timed(
            "pedestrian only imgsz=640 (yolo11m)",
            lambda im: pedm.predict(
                im, device=DEVICE, imgsz=640, conf=0.75, classes=[0], verbose=False
            ),
        )
        timed(
            "veh1088 + pedm640",
            lambda im: (
                veh.predict(im, device=DEVICE, imgsz=1088, conf=0.75, verbose=False),
                pedm.predict(
                    im, device=DEVICE, imgsz=640, conf=0.75, classes=[0], verbose=False
                ),
            ),
        )


if __name__ == "__main__":
    main()
