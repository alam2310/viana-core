"""Ultralytics dual-model detector (optional; tests inject a fake detector)."""

from __future__ import annotations

import concurrent.futures as futures
import os
from pathlib import Path
from typing import Any

from viana.config.defaults import DetectionDefaults
from viana.config.files import repo_root
from viana.domain.boxes import Detection
from viana.stages.detect import VEHICLE_CLASS_IDS
from viana.stages.video import VideoFrame


def resolve_weights_path(path: Path) -> Path:
    """Resolve model weights relative to the repo root when needed."""
    if path.is_file():
        return path
    candidate = repo_root() / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Model weights not found: {path}")


def dual_detect_concurrent_enabled() -> bool:
    """Return whether vehicle+pedestrian predict should overlap on threads.

    Set ``VIANA_DUAL_DETECT_CONCURRENT=0`` for sequential A/B FPS comparison.
    Default is concurrent (``1``).
    """
    raw = os.environ.get("VIANA_DUAL_DETECT_CONCURRENT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class UltralyticsDualDetector:
    """Vehicle YOLO + pedestrian YOLO on the assigned ``cuda:0`` / ``cuda:1`` device."""

    def __init__(
        self,
        vehicle_weights: Path,
        pedestrian_weights: Path,
        *,
        device: str,
        detection: DetectionDefaults,
        concurrent: bool | None = None,
        vehicle_model: Any | None = None,
        pedestrian_model: Any | None = None,
    ) -> None:
        if vehicle_model is None or pedestrian_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for live YOLO inference") from exc
            if vehicle_model is None:
                vehicle_model = YOLO(str(resolve_weights_path(vehicle_weights)))
            if pedestrian_model is None:
                pedestrian_model = YOLO(str(resolve_weights_path(pedestrian_weights)))
        self._vehicle = vehicle_model
        self._pedestrian = pedestrian_model
        self._device = device
        self._imgsz = detection.imgsz
        self._conf = detection.confidence_threshold
        self._concurrent = dual_detect_concurrent_enabled() if concurrent is None else concurrent
        self._pool = futures.ThreadPoolExecutor(max_workers=2) if self._concurrent else None
        self._closed = False

    def detect(self, frame: VideoFrame) -> tuple[list[Detection], list[Detection]]:
        """Run both models on ``frame.image`` (must be a BGR array)."""
        if frame.image is None:
            return [], []
        if self._pool is None:
            vehicles = self._predict(
                self._vehicle, frame.image, class_ids=sorted(VEHICLE_CLASS_IDS)
            )
            people = self._predict(self._pedestrian, frame.image, class_ids=[0])
            return vehicles, people

        future_vehicles = self._pool.submit(
            self._predict, self._vehicle, frame.image, class_ids=sorted(VEHICLE_CLASS_IDS)
        )
        future_people = self._pool.submit(
            self._predict, self._pedestrian, frame.image, class_ids=[0]
        )
        return future_vehicles.result(), future_people.result()

    def close(self) -> None:
        """Shut down the prediction thread pool (idempotent)."""
        if self._closed:
            return
        self._closed = True
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    def _predict(
        self, model: Any, image: object, *, class_ids: list[int] | None = None
    ) -> list[Detection]:
        predict_kw: dict[str, Any] = {
            "device": self._device,
            "imgsz": self._imgsz,
            "conf": self._conf,
            "verbose": False,
        }
        if class_ids is not None:
            predict_kw["classes"] = class_ids
        results = model.predict(image, **predict_kw)
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                )
            )
        return detections
