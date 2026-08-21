"""Ultralytics dual-model detector (optional; tests inject a fake detector)."""

from __future__ import annotations

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


class UltralyticsDualDetector:
    """Vehicle YOLO + pedestrian YOLO on the assigned ``cuda:0`` / ``cuda:1`` device."""

    def __init__(
        self,
        vehicle_weights: Path,
        pedestrian_weights: Path,
        *,
        device: str,
        detection: DetectionDefaults,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for live YOLO inference") from exc
        self._vehicle = YOLO(str(resolve_weights_path(vehicle_weights)))
        self._pedestrian = YOLO(str(resolve_weights_path(pedestrian_weights)))
        self._device = device
        self._imgsz = detection.imgsz
        self._conf = detection.confidence_threshold

    def detect(self, frame: VideoFrame) -> tuple[list[Detection], list[Detection]]:
        """Run both models on ``frame.image`` (must be a BGR array)."""
        if frame.image is None:
            return [], []
        vehicles = self._predict(self._vehicle, frame.image, class_ids=sorted(VEHICLE_CLASS_IDS))
        people = self._predict(self._pedestrian, frame.image, class_ids=[0])
        return vehicles, people

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
        # Convert tensors to numpy arrays once to avoid per-box iteration overhead
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            detections.append(
                Detection(
                    x1=float(xyxy_arr[i, 0]),
                    y1=float(xyxy_arr[i, 1]),
                    x2=float(xyxy_arr[i, 2]),
                    y2=float(xyxy_arr[i, 3]),
                    confidence=float(conf_arr[i]),
                    class_id=int(cls_arr[i]),
                )
            )
        return detections
