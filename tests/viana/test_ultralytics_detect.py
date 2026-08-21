"""Concurrent dual-YOLO detector helpers (no real Ultralytics weights)."""

from __future__ import annotations

import time
from types import SimpleNamespace

from viana.config.defaults import DetectionDefaults
from viana.stages.ultralytics_detect import UltralyticsDualDetector
from viana.stages.video import VideoFrame


class _FakeBoxes:
    def __iter__(self):
        return iter(())


class _FakeResult:
    boxes = _FakeBoxes()


class _SleepModel:
    def __init__(self, delay_sec: float) -> None:
        self.delay_sec = delay_sec
        self.calls = 0

    def predict(self, _image: object, **_kwargs: object) -> list[_FakeResult]:
        self.calls += 1
        time.sleep(self.delay_sec)
        return [_FakeResult()]


def _detector(*, concurrent: bool, delay_sec: float = 0.05) -> UltralyticsDualDetector:
    return UltralyticsDualDetector(
        vehicle_weights=__file__,  # unused when models injected
        pedestrian_weights=__file__,
        device="cpu",
        detection=DetectionDefaults(
            confidence_threshold=0.75,
            imgsz=640,
            nms_threshold=0.5,
            suppression_ioa=0.3,
        ),
        concurrent=concurrent,
        vehicle_model=_SleepModel(delay_sec),
        pedestrian_model=_SleepModel(delay_sec),
    )


def test_concurrent_detect_overlaps_model_predict() -> None:
    """Two sleeping models finish near max(delay) when concurrent, not 2×delay."""
    frame = VideoFrame(index=0, pts_ms=0, width=64, height=64, image=SimpleNamespace())
    concurrent = _detector(concurrent=True, delay_sec=0.05)
    sequential = _detector(concurrent=False, delay_sec=0.05)
    try:
        t0 = time.perf_counter()
        concurrent.detect(frame)
        concurrent_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        sequential.detect(frame)
        sequential_sec = time.perf_counter() - t1
    finally:
        concurrent.close()
        sequential.close()

    assert concurrent_sec < sequential_sec * 0.75
    assert sequential_sec >= 0.09


def test_close_is_idempotent() -> None:
    detector = _detector(concurrent=True)
    detector.close()
    detector.close()
