"""Phase 3 — heuristic classification (legacy ClassificationEngine)."""

from __future__ import annotations

from viana.config.defaults import load_engine_defaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.stages.classify import HEAVY_TRUCK_ID, LCV_ID, TRAILER_ID, ClassificationEngine


def _engine(*, heuristic: bool = True) -> ClassificationEngine:
    defaults = load_engine_defaults().classification
    defaults.use_heuristic_truck_split = heuristic
    horizon = LineSegment(start=(0, 400), end=(1920, 400))
    return ClassificationEngine(defaults, horizon, 1080)


def test_truck_split_trailer_by_aspect() -> None:
    """Wide commercial boxes become Trailer (id 13)."""
    engine = _engine()
    box = Detection(x1=100, y1=800, x2=400, y2=900, confidence=0.9, class_id=7)
    final_id, area = engine.process_vehicle(1, 7, box)
    assert final_id == TRAILER_ID
    assert area > 0


def test_truck_split_lcv_by_norm_area() -> None:
    """Smaller commercial boxes become LCV (id 8)."""
    engine = _engine()
    box = Detection(x1=100, y1=850, x2=180, y2=930, confidence=0.9, class_id=7)
    final_id, _area = engine.process_vehicle(1, 7, box)
    assert final_id == LCV_ID


def test_heuristic_flag_skips_truck_split() -> None:
    """use_heuristic_truck_split=false keeps the raw commercial class."""
    engine = _engine(heuristic=False)
    box = Detection(x1=100, y1=800, x2=400, y2=900, confidence=0.9, class_id=7)
    final_id, _area = engine.process_vehicle(1, 7, box)
    assert final_id == HEAVY_TRUCK_ID


def test_class_lock_after_lock_frames() -> None:
    """Majority vote locks after lock_frames samples."""
    engine = _engine()
    box = Detection(x1=10, y1=800, x2=40, y2=840, confidence=0.9, class_id=0)
    for _ in range(15):
        engine.process_vehicle(3, 0, box)
    # After lock, a single conflicting vote must not change the locked class.
    final_id, _area = engine.process_vehicle(3, 1, box)
    assert final_id == 0
