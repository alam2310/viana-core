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
    counting = LineSegment(start=(0, 820), end=(1920, 820))
    return ClassificationEngine(defaults, horizon, 1080, counting)


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


def test_pedestrian_votes_then_locks() -> None:
    """Persons use the same N-frame lock as vehicles (not a per-frame raw id)."""
    engine = _engine()
    box = Detection(x1=10, y1=800, x2=30, y2=860, confidence=0.9, class_id=11)
    for _ in range(15):
        engine.process_vehicle(9, 11, box)
    final_id, _area = engine.process_vehicle(9, 0, box)
    assert final_id == 11


def test_lock_survives_area_jump() -> None:
    """Once locked, a 1.5× larger box must not unlock and flip class."""
    engine = _engine()
    small = Detection(x1=10, y1=800, x2=40, y2=840, confidence=0.9, class_id=0)
    large = Detection(x1=10, y1=600, x2=200, y2=900, confidence=0.9, class_id=1)
    for _ in range(15):
        engine.process_vehicle(4, 0, small)
    final_id, _area = engine.process_vehicle(4, 1, large)
    assert final_id == 0


def test_bus_is_not_truck_split() -> None:
    """Bus (Passenger / Heavy Fast / Bus) must not become MCV via area heuristic."""
    engine = _engine()
    box = Detection(x1=100, y1=800, x2=400, y2=1000, confidence=0.9, class_id=6)
    for _ in range(15):
        engine.process_vehicle(2, 6, box)
    final_id, _area = engine.process_vehicle(2, 6, box)
    assert final_id == 6


def test_distant_overlay_ignores_single_frame_flip() -> None:
    """Far from the counting line, overlay follows recent majority, not one YOLO miss."""
    engine = _engine()
    far_auto = Detection(x1=100, y1=390, x2=160, y2=450, confidence=0.9, class_id=5)
    for _ in range(10):
        engine.process_vehicle(12, 5, far_auto)
    final_id, _area = engine.process_vehicle(12, 8, far_auto)
    assert final_id == 5


def test_distant_votes_do_not_lock_class() -> None:
    """Blobs near the horizon do not lock taxonomy before the object is closer."""
    engine = _engine()
    far = Detection(x1=100, y1=390, x2=140, y2=430, confidence=0.9, class_id=7)
    near = Detection(x1=100, y1=800, x2=180, y2=900, confidence=0.9, class_id=6)
    for _ in range(20):
        engine.process_vehicle(8, 7, far)
    for _ in range(15):
        engine.process_vehicle(8, 6, near)
    final_id, _area = engine.process_vehicle(8, 6, near)
    assert final_id == 6


def test_lcv_vote_is_not_promoted_to_heavy_truck() -> None:
    """YOLO LCV (8) must stay LCV even when the box grows past the heavy-truck area cut."""
    engine = _engine()
    small = Detection(x1=100, y1=800, x2=180, y2=880, confidence=0.9, class_id=8)
    large = Detection(x1=50, y1=600, x2=400, y2=1000, confidence=0.9, class_id=8)
    for _ in range(15):
        engine.process_vehicle(5, 8, small)
    final_id, _area = engine.process_vehicle(5, 8, large)
    assert final_id == LCV_ID


def test_counted_track_stays_frozen() -> None:
    """After a crossing, later raw ids do not change the locked class."""
    engine = _engine()
    box = Detection(x1=10, y1=800, x2=40, y2=840, confidence=0.9, class_id=0)
    engine.freeze(6, 0)
    final_id, _area = engine.process_vehicle(6, 8, box, counted=True)
    assert final_id == 0
