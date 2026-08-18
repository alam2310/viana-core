"""Phase 3 — detection merge, NMS, and horizon filter."""

from __future__ import annotations

from viana.config.job import LineSegment
from viana.domain.boxes import Detection, nms_class_agnostic
from viana.domain.geometry import filter_below_horizon
from viana.stages.detect import merge_detections


def _box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    conf: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=class_id)


def test_nms_keeps_highest_confidence() -> None:
    """Class-agnostic NMS drops the overlapping lower-score box."""
    kept = nms_class_agnostic(
        [_box(0, 0, 10, 10, conf=0.9), _box(1, 1, 11, 11, conf=0.5)],
        0.3,
    )
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_pedestrian_inside_vehicle_suppressed() -> None:
    """Persons contained in a vehicle box are not merged (legacy IOA)."""
    vehicle = _box(0, 0, 100, 100, class_id=0, conf=0.9)
    person = _box(20, 20, 40, 60, class_id=0, conf=0.8)
    merged = merge_detections(
        [vehicle],
        [person],
        suppression_ioa=0.3,
        nms_threshold=0.5,
        confidence_threshold=0.75,
    )
    assert [item.class_id for item in merged] == [0]


def test_standalone_pedestrian_kept() -> None:
    """A person not overlapping vehicles is remapped to class 11."""
    vehicle = _box(0, 0, 20, 20, class_id=0, conf=0.9)
    person = _box(80, 80, 100, 140, class_id=0, conf=0.8)
    merged = merge_detections(
        [vehicle],
        [person],
        suppression_ioa=0.3,
        nms_threshold=0.5,
        confidence_threshold=0.75,
    )
    ids = sorted(item.class_id for item in merged)
    assert ids == [0, 11]


def test_coco_non_person_not_mapped_to_pedestrian() -> None:
    """YOLO11 bicycle/car ids must not become Pedestrian taxonomy rows."""
    bike = _box(80, 80, 100, 140, class_id=1, conf=0.9)
    merged = merge_detections(
        [],
        [bike],
        suppression_ioa=0.3,
        nms_threshold=0.5,
        confidence_threshold=0.75,
    )
    assert merged == []


def test_unknown_vehicle_class_dropped() -> None:
    """Ids outside the ITVA vehicle set are not mapped through classes.yaml."""
    ghost = _box(0, 0, 20, 20, class_id=11, conf=0.9)
    merged = merge_detections(
        [ghost],
        [],
        suppression_ioa=0.3,
        nms_threshold=0.5,
        confidence_threshold=0.75,
    )
    assert merged == []


def test_horizon_filter_drops_far_boxes() -> None:
    """Centers above the horizon line are ignored (legacy TrafficPipeline)."""
    horizon = LineSegment(start=(0, 50), end=(200, 50))
    near = _box(10, 60, 30, 90)
    far = _box(10, 0, 30, 20)
    kept = filter_below_horizon([near, far], horizon)
    assert kept == [near]
