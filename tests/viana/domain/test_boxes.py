from viana.domain.boxes import (
    Detection,
    intersection_area,
    ioa_child_in_parent,
    iou,
    nms_class_agnostic,
)


def test_detection_properties():
    # Normal box: x1=10, y1=20, x2=30, y2=50
    d1 = Detection(x1=10.0, y1=20.0, x2=30.0, y2=50.0, confidence=0.9, class_id=1)

    assert d1.width == 20.0
    assert d1.height == 30.0
    assert d1.area == 600.0
    assert d1.cx == 20.0
    assert d1.cy == 35.0
    assert d1.bottom_center == (20.0, 50.0)
    assert d1.xyxy == (10.0, 20.0, 30.0, 50.0)


def test_detection_negative_dimensions():
    # Inverted coordinates: x1 > x2, y1 > y2
    d2 = Detection(x1=30.0, y1=50.0, x2=10.0, y2=20.0, confidence=0.8, class_id=1)

    assert d2.width == -20.0
    assert d2.height == -30.0
    assert d2.area == 0.0  # max(0, width) * max(0, height)


def test_intersection_area():
    # Box A: 0, 0, 10, 10 (Area 100)
    box_a = Detection(0.0, 0.0, 10.0, 10.0, 1.0, 1)

    # Box B: 5, 5, 15, 15 (Area 100)
    box_b = Detection(5.0, 5.0, 15.0, 15.0, 1.0, 1)
    assert intersection_area(box_a, box_b) == 25.0
    assert intersection_area(box_b, box_a) == 25.0

    # Box C: 10, 10, 20, 20 (Non-overlapping)
    box_c = Detection(10.0, 10.0, 20.0, 20.0, 1.0, 1)
    assert intersection_area(box_a, box_c) == 0.0

    # Box D: 15, 15, 20, 20 (Non-overlapping)
    box_d = Detection(15.0, 15.0, 20.0, 20.0, 1.0, 1)
    assert intersection_area(box_a, box_d) == 0.0

    # Box E: 2, 2, 8, 8 (Inside Box A, Area 36)
    box_e = Detection(2.0, 2.0, 8.0, 8.0, 1.0, 1)
    assert intersection_area(box_a, box_e) == 36.0


def test_iou():
    # Box A: 0, 0, 10, 10 (Area 100)
    box_a = Detection(0.0, 0.0, 10.0, 10.0, 1.0, 1)

    # Box B: 5, 5, 15, 15 (Area 100, Inter 25, Union 175)
    box_b = Detection(5.0, 5.0, 15.0, 15.0, 1.0, 1)
    assert iou(box_a, box_b) == 25.0 / 175.0

    # Perfect overlap
    assert iou(box_a, box_a) == 1.0

    # No overlap
    box_c = Detection(10.0, 10.0, 20.0, 20.0, 1.0, 1)
    assert iou(box_a, box_c) == 0.0


def test_ioa_child_in_parent():
    # Parent: 0, 0, 100, 100 (Area 10000)
    parent = Detection(0.0, 0.0, 100.0, 100.0, 1.0, 1)

    # Child fully inside: 10, 10, 20, 20 (Area 100)
    child1 = Detection(10.0, 10.0, 20.0, 20.0, 1.0, 1)
    assert ioa_child_in_parent(child1, parent) == 1.0

    # Child partially inside: 90, 90, 110, 110 (Area 400, Inter 100)
    child2 = Detection(90.0, 90.0, 110.0, 110.0, 1.0, 1)
    assert ioa_child_in_parent(child2, parent) == 0.25

    # Child completely outside
    child3 = Detection(110.0, 110.0, 120.0, 120.0, 1.0, 1)
    assert ioa_child_in_parent(child3, parent) == 0.0

    # Edge case: zero area child
    zero_child = Detection(0.0, 0.0, 0.0, 0.0, 1.0, 1)
    assert ioa_child_in_parent(zero_child, parent) == 0.0


def test_nms_class_agnostic():
    # Detections list
    # d1 is main box
    d1 = Detection(0.0, 0.0, 10.0, 10.0, confidence=0.9, class_id=1)
    # d2 heavily overlaps d1, lower confidence -> should be suppressed
    d2 = Detection(1.0, 1.0, 9.0, 9.0, confidence=0.8, class_id=2)
    # d3 overlaps d1 slightly, lower confidence -> should be kept with threshold 0.5
    d3 = Detection(8.0, 8.0, 18.0, 18.0, confidence=0.7, class_id=1)
    # d4 is far away -> should be kept
    d4 = Detection(100.0, 100.0, 110.0, 110.0, confidence=0.6, class_id=3)

    detections = [d4, d2, d1, d3]  # un-ordered

    # Overlap d1/d2 is 64 / 100 = 0.64
    # Overlap d1/d3 is 4 / 196 = ~0.02

    kept = nms_class_agnostic(detections, threshold=0.5)

    assert len(kept) == 3
    assert kept[0] == d1
    assert kept[1] == d3
    assert kept[2] == d4
    assert d2 not in kept

    # Lower threshold to suppress d3 as well
    kept_lower = nms_class_agnostic(detections, threshold=0.01)
    assert len(kept_lower) == 2
    assert kept_lower[0] == d1
    assert kept_lower[1] == d4
    assert d2 not in kept_lower
    assert d3 not in kept_lower
