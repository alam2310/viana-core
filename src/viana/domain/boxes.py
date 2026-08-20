"""Pixel-space detections and NMS (no GPU / numpy)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """Axis-aligned box in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in pixels²."""
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        """Horizontal center."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Vertical center."""
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Anchor used for line crossing (legacy BOTTOM_CENTER)."""
        return (self.cx, self.y2)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


def intersection_area(a: Detection, b: Detection) -> float:
    """Return the overlapping rectangle area of two boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two boxes."""
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def ioa_child_in_parent(child: Detection, parent: Detection) -> float:
    """Intersection over child's area (legacy pedestrian-in-vehicle test)."""
    if child.area <= 0.0:
        return 0.0
    return intersection_area(child, parent) / child.area


def nms_class_agnostic(detections: list[Detection], threshold: float) -> list[Detection]:
    """Greedy class-agnostic NMS (legacy ``Detections.with_nms(..., class_agnostic=True)``)."""
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []
    suppressed = [False] * len(ordered)
    for index, candidate in enumerate(ordered):
        if suppressed[index]:
            continue
        kept.append(candidate)
        cx1, cy1, cx2, cy2, carea = (
            candidate.x1,
            candidate.y1,
            candidate.x2,
            candidate.y2,
            candidate.area,
        )
        for other_index in range(index + 1, len(ordered)):
            if suppressed[other_index]:
                continue
            other = ordered[other_index]
            ox1, oy1, ox2, oy2 = other.x1, other.y1, other.x2, other.y2

            # Fast AABB intersection check
            if cx1 >= ox2 or cx2 <= ox1 or cy1 >= oy2 or cy2 <= oy1:
                continue

            ix1 = cx1 if cx1 > ox1 else ox1
            iy1 = cy1 if cy1 > oy1 else oy1
            ix2 = cx2 if cx2 < ox2 else ox2
            iy2 = cy2 if cy2 < oy2 else oy2

            inter = (ix2 - ix1) * (iy2 - iy1)
            union = carea + other.area - inter

            if (inter / union) >= threshold:
                suppressed[other_index] = True
    return kept
