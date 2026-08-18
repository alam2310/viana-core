"""Calibration-line geometry in pixel space (v2: endpoints inside the frame)."""

from __future__ import annotations

from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.io.csv_schema import CrossingDirection


def line_y_at_x(line: LineSegment, x: float) -> float:
    """Interpolate the line's y at ``x`` (vertical lines use start.y)."""
    x1, y1 = float(line.start[0]), float(line.start[1])
    x2, y2 = float(line.end[0]), float(line.end[1])
    if abs(x2 - x1) < 1e-9:
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def point_side(line: LineSegment, point: tuple[float, float]) -> float:
    """Signed side of ``point`` relative to the directed line (start → end).

    Positive is to the left of the direction vector.
    """
    x1, y1 = float(line.start[0]), float(line.start[1])
    x2, y2 = float(line.end[0]), float(line.end[1])
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def is_below_horizon(detection: Detection, horizon: LineSegment) -> bool:
    """True when the box center is on the near side of the horizon (larger image y)."""
    cutoff = line_y_at_x(horizon, detection.cx)
    return detection.cy > cutoff


def filter_below_horizon(detections: list[Detection], horizon: LineSegment) -> list[Detection]:
    """Keep detections whose center is below the horizon line (legacy TrafficPipeline)."""
    return [item for item in detections if is_below_horizon(item, horizon)]


def crossing_direction(
    counting_line: LineSegment,
    previous: tuple[float, float],
    current: tuple[float, float],
) -> CrossingDirection | None:
    """Return in/out if previous→current crosses the counting line.

    ``in`` means the track ended on the left-hand side of start→end.
    No event when the track starts on the line or stays on one side.
    """
    prev_side = point_side(counting_line, previous)
    curr_side = point_side(counting_line, current)
    if prev_side * curr_side >= 0:
        return None
    return "in" if curr_side > 0 else "out"
