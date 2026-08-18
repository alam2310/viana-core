"""Calibration-line geometry in pixel space (v2: endpoints inside the frame)."""

from __future__ import annotations

from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.io.csv_schema import CrossingDirection


def clamp_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """Clamp a pixel to ``[0, width) × [0, height)``."""
    if width < 1 or height < 1:
        raise ValueError("frame size must be at least 1x1")
    return min(max(0, x), width - 1), min(max(0, y), height - 1)


def scale_point(
    point: tuple[int, int],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale a pixel from ``from_size`` to ``to_size`` then clamp (UI profile apply)."""
    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w < 1 or from_h < 1:
        raise ValueError("reference resolution must be at least 1x1")
    x = round(point[0] * to_w / from_w)
    y = round(point[1] * to_h / from_h)
    return clamp_point(x, y, to_w, to_h)


def scale_line(
    line: LineSegment,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> LineSegment:
    """Scale a line segment between frame sizes; nudge if endpoints collapse."""
    start = scale_point(line.start, from_size, to_size)
    end = scale_point(line.end, from_size, to_size)
    if start == end:
        end = clamp_point(end[0] + 1, end[1], to_size[0], to_size[1])
        if start == end:
            end = clamp_point(end[0], end[1] + 1, to_size[0], to_size[1])
    scaled = LineSegment(start=start, end=end)
    scaled.assert_within_frame(to_size[0], to_size[1], "scaled_line")
    return scaled


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
