"""Auto-propose horizon and counting lines in pixel space."""

from __future__ import annotations

from math import sqrt

from pydantic import BaseModel, ConfigDict, Field

from viana.config.job import LineSegment
from viana.domain.geometry import clamp_point
from viana.io.profiles import CalibrationProfile, parse_created_at

# Normalized y (left, right). X is always pinned to frame left/right edges.
_HORIZON_Y = (0.6, 0.0)
_COUNTING_Y = (1.0, 0.0)
GEOMETRIC_CONFIDENCE = 0.4
PROFILE_CONFIDENCE = 0.85
# Parallel gap used by human review geometry B/C/D on hiv000001 (~0.25–0.28 H).
_COUNTING_OFFSET_RATIO = 0.26
_MAX_SLOPE = 0.45


class ProposedLines(BaseModel):
    """Prescan ``proposed_lines`` object."""

    model_config = ConfigDict(extra="forbid")

    horizon_line: LineSegment
    counting_line: LineSegment
    confidence: float = Field(ge=0.0, le=1.0)


def geometric_lines(width: int, height: int) -> ProposedLines:
    """Far/near lines spanning left and right frame edges (legacy in-frame clamp)."""
    x0, x1 = 0, width - 1
    counting_left_y = height - 1 if _COUNTING_Y[0] >= 1.0 else int(_COUNTING_Y[0] * height)
    horizon = LineSegment(
        start=clamp_point(x0, int(_HORIZON_Y[0] * height), width, height),
        end=clamp_point(x1, int(_HORIZON_Y[1] * height), width, height),
    )
    counting = LineSegment(
        start=clamp_point(x0, counting_left_y, width, height),
        end=clamp_point(x1, int(_COUNTING_Y[1] * height), width, height),
    )
    horizon.assert_within_frame(width, height, "horizon_line")
    counting.assert_within_frame(width, height, "counting_line")
    return ProposedLines(
        horizon_line=horizon,
        counting_line=counting,
        confidence=GEOMETRIC_CONFIDENCE,
    )


def _aspect_ratio(size: tuple[int, int]) -> float:
    return size[0] / size[1]


def best_matching_profile(
    profiles: list[CalibrationProfile],
    width: int,
    height: int,
    *,
    max_aspect_delta: float = 0.08,
) -> CalibrationProfile | None:
    """Pick the newest profile whose aspect ratio is close to the current frame."""
    target = _aspect_ratio((width, height))
    matches: list[CalibrationProfile] = []
    for profile in profiles:
        ref_w, ref_h = profile.reference_resolution
        if ref_w < 1 or ref_h < 1:
            continue
        if abs(_aspect_ratio((ref_w, ref_h)) - target) <= max_aspect_delta:
            matches.append(profile)
    if not matches:
        return None
    return max(matches, key=lambda item: parse_created_at(item.created_at))


def propose_lines(
    width: int,
    height: int,
    profiles: list[CalibrationProfile] | None = None,
    frame: object | None = None,
) -> ProposedLines:
    """Propose lines from a matching profile, else geometric defaults.

    User canvas edits remain authoritative; this is a starting overlay only.
    """
    if width < 2 or height < 2:
        raise ValueError("frame must be at least 2x2 to propose lines")
    match = best_matching_profile(profiles or [], width, height)
    if match is not None:
        scaled = match.scaled_to(width, height)
        scaled.horizon_line.assert_within_frame(width, height, "horizon_line")
        scaled.counting_line.assert_within_frame(width, height, "counting_line")
        return ProposedLines(
            horizon_line=scaled.horizon_line,
            counting_line=scaled.counting_line,
            confidence=PROFILE_CONFIDENCE,
        )
    framed = _frame_guided_lines(width, height, frame)
    if framed is not None:
        return framed
    return geometric_lines(width, height)


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    ordered = sorted(zip(values, weights, strict=False), key=lambda item: item[0])
    total = sum(weights)
    if total <= 0:
        return ordered[-1][0]
    target = total * min(max(quantile, 0.0), 1.0)
    running = 0.0
    chosen = ordered[-1][0]
    for value, weight in ordered:
        running += weight
        if running >= target:
            chosen = value
            break
    return chosen


def _span_line(width: int, height: int, y_left: float, y_right: float) -> LineSegment:
    return LineSegment(
        start=clamp_point(0, int(round(y_left)), width, height),
        end=clamp_point(width - 1, int(round(y_right)), width, height),
    )


def _edge_candidates(
    image: object,
    width: int,
    height: int,
) -> list[tuple[float, float, float, float, float, float, float]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 170)
    edges[: int(height * 0.10), :] = 0
    edges[int(height * 0.94) :, : int(width * 0.42)] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(30, int(width * 0.02)),
        minLineLength=max(50, int(width * 0.12)),
        maxLineGap=max(20, int(width * 0.04)),
    )
    candidates: list[tuple[float, float, float, float, float, float, float]] = []
    if lines is None:
        return candidates
    for raw in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in raw]
        dx = x2 - x1
        if abs(dx) < 1.0:
            continue
        dy = y2 - y1
        slope = dy / dx
        if abs(slope) > 0.55:
            continue
        length = sqrt((dx * dx) + (dy * dy))
        if length < max(30.0, width * 0.08):
            continue
        y_mid = (y1 + y2) * 0.5
        if y_mid < height * 0.22 or y_mid > height * 0.94:
            continue
        candidates.append((x1, y1, x2, y2, slope, y_mid, length))
    return candidates


def _dominant_road_slope(
    candidates: list[tuple[float, float, float, float, float, float, float]],
    height: int,
) -> float | None:
    buckets: dict[str, list[tuple[float, float, float]]] = {"neg": [], "flat": [], "pos": []}
    for _x1, _y1, _x2, _y2, slope, y_mid, length in candidates:
        road_w = length * (0.35 + 0.65 * (y_mid / height))
        if slope < -0.05:
            buckets["neg"].append((slope, road_w, y_mid))
        elif slope > 0.05:
            buckets["pos"].append((slope, road_w, y_mid))
        else:
            buckets["flat"].append((slope, road_w, y_mid))
    scored: list[tuple[float, list[tuple[float, float, float]]]] = []
    for rows in buckets.values():
        if len(rows) < 3:
            continue
        weight_sum = sum(item[1] for item in rows)
        y_mean = sum(item[2] * item[1] for item in rows) / weight_sum
        scored.append((weight_sum * (0.5 + y_mean / height), rows))
    if not scored:
        return None
    _score, rows = max(scored, key=lambda item: item[0])
    slope = _weighted_quantile([item[0] for item in rows], [item[1] for item in rows], 0.5)
    return max(-_MAX_SLOPE, min(_MAX_SLOPE, slope))


def _fit_intercept(
    candidates: list[tuple[float, float, float, float, float, float, float]],
    slope: float,
    y_min: float,
    y_max: float,
    quantile: float,
) -> tuple[float, float] | None:
    intercepts: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2, cand_slope, y_mid, length in candidates:
        if abs(cand_slope - slope) > 0.18:
            continue
        if y_mid < y_min or y_mid > y_max:
            continue
        intercepts.extend((y1 - (slope * x1), y2 - (slope * x2)))
        weights.extend((length, length))
    if len(intercepts) < 4:
        return None
    intercept = _weighted_quantile(intercepts, weights, quantile)
    support = min(1.0, len(intercepts) / 36.0)
    return intercept, support


def _frame_guided_lines(width: int, height: int, frame: object | None) -> ProposedLines | None:
    """Use road-band edge cues when a sampled BGR frame is available."""
    if frame is None:
        return None
    try:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        return None
    image = frame
    if not hasattr(image, "shape"):
        return None
    shape = image.shape
    if len(shape) < 2 or int(shape[0]) != height or int(shape[1]) != width:
        return None
    candidates = _edge_candidates(image, width, height)
    if len(candidates) < 6:
        return None
    dominant_slope = _dominant_road_slope(candidates, height)
    if dominant_slope is None:
        return None

    fitted = _fit_intercept(
        candidates,
        dominant_slope,
        height * 0.26,
        height * 0.58,
        0.35,
    )
    if fitted is None:
        intercept = height * 0.46 - (dominant_slope * (width - 1) * 0.5)
        horizon_support = 0.15
    else:
        intercept, horizon_support = fitted

    horizon = _span_line(
        width,
        height,
        intercept,
        (dominant_slope * (width - 1)) + intercept,
    )
    offset = max(24, int(height * _COUNTING_OFFSET_RATIO))
    counting = _span_line(
        width,
        height,
        horizon.start[1] + offset,
        horizon.end[1] + offset,
    )
    snapped = _fit_intercept(
        candidates,
        dominant_slope,
        height * 0.56,
        height * 0.88,
        0.45,
    )
    if snapped is not None:
        snap_intercept, counting_support = snapped
        blend = 0.35
        mixed = ((1.0 - blend) * (intercept + offset)) + (blend * snap_intercept)
        candidate = _span_line(
            width,
            height,
            mixed,
            (dominant_slope * (width - 1)) + mixed,
        )
        mid_ok = ((candidate.start[1] + candidate.end[1]) * 0.5) >= (
            (horizon.start[1] + horizon.end[1]) * 0.5 + height * 0.12
        )
        if mid_ok:
            counting = candidate
    else:
        counting_support = 0.0

    if counting.start[1] <= horizon.start[1] and counting.end[1] <= horizon.end[1]:
        counting = _span_line(
            width,
            height,
            horizon.start[1] + offset,
            horizon.end[1] + offset,
        )

    support_mean = (horizon_support + max(counting_support, 0.2)) / 2.0
    if support_mean <= 0.0:
        return None
    confidence = min(0.8, max(GEOMETRIC_CONFIDENCE + 0.05, 0.45 + (0.3 * support_mean)))
    horizon.assert_within_frame(width, height, "horizon_line")
    counting.assert_within_frame(width, height, "counting_line")
    return ProposedLines(
        horizon_line=horizon,
        counting_line=counting,
        confidence=confidence,
    )
