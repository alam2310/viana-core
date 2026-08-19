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


def _frame_guided_lines(width: int, height: int, frame: object | None) -> ProposedLines | None:
    """Use deterministic edge/line cues when a sampled BGR frame is available."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    image = frame
    if not hasattr(image, "shape"):
        return None
    shape = image.shape
    if len(shape) < 2 or int(shape[0]) != height or int(shape[1]) != width:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 170)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(30, int(width * 0.02)),
        minLineLength=max(50, int(width * 0.14)),
        maxLineGap=max(20, int(width * 0.04)),
    )
    candidates: list[tuple[float, float, float, float, float, float]] = []
    if lines is not None:
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
            candidates.append((x1, y1, x2, y2, slope, y_mid))

    def fit_band_line(
        y_min_ratio: float,
        y_max_ratio: float,
        slope_limit: float,
        fallback_y: float,
    ) -> tuple[LineSegment, float]:
        y_min = height * y_min_ratio
        y_max = height * y_max_ratio
        points_x: list[float] = []
        points_y: list[float] = []
        weights: list[float] = []
        for x1, y1, x2, y2, slope, y_mid in candidates:
            if abs(slope) > slope_limit:
                continue
            if y_mid < y_min or y_mid > y_max:
                continue
            seg_len = sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
            points_x.extend((x1, x2))
            points_y.extend((y1, y2))
            weights.extend((seg_len, seg_len))
        if len(points_x) < 4:
            return (
                LineSegment(
                    start=clamp_point(0, int(fallback_y), width, height),
                    end=clamp_point(width - 1, int(fallback_y), width, height),
                ),
                0.0,
            )
        coeff = np.polyfit(
            np.asarray(points_x, dtype=np.float64),
            np.asarray(points_y, dtype=np.float64),
            1,
            w=np.asarray(weights, dtype=np.float64),
        )
        slope = float(coeff[0])
        intercept = float(coeff[1])
        y_left = int(round(intercept))
        y_right = int(round((slope * (width - 1)) + intercept))
        support = min(1.0, len(points_x) / 32.0)
        return (
            LineSegment(
                start=clamp_point(0, y_left, width, height),
                end=clamp_point(width - 1, y_right, width, height),
            ),
            support,
        )

    horizon, horizon_support = fit_band_line(0.22, 0.72, 0.28, height * 0.6)
    counting, counting_support = fit_band_line(0.45, 0.96, 0.35, height * 0.86)

    if counting.start[1] <= horizon.start[1] and counting.end[1] <= horizon.end[1]:
        offset = max(12, int(height * 0.08))
        counting = LineSegment(
            start=clamp_point(0, horizon.start[1] + offset, width, height),
            end=clamp_point(width - 1, horizon.end[1] + offset, width, height),
        )

    support_mean = (horizon_support + counting_support) / 2.0
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
