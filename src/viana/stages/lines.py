"""Auto-propose horizon and counting lines in pixel space."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from viana.config.job import LineSegment
from viana.domain.geometry import clamp_point
from viana.io.profiles import CalibrationProfile, parse_created_at

# Normalized y (left, right). X is always pinned to frame left/right edges.
_HORIZON_Y = (0.6, 0.0)
_COUNTING_Y = (1.0, 0.0)
GEOMETRIC_CONFIDENCE = 0.4
PROFILE_CONFIDENCE = 0.85
FRAME_CONFIDENCE = 0.65


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
    """Use Hough line cues when a sampled BGR frame is available."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=50,
        minLineLength=max(40, int(width * 0.12)),
        maxLineGap=max(20, int(width * 0.03)),
    )
    if lines is None:
        return None
    horizon_candidates: list[tuple[float, float, float, float, float]] = []
    counting_candidates: list[tuple[float, float, float, float, float]] = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        if abs(dx) < 1:
            continue
        dy = y2 - y1
        slope = dy / dx
        length = float(((dx**2) + (dy**2)) ** 0.5)
        y_mid = (y1 + y2) * 0.5
        if abs(slope) <= 0.25 and y_mid <= height * 0.78:
            horizon_candidates.append((x1, y1, x2, y2, length))
        if abs(slope) <= 0.35 and y_mid >= height * 0.45:
            counting_candidates.append((x1, y1, x2, y2, length))
    if not horizon_candidates and not counting_candidates:
        return None

    def best_line(
        candidates: list[tuple[float, float, float, float, float]],
        fallback_y: float,
    ) -> LineSegment:
        if not candidates:
            return LineSegment(
                start=clamp_point(0, int(fallback_y), width, height),
                end=clamp_point(width - 1, int(fallback_y), width, height),
            )
        best = max(candidates, key=lambda item: item[4])
        x1, y1, x2, y2, _len = best
        if abs(x2 - x1) < 1:
            y = int((y1 + y2) * 0.5)
            return LineSegment(
                start=clamp_point(0, y, width, height),
                end=clamp_point(width - 1, y, width, height),
            )
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        y_left = int(intercept)
        y_right = int((slope * (width - 1)) + intercept)
        return LineSegment(
            start=clamp_point(0, y_left, width, height),
            end=clamp_point(width - 1, y_right, width, height),
        )

    horizon = best_line(horizon_candidates, fallback_y=height * 0.6)
    counting = best_line(counting_candidates, fallback_y=height * 0.85)
    horizon.assert_within_frame(width, height, "horizon_line")
    counting.assert_within_frame(width, height, "counting_line")
    return ProposedLines(
        horizon_line=horizon,
        counting_line=counting,
        confidence=FRAME_CONFIDENCE,
    )
