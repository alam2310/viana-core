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
    return geometric_lines(width, height)
