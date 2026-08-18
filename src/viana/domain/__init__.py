"""ViAna domain types (Phase 3+)."""

from viana.domain.boxes import Detection
from viana.domain.geometry import clamp_point, crossing_direction, filter_below_horizon, scale_line

__all__ = [
    "Detection",
    "clamp_point",
    "crossing_direction",
    "filter_below_horizon",
    "scale_line",
]
