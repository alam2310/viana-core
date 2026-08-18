"""Heuristic class lock and truck split (legacy ClassificationEngine)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from viana.config.defaults import ClassificationDefaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.domain.geometry import line_y_at_x
from viana.stages.detect import PEDESTRIAN_ID

HEAVY_TRUCK_ID = 7
LCV_ID = 8
MCV_ID = 12
TRAILER_ID = 13
COMMERCIAL_TRUCK_IDS = (HEAVY_TRUCK_ID, LCV_ID, MCV_ID)


@dataclass
class TrackClassState:
    """Per-track voting lock used by the heuristic classifier."""

    votes: list[int] = field(default_factory=list)
    locked_class: int | None = None
    max_ratio: float = 0.0
    max_norm_area: float = 0.0


class ClassificationEngine:
    """N-frame majority lock plus optional LCV/MCV/truck/trailer split."""

    def __init__(
        self,
        defaults: ClassificationDefaults,
        horizon: LineSegment,
        frame_height: int,
    ) -> None:
        if frame_height < 1:
            raise ValueError("frame_height must be positive")
        self.defaults = defaults
        self.horizon = horizon
        self.frame_height = frame_height
        self._history: dict[int, TrackClassState] = defaultdict(TrackClassState)

    def process_vehicle(
        self, tracker_id: int, raw_class: int, detection: Detection
    ) -> tuple[int, int]:
        """Return ``(final_class_id, norm_area)`` for one tracked box."""
        horizon_y = line_y_at_x(self.horizon, detection.cx)
        denom = max(self.frame_height - horizon_y, 1e-6)
        rel_y = (detection.cy - horizon_y) / denom
        rel_y = min(1.0, max(0.001, rel_y))
        raw_area = detection.width * detection.height
        scale = self.defaults.perspective_scale
        norm_area = raw_area * (1.0 + (scale - 1.0) * (1.0 - rel_y))
        ratio = detection.width / detection.height if detection.height > 0 else 0.0
        state = self._history[tracker_id]

        if raw_class == PEDESTRIAN_ID:
            return PEDESTRIAN_ID, int(norm_area)

        if norm_area > state.max_norm_area * 1.5:
            state.votes.clear()
            state.locked_class = None
            state.max_norm_area = norm_area

        if state.locked_class is not None:
            base_class = state.locked_class
        else:
            state.votes.append(raw_class)
            base_class = Counter(state.votes).most_common(1)[0][0]
            if len(state.votes) >= self.defaults.lock_frames:
                state.locked_class = base_class

        final_class = base_class
        if self.defaults.use_heuristic_truck_split and base_class in COMMERCIAL_TRUCK_IDS:
            state.max_ratio = max(state.max_ratio, ratio)
            if state.max_ratio > self.defaults.trailer_ratio:
                final_class = TRAILER_ID
            elif norm_area < self.defaults.lcv_max_area:
                final_class = LCV_ID
            elif norm_area < self.defaults.mcv_max_area:
                final_class = MCV_ID
            else:
                final_class = HEAVY_TRUCK_ID

        return final_class, int(norm_area)
