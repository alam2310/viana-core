"""Heuristic class lock and truck split (legacy ClassificationEngine)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from viana.config.defaults import ClassificationDefaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.domain.geometry import line_y_at_x, point_to_line_distance
from viana.stages.detect import PEDESTRIAN_ID

HEAVY_TRUCK_ID = 7
LCV_ID = 8
MCV_ID = 12
TRAILER_ID = 13
COMMERCIAL_TRUCK_IDS = (HEAVY_TRUCK_ID, LCV_ID, MCV_ID)
# Vote/lock only when the bottom-center is this close to the counting line.
COUNTING_LOCK_FRAC = 0.08
COUNTING_LOCK_MIN_PX = 48.0


@dataclass
class TrackClassState:
    """Per-track voting lock used by the heuristic classifier."""

    votes: list[int] = field(default_factory=list)
    recent: list[int] = field(default_factory=list)
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
        counting_line: LineSegment,
    ) -> None:
        if frame_height < 1:
            raise ValueError("frame_height must be positive")
        self.defaults = defaults
        self.horizon = horizon
        self.counting_line = counting_line
        self.frame_height = frame_height
        self._lock_distance = max(COUNTING_LOCK_MIN_PX, COUNTING_LOCK_FRAC * frame_height)
        self._history: dict[int, TrackClassState] = defaultdict(TrackClassState)

    def freeze(self, tracker_id: int, class_id: int) -> None:
        """Keep this class after the track has crossed the counting line."""
        self._history[int(tracker_id)].locked_class = int(class_id)

    def process_vehicle(
        self,
        tracker_id: int,
        raw_class: int,
        detection: Detection,
        *,
        counted: bool = False,
    ) -> tuple[int, int]:
        """Return ``(final_class_id, norm_area)`` for one tracked box."""
        tracker_id = int(tracker_id)
        horizon_y = line_y_at_x(self.horizon, detection.cx)
        denom = max(self.frame_height - horizon_y, 1e-6)
        rel_y = (detection.cy - horizon_y) / denom
        rel_y = min(1.0, max(0.001, rel_y))
        raw_area = detection.width * detection.height
        scale = self.defaults.perspective_scale
        norm_area = raw_area * (1.0 + (scale - 1.0) * (1.0 - rel_y))
        ratio = detection.width / detection.height if detection.height > 0 else 0.0
        state = self._history[tracker_id]
        tentative_area = max(state.max_norm_area, norm_area)
        live = self._sticky_class(state, raw_class, ratio, tentative_area)

        if state.locked_class is not None:
            state.max_norm_area = tentative_area
            return state.locked_class, int(norm_area)

        if counted:
            state.max_norm_area = tentative_area
            state.locked_class = live
            return live, int(norm_area)

        near_line = (
            point_to_line_distance(self.counting_line, detection.bottom_center)
            <= self._lock_distance
        )
        if not near_line:
            state.max_norm_area = tentative_area
            return live, int(norm_area)

        if state.max_norm_area > 0 and norm_area > state.max_norm_area * 1.5:
            state.votes.clear()
        state.max_norm_area = tentative_area
        state.votes.append(raw_class)
        base_class = Counter(state.votes).most_common(1)[0][0]
        final_class = self._finalize(state, base_class, ratio, state.max_norm_area)
        if len(state.votes) >= self.defaults.lock_frames:
            state.locked_class = final_class
        return final_class, int(norm_area)

    def _sticky_class(
        self,
        state: TrackClassState,
        raw_class: int,
        ratio: float,
        tentative_area: float,
    ) -> int:
        """Majority of recent raw ids so overlay does not follow single-frame YOLO flips."""
        keep = max(1, self.defaults.lock_frames)
        state.recent.append(int(raw_class))
        if len(state.recent) > keep:
            del state.recent[:-keep]
        base = Counter(state.recent).most_common(1)[0][0]
        return self._finalize(state, base, ratio, tentative_area)

    def _finalize(
        self,
        state: TrackClassState,
        base_class: int,
        ratio: float,
        norm_area: float,
    ) -> int:
        """Apply truck-split only when the voted class is undifferentiated Heavy Truck."""
        if base_class == PEDESTRIAN_ID:
            return PEDESTRIAN_ID
        if not self.defaults.use_heuristic_truck_split:
            return base_class
        if base_class in COMMERCIAL_TRUCK_IDS and base_class != HEAVY_TRUCK_ID:
            return base_class
        if base_class != HEAVY_TRUCK_ID:
            return base_class
        state.max_ratio = max(state.max_ratio, ratio)
        if state.max_ratio > self.defaults.trailer_ratio:
            return TRAILER_ID
        if norm_area < self.defaults.lcv_max_area:
            return LCV_ID
        if norm_area < self.defaults.mcv_max_area:
            return MCV_ID
        return HEAVY_TRUCK_ID
