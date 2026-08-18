"""Line-crossing events with once-per-track de-dupe (no 15-min bins)."""

from __future__ import annotations

from dataclasses import dataclass, field

from viana.config.job import LineSegment
from viana.domain.geometry import crossing_direction
from viana.io.csv_schema import CrossingDirection
from viana.stages.track import TrackedDetection


@dataclass(frozen=True, slots=True)
class Crossing:
    """One unique track crossing of the counting line."""

    track_id: int
    class_id: int
    raw_class_id: int
    direction: CrossingDirection
    confidence: float
    norm_area: int
    anchor_x: float
    anchor_y: float
    frame_index: int
    video_pts_ms: float


@dataclass
class CrossingState:
    """Remember last anchors and which tracks have already counted."""

    counting_line: LineSegment
    counted_track_ids: set[int] = field(default_factory=set)
    _previous: dict[int, tuple[float, float]] = field(default_factory=dict)

    def update(
        self,
        tracked: list[TrackedDetection],
        *,
        class_ids: dict[int, int],
        norm_areas: dict[int, int],
        frame_index: int,
        video_pts_ms: float,
    ) -> list[Crossing]:
        """Emit crossings for tracks that just crossed and are not yet counted."""
        events: list[Crossing] = []
        live: set[int] = set()
        for item in tracked:
            live.add(item.track_id)
            current = item.detection.bottom_center
            previous = self._previous.get(item.track_id)
            self._previous[item.track_id] = current
            if previous is None:
                continue
            if item.track_id in self.counted_track_ids:
                continue
            direction = crossing_direction(self.counting_line, previous, current)
            if direction is None:
                continue
            self.counted_track_ids.add(item.track_id)
            events.append(
                Crossing(
                    track_id=item.track_id,
                    class_id=class_ids.get(item.track_id, item.detection.class_id),
                    raw_class_id=item.raw_class_id,
                    direction=direction,
                    confidence=item.detection.confidence,
                    norm_area=norm_areas.get(item.track_id, 0),
                    anchor_x=current[0],
                    anchor_y=current[1],
                    frame_index=frame_index,
                    video_pts_ms=video_pts_ms,
                )
            )
        lost = [track_id for track_id in self._previous if track_id not in live]
        for track_id in lost:
            del self._previous[track_id]
        return events
