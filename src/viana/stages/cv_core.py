"""Per-frame CV core: merge → horizon filter → track → classify → crossing."""

from __future__ import annotations

from dataclasses import dataclass, field

from viana.config.defaults import ClassificationDefaults, DetectionDefaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.domain.geometry import filter_below_horizon
from viana.stages.classify import ClassificationEngine
from viana.stages.crossing import Crossing, CrossingState
from viana.stages.detect import merge_detections
from viana.stages.track import IoUTracker, TrackedDetection


@dataclass
class FrameCVResult:
    """Outputs of one processed frame (events only; no 15-min aggregation)."""

    tracked: list[TrackedDetection]
    crossings: list[Crossing]
    class_ids: dict[int, int]
    norm_areas: dict[int, int]


@dataclass
class FrameCVEngine:
    """Stateful CV stages shared by the Phase 5 process loop."""

    horizon: LineSegment
    counting_line: LineSegment
    frame_height: int
    detection: DetectionDefaults
    classification: ClassificationDefaults
    tracker: IoUTracker = field(default_factory=IoUTracker)
    classifier: ClassificationEngine = field(init=False)
    crossings: CrossingState = field(init=False)

    def __post_init__(self) -> None:
        self.classifier = ClassificationEngine(self.classification, self.horizon, self.frame_height)
        self.crossings = CrossingState(counting_line=self.counting_line)

    def process_models(
        self,
        vehicles: list[Detection],
        pedestrians: list[Detection],
        *,
        frame_index: int,
        video_pts_ms: float,
    ) -> FrameCVResult:
        """Merge dual-model boxes then run the tracked crossing pipeline."""
        merged = merge_detections(
            vehicles,
            pedestrians,
            suppression_ioa=self.detection.suppression_ioa,
            nms_threshold=self.detection.nms_threshold,
            confidence_threshold=self.detection.confidence_threshold,
        )
        return self.process_detections(merged, frame_index=frame_index, video_pts_ms=video_pts_ms)

    def process_detections(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        video_pts_ms: float,
    ) -> FrameCVResult:
        """Filter, track, classify, and emit unique line crossings."""
        filtered = filter_below_horizon(detections, self.horizon)
        tracked = self.tracker.update(filtered, frame_index)
        class_ids: dict[int, int] = {}
        norm_areas: dict[int, int] = {}
        for item in tracked:
            final_id, area = self.classifier.process_vehicle(
                item.track_id, item.raw_class_id, item.detection
            )
            class_ids[item.track_id] = final_id
            norm_areas[item.track_id] = area
        crossings = self.crossings.update(
            tracked,
            class_ids=class_ids,
            norm_areas=norm_areas,
            frame_index=frame_index,
            video_pts_ms=video_pts_ms,
        )
        return FrameCVResult(
            tracked=tracked,
            crossings=crossings,
            class_ids=class_ids,
            norm_areas=norm_areas,
        )
