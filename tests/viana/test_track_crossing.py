"""Phase 3 — tracking and once-per-track counting-line crossings."""

from __future__ import annotations

from viana.config.defaults import load_engine_defaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.stages.crossing import CrossingState
from viana.stages.cv_core import FrameCVEngine
from viana.stages.track import IoUTracker, TrackedDetection


def test_iou_tracker_stable_id() -> None:
    """Overlapping boxes across frames keep the same track id."""
    tracker = IoUTracker()
    a = Detection(x1=10, y1=10, x2=30, y2=30, confidence=0.9, class_id=0)
    b = Detection(x1=12, y1=12, x2=32, y2=32, confidence=0.9, class_id=0)
    first = tracker.update([a], 0)
    second = tracker.update([b], 1)
    assert first[0].track_id == second[0].track_id == 1


def test_crossing_emits_once_per_track() -> None:
    """A track that crosses the counting line yields a single event."""
    line = LineSegment(start=(0, 100), end=(200, 100))
    state = CrossingState(counting_line=line)
    det_below = Detection(x1=40, y1=120, x2=60, y2=160, confidence=0.9, class_id=0)
    det_above = Detection(x1=40, y1=20, x2=60, y2=60, confidence=0.9, class_id=0)
    tracked_below = TrackedDetection(track_id=1, detection=det_below, raw_class_id=0)
    tracked_above = TrackedDetection(track_id=1, detection=det_above, raw_class_id=0)
    first = state.update(
        [tracked_below], class_ids={1: 0}, norm_areas={1: 10}, frame_index=1, video_pts_ms=40
    )
    second = state.update(
        [tracked_above], class_ids={1: 0}, norm_areas={1: 10}, frame_index=2, video_pts_ms=80
    )
    third = state.update(
        [tracked_above], class_ids={1: 0}, norm_areas={1: 10}, frame_index=3, video_pts_ms=120
    )
    assert first == []
    assert len(second) == 1
    assert second[0].direction in ("in", "out")
    assert second[0].track_id == 1
    assert third == []
    assert 1 in state.counted_track_ids


def test_frame_engine_counts_crossing_without_15min() -> None:
    """FrameCVEngine emits crossings only; it does not bin 15-minute counts."""
    defaults = load_engine_defaults()
    engine = FrameCVEngine(
        horizon=LineSegment(start=(0, 20), end=(200, 20)),
        counting_line=LineSegment(start=(0, 100), end=(200, 100)),
        frame_height=200,
        detection=defaults.detection,
        classification=defaults.classification,
        tracker=IoUTracker(iou_threshold=0.05),
    )
    below = Detection(x1=40, y1=70, x2=80, y2=150, confidence=0.9, class_id=0)
    above = Detection(x1=42, y1=0, x2=82, y2=90, confidence=0.9, class_id=0)
    engine.process_detections([below], frame_index=0, video_pts_ms=0)
    result = engine.process_detections([above], frame_index=1, video_pts_ms=40)
    assert len(result.crossings) == 1
    assert not hasattr(result, "bins")
