"""Phase 3 — tracking and once-per-track counting-line crossings."""

from __future__ import annotations

import pytest

from viana.config.defaults import load_engine_defaults
from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.stages.crossing import CrossingState
from viana.stages.cv_core import FrameCVEngine
from viana.stages.track import ByteTrackTracker, IoUTracker, TrackedDetection


def test_iou_tracker_stable_id() -> None:
    """Overlapping boxes across frames keep the same track id."""
    tracker = IoUTracker()
    a = Detection(x1=10, y1=10, x2=30, y2=30, confidence=0.9, class_id=0)
    b = Detection(x1=12, y1=12, x2=32, y2=32, confidence=0.9, class_id=0)
    first = tracker.update([a], 0)
    second = tracker.update([b], 1)
    assert first[0].track_id == second[0].track_id == 1


def test_byte_track_keeps_id_and_splits_classes() -> None:
    """ByteTrack (if installed) holds overlapping boxes and isolates pedestrians."""
    pytest.importorskip("trackers")
    tracker = ByteTrackTracker(frame_rate=15)
    ids: list[int] = []
    for index in range(8):
        box = Detection(
            x1=10 + index,
            y1=10 + index,
            x2=40 + index,
            y2=40 + index,
            confidence=0.9,
            class_id=0,
        )
        tracked = tracker.update([box], index)
        if tracked:
            ids.append(tracked[0].track_id)
    assert ids
    assert len(set(ids)) == 1
    person = Detection(x1=12, y1=12, x2=38, y2=38, confidence=0.9, class_id=11)
    people = tracker.update([person], 20)
    assert people
    assert people[0].track_id != ids[0]
    assert people[0].track_id >= 1_000_000


def test_iou_tracker_does_not_match_different_classes() -> None:
    """A pedestrian box must not inherit a vehicle track id by overlap."""
    tracker = IoUTracker()
    car = Detection(x1=10, y1=10, x2=40, y2=40, confidence=0.9, class_id=0)
    person = Detection(x1=12, y1=12, x2=38, y2=38, confidence=0.9, class_id=11)
    first = tracker.update([car], 0)
    second = tracker.update([person], 1)
    assert first[0].track_id != second[0].track_id


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


def test_crossing_survives_brief_detection_gap() -> None:
    """hiv00013-style miss: class flicker / conf dip drops the box for 1–2 frames.

    If the bottom-center reappears already on the far side of the counting line,
    the last pre-gap anchor must still produce one event (once-per-track).
    """
    line = LineSegment(start=(0, 100), end=(200, 100))
    state = CrossingState(counting_line=line, max_gap_frames=15)
    before = TrackedDetection(
        track_id=3,
        detection=Detection(x1=40, y1=120, x2=60, y2=160, confidence=0.8, class_id=0),
        raw_class_id=0,
    )
    after = TrackedDetection(
        track_id=3,
        detection=Detection(x1=42, y1=20, x2=62, y2=70, confidence=0.85, class_id=1),
        raw_class_id=1,
    )
    assert (
        state.update(
            [before], class_ids={3: 0}, norm_areas={3: 10}, frame_index=10, video_pts_ms=400
        )
        == []
    )
    # Two empty frames while the jeep straddles / passes the line (shimoga ~06:44:50).
    assert state.update([], class_ids={}, norm_areas={}, frame_index=11, video_pts_ms=440) == []
    assert state.update([], class_ids={}, norm_areas={}, frame_index=12, video_pts_ms=480) == []
    recovered = state.update(
        [after], class_ids={3: 1}, norm_areas={3: 12}, frame_index=13, video_pts_ms=520
    )
    assert len(recovered) == 1
    assert recovered[0].track_id == 3
    assert recovered[0].class_id == 1
    assert recovered[0].direction in ("in", "out")
    # Still once-per-track after recovery.
    assert (
        state.update(
            [after], class_ids={3: 1}, norm_areas={3: 12}, frame_index=14, video_pts_ms=560
        )
        == []
    )


def test_crossing_forgets_anchor_after_max_gap() -> None:
    """Long disappearances still reset the previous anchor (no false late counts)."""
    line = LineSegment(start=(0, 100), end=(200, 100))
    state = CrossingState(counting_line=line, max_gap_frames=2)
    before = TrackedDetection(
        track_id=9,
        detection=Detection(x1=40, y1=120, x2=60, y2=160, confidence=0.9, class_id=0),
        raw_class_id=0,
    )
    after = TrackedDetection(
        track_id=9,
        detection=Detection(x1=40, y1=20, x2=60, y2=60, confidence=0.9, class_id=0),
        raw_class_id=0,
    )
    state.update([before], class_ids={9: 0}, norm_areas={9: 1}, frame_index=1, video_pts_ms=40)
    for index in (2, 3, 4):
        state.update([], class_ids={}, norm_areas={}, frame_index=index, video_pts_ms=index * 40)
    late = state.update(
        [after], class_ids={9: 0}, norm_areas={9: 1}, frame_index=5, video_pts_ms=200
    )
    assert late == []


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
    locked = result.class_ids[result.crossings[0].track_id]
    flipped = Detection(x1=42, y1=0, x2=82, y2=90, confidence=0.9, class_id=1)
    later = engine.process_detections([flipped], frame_index=2, video_pts_ms=80)
    assert later.class_ids[result.crossings[0].track_id] == locked
