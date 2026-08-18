"""Box trackers: ByteTrack when supervision is available, else greedy IoU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from viana.domain.boxes import Detection, iou
from viana.stages.detect import PEDESTRIAN_ID

# Pedestrian ByteTrack ids are offset so they never collide with vehicle ids.
_PEDESTRIAN_ID_OFFSET = 1_000_000


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection bound to a stable track id for this frame."""

    track_id: int
    detection: Detection
    raw_class_id: int


class BoxTracker(Protocol):
    """Assign integer track ids to detections for one frame."""

    def update(self, detections: list[Detection], frame_index: int) -> list[TrackedDetection]:
        """Match detections to existing tracks; spawn ids for unmatched boxes."""


class IoUTracker:
    """Assign integer track ids by greedy IoU matching (CPU, no supervision).

    Matches vehicles to vehicles (class flips allowed) and people to people so a
    person box cannot steal a vehicle id. Used when ByteTrack is not installed.
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, tuple[Detection, int]] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[TrackedDetection]:
        """Match detections to existing tracks; spawn ids for unmatched boxes."""
        assigned: list[TrackedDetection] = []
        used_tracks: set[int] = set()
        matched_det_ids: set[int] = set()

        pairs: list[tuple[float, float, int, Detection]] = []
        for detection in detections:
            for track_id, (previous, _age) in self._tracks.items():
                person_mismatch = (detection.class_id == PEDESTRIAN_ID) != (
                    previous.class_id == PEDESTRIAN_ID
                )
                if person_mismatch:
                    continue
                score = iou(detection, previous)
                if score < self.iou_threshold:
                    continue
                same = 1.0 if detection.class_id == previous.class_id else 0.0
                pairs.append((same, score, track_id, detection))
        pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)

        for _same, _score, track_id, detection in pairs:
            det_key = id(detection)
            if track_id in used_tracks or det_key in matched_det_ids:
                continue
            used_tracks.add(track_id)
            matched_det_ids.add(det_key)
            self._tracks[track_id] = (detection, 0)
            assigned.append(
                TrackedDetection(
                    track_id=track_id, detection=detection, raw_class_id=detection.class_id
                )
            )

        for detection in detections:
            if id(detection) in matched_det_ids:
                continue
            track_id = self._next_id
            self._next_id += 1
            used_tracks.add(track_id)
            self._tracks[track_id] = (detection, 0)
            assigned.append(
                TrackedDetection(
                    track_id=track_id, detection=detection, raw_class_id=detection.class_id
                )
            )

        for track_id in list(self._tracks):
            if track_id in used_tracks:
                continue
            detection, age = self._tracks[track_id]
            if age + 1 > self.max_age:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = (detection, age + 1)

        _ = frame_index
        return assigned


def _detections_to_sv(detections: list[Detection]) -> Any:
    import numpy as np
    import supervision as sv

    if not detections:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(
            [[item.x1, item.y1, item.x2, item.y2] for item in detections],
            dtype=np.float32,
        ),
        confidence=np.array([item.confidence for item in detections], dtype=np.float32),
        class_id=np.array([item.class_id for item in detections], dtype=int),
    )


def _sv_update(tracker: Any, detections: Any) -> Any:
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(detections)
    return tracker.update(detections)


def _make_byte_track(frame_rate: float) -> Any:
    """Roboflow ``ByteTrackTracker`` (replaces deprecated ``supervision.ByteTrack``)."""
    from trackers import ByteTrackTracker as Backend

    fps = max(1, int(round(frame_rate)))
    return Backend(frame_rate=fps, lost_track_buffer=60)


class ByteTrackTracker:
    """Roboflow ByteTrackTracker with separate vehicle and pedestrian pools.

    Same motion model as legacy, but a person box cannot inherit a vehicle id.
    """

    def __init__(self, *, frame_rate: float = 30.0) -> None:
        self._vehicles = _make_byte_track(frame_rate)
        self._people = _make_byte_track(frame_rate)

    def update(self, detections: list[Detection], frame_index: int) -> list[TrackedDetection]:
        """Run ByteTrack on vehicles and pedestrians separately."""
        vehicles = [item for item in detections if item.class_id != PEDESTRIAN_ID]
        people = [item for item in detections if item.class_id == PEDESTRIAN_ID]
        assigned = self._run_pool(self._vehicles, vehicles, id_offset=0)
        assigned.extend(self._run_pool(self._people, people, id_offset=_PEDESTRIAN_ID_OFFSET))
        _ = frame_index
        return assigned

    def _run_pool(
        self, tracker: Any, detections: list[Detection], *, id_offset: int
    ) -> list[TrackedDetection]:
        result = _sv_update(tracker, _detections_to_sv(detections))
        if result.tracker_id is None or len(result) == 0:
            return []
        assigned: list[TrackedDetection] = []
        xyxy = result.xyxy
        class_ids = result.class_id
        confs = result.confidence
        for index, raw_tid in enumerate(result.tracker_id):
            if raw_tid is None:
                continue
            track_id = int(raw_tid) + id_offset
            box = xyxy[index]
            class_id = int(class_ids[index]) if class_ids is not None else detections[0].class_id
            confidence = float(confs[index]) if confs is not None else 1.0
            assigned.append(
                TrackedDetection(
                    track_id=track_id,
                    detection=Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=confidence,
                        class_id=class_id,
                    ),
                    raw_class_id=class_id,
                )
            )
        return assigned


def build_tracker(*, frame_rate: float = 30.0) -> BoxTracker:
    """Prefer ``trackers.ByteTrackTracker``; fall back to IoU when it is missing."""
    try:
        import trackers
    except ImportError:
        return IoUTracker()
    _ = trackers
    return ByteTrackTracker(frame_rate=frame_rate)
