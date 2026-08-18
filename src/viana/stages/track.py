"""Greedy IoU tracker used until ByteTrack is wired in the Phase 5 GPU loop."""

from __future__ import annotations

from dataclasses import dataclass

from viana.domain.boxes import Detection, iou


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection bound to a stable track id for this frame."""

    track_id: int
    detection: Detection
    raw_class_id: int


class IoUTracker:
    """Assign integer track ids by greedy IoU matching (CPU, no supervision)."""

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

        pairs: list[tuple[float, int, Detection]] = []
        for detection in detections:
            for track_id, (previous, _age) in self._tracks.items():
                score = iou(detection, previous)
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, detection))
        pairs.sort(key=lambda item: item[0], reverse=True)

        for _score, track_id, detection in pairs:
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
