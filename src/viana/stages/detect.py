"""Merge vehicle + pedestrian detections (legacy DetectionEngine.predict without YOLO)."""

from __future__ import annotations

from viana.domain.boxes import Detection, ioa_child_in_parent, nms_class_agnostic

PEDESTRIAN_ID = 11


def pedestrian_inside_vehicle(
    person: Detection,
    vehicles: list[Detection],
    suppression_ioa: float,
) -> bool:
    """True when a pedestrian box is mostly inside a vehicle box (legacy IOA test)."""
    return any(ioa_child_in_parent(person, vehicle) > suppression_ioa for vehicle in vehicles)


def merge_detections(
    vehicles: list[Detection],
    pedestrians: list[Detection],
    *,
    suppression_ioa: float,
    nms_threshold: float,
    confidence_threshold: float,
    pedestrian_id: int = PEDESTRIAN_ID,
) -> list[Detection]:
    """Combine model-A vehicles and model-B persons, then class-agnostic NMS.

    Pedestrian YOLO class 0 is remapped to ``pedestrian_id``. Persons overlapping a
    vehicle above ``suppression_ioa`` are dropped. Confidence is applied after merge.
    """
    mapped_people: list[Detection] = []
    for person in pedestrians:
        remapped = Detection(
            x1=person.x1,
            y1=person.y1,
            x2=person.x2,
            y2=person.y2,
            confidence=person.confidence,
            class_id=pedestrian_id,
        )
        if pedestrian_inside_vehicle(remapped, vehicles, suppression_ioa):
            continue
        mapped_people.append(remapped)

    merged = [item for item in vehicles + mapped_people if item.confidence >= confidence_threshold]
    if not merged:
        return []
    return nms_class_agnostic(merged, nms_threshold)
