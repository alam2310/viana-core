"""Merge vehicle + pedestrian detections (legacy DetectionEngine.predict without YOLO)."""

from __future__ import annotations

from viana.domain.boxes import Detection, ioa_child_in_parent, nms_class_agnostic

PEDESTRIAN_ID = 11
# ITVA vehicle-head ids (legacy target_classes). Pedestrian is model-B only.
VEHICLE_CLASS_IDS = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14})
COCO_PERSON_ID = 0


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
    """Combine model-A vehicles and model-B persons, then per-pool NMS.

    Pedestrian YOLO **person** (COCO 0) is remapped to ``pedestrian_id``. Other
    model-B classes (car, bicycle, …) are dropped so they cannot inherit the
    passenger/slow/pedestrian taxonomy row. Vehicle boxes outside the ITVA id
    set are dropped. Persons overlapping a vehicle above ``suppression_ioa``
    are dropped. Confidence is applied after merge. NMS runs separately on
    vehicles and people so a roadside pedestrian is not suppressed by a
    nearby vehicle box.
    """
    kept_vehicles = [item for item in vehicles if item.class_id in VEHICLE_CLASS_IDS]
    mapped_people: list[Detection] = []
    for person in pedestrians:
        if person.class_id != COCO_PERSON_ID:
            continue
        remapped = Detection(
            x1=person.x1,
            y1=person.y1,
            x2=person.x2,
            y2=person.y2,
            confidence=person.confidence,
            class_id=pedestrian_id,
        )
        if pedestrian_inside_vehicle(remapped, kept_vehicles, suppression_ioa):
            continue
        mapped_people.append(remapped)

    kept_vehicles = [item for item in kept_vehicles if item.confidence >= confidence_threshold]
    mapped_people = [item for item in mapped_people if item.confidence >= confidence_threshold]
    if not kept_vehicles and not mapped_people:
        return []
    # NMS within each pool only. Class-agnostic NMS across vehicles+people would
    # drop roadside pedestrians whose boxes overlap a nearby vehicle (IoU≥thr)
    # even when they are not "inside" by the IOA passenger test.
    return [
        *nms_class_agnostic(kept_vehicles, nms_threshold),
        *nms_class_agnostic(mapped_people, nms_threshold),
    ]
