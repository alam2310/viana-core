"""Read and write ``{stem}_events.csv`` using the events_raw column contract."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from viana.io.csv_schema import (
    RawCrossingEventRow,
    WallTimeSource,
    events_raw_columns,
    validate_csv_header,
)

_WALL_TIME_SOURCES = {
    "ocr_recalibrated",
    "ocr_anchor",
    "user_fallback",
    "unavailable",
}


def _cell(value: object) -> str:
    """Serialize a Pydantic field to a CSV cell (empty string for None)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, UUID):
        return str(value)
    return str(value)


def event_to_csv_dict(row: RawCrossingEventRow) -> dict[str, str]:
    """Map an event row to schema-ordered CSV cells."""
    dumped = row.model_dump()
    return {column: _cell(dumped[column]) for column in events_raw_columns()}


def _parse_optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)


def _parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def csv_dict_to_event(record: dict[str, str]) -> RawCrossingEventRow:
    """Parse one CSV record into a validated event row."""
    wall_source = record.get("wall_time_source") or ""
    direction = record["direction"]
    if direction not in ("in", "out"):
        raise ValueError(f"invalid direction: {direction}")
    source: WallTimeSource | None
    if wall_source == "":
        source = None
    elif wall_source in _WALL_TIME_SOURCES:
        source = cast(WallTimeSource, wall_source)
    else:
        raise ValueError(f"invalid wall_time_source: {wall_source}")
    payload: dict[str, Any] = {
        "event_id": UUID(record["event_id"]),
        "job_id": record["job_id"],
        "video_file": record["video_file"],
        "track_id": int(record["track_id"]),
        "frame_index": int(record["frame_index"]),
        "video_pts_ms": float(record["video_pts_ms"]),
        "class_name": record["class_name"],
        "direction": direction,
        "confidence": float(record["confidence"]),
        "wall_time": record.get("wall_time") or None,
        "wall_time_source": source,
        "ocr_confidence": _parse_optional_float(record.get("ocr_confidence", "")),
        "date": record.get("date") or None,
        "location": record.get("location") or None,
        "class_id": _parse_optional_int(record.get("class_id", "")),
        "raw_class_id": _parse_optional_int(record.get("raw_class_id", "")),
        "raw_class_name": record.get("raw_class_name") or None,
        "category": record.get("category") or None,
        "class_type": record.get("class_type") or None,
        "sub_class": record.get("sub_class") or None,
        "norm_area": _parse_optional_int(record.get("norm_area", "")),
        "anchor_x": _parse_optional_float(record.get("anchor_x", "")),
        "anchor_y": _parse_optional_float(record.get("anchor_y", "")),
    }
    return RawCrossingEventRow.model_validate(payload)


class EventsCsvWriter:
    """Append-only events CSV writer. Does not compute 15-minute bins."""

    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not (append and self.path.is_file() and self.path.stat().st_size > 0)
        if not new_file:
            with self.path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            validate_csv_header(header, events_raw_columns())
            self._handle = self.path.open("a", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=list(events_raw_columns()))
        else:
            self._handle = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=list(events_raw_columns()))
            self._writer.writeheader()
            self._handle.flush()

    def write_row(self, row: RawCrossingEventRow) -> None:
        """Append one crossing event (buffered)."""
        self._writer.writerow(event_to_csv_dict(row))

    def write_rows(self, rows: Iterable[RawCrossingEventRow]) -> None:
        """Append many crossing events."""
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        """Flush the underlying file handle to disk."""
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()

    def __enter__(self) -> EventsCsvWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_events(path: Path) -> list[RawCrossingEventRow]:
    """Load and validate every row in an events CSV."""
    return list(iter_events(path))


def iter_events(path: Path) -> Iterator[RawCrossingEventRow]:
    """Yield validated event rows from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Events CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Events CSV has no header: {path}")
        validate_csv_header(reader.fieldnames, events_raw_columns())
        for record in reader:
            yield csv_dict_to_event(record)
