"""CSV column contracts from ``events_*.schema.json`` (no aggregation here)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from viana.config.files import contracts_schemas_dir

EVENTS_RAW_SCHEMA = "events_raw.schema.json"
EVENTS_REPORT_SCHEMA = "events_report.schema.json"
EVENTS_15MIN_SCHEMA = "events_15min.schema.json"

WallTimeSource = Literal["ocr_recalibrated", "ocr_anchor", "user_fallback", "unavailable"]
CrossingDirection = Literal["in", "out"]

# Human-readable CSV headers (property order in events_report / events_15min schemas).
EVENTS_REPORT_CSV_HEADERS: tuple[str, ...] = (
    "Video Filename",
    "Date",
    "Time",
    "Location",
    "Class Id",
    "Class Name",
    "Category",
    "Class Type",
    "Direction",
)

EVENTS_15MIN_CSV_HEADERS: tuple[str, ...] = (
    "Date",
    "Window Start",
    "Window End",
    "Location",
    "Class Name",
    "Category",
    "Class Type",
    "Direction",
    "Count",
)

# Map schema property keys → CSV header labels for reporting exports.
EVENTS_REPORT_FIELD_TO_HEADER: dict[str, str] = {
    "video_filename": "Video Filename",
    "date": "Date",
    "time": "Time",
    "location": "Location",
    "class_id": "Class Id",
    "class_name": "Class Name",
    "category": "Category",
    "class_type": "Class Type",
    "direction": "Direction",
}

EVENTS_15MIN_FIELD_TO_HEADER: dict[str, str] = {
    "date": "Date",
    "window_start": "Window Start",
    "window_end": "Window End",
    "location": "Location",
    "class_name": "Class Name",
    "category": "Category",
    "class_type": "Class Type",
    "direction": "Direction",
    "count": "Count",
}


def load_json_schema(filename: str) -> dict[str, Any]:
    """Load a contract JSON schema by filename."""
    path = contracts_schemas_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def csv_columns_from_schema(schema: dict[str, Any]) -> tuple[str, ...]:
    """Return CSV column names in schema ``properties`` order."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("schema has no properties")
    return tuple(properties.keys())


@cache
def events_raw_columns() -> tuple[str, ...]:
    """Column order for ``{stem}_events.csv`` (debug)."""
    return csv_columns_from_schema(load_json_schema(EVENTS_RAW_SCHEMA))


@cache
def events_report_columns() -> tuple[str, ...]:
    """Property keys for ``{stem}_events_report.csv`` rows."""
    return csv_columns_from_schema(load_json_schema(EVENTS_REPORT_SCHEMA))


@cache
def events_15min_columns() -> tuple[str, ...]:
    """Property keys for ``{stem}_15min.csv`` rows (CSV uses human headers)."""
    return csv_columns_from_schema(load_json_schema(EVENTS_15MIN_SCHEMA))


def validate_csv_header(header: Sequence[str], expected: Sequence[str]) -> None:
    """Require an exact header match (names and order) against the contract."""
    actual = tuple(header)
    want = tuple(expected)
    if actual != want:
        raise ValueError(f"CSV columns {list(actual)!r} do not match schema {list(want)!r}")


class RawCrossingEventRow(BaseModel):
    """One row in ``{stem}_events.csv`` (``events_raw.schema.json``)."""

    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    job_id: str
    video_file: str
    track_id: int
    frame_index: int
    video_pts_ms: float
    class_name: str
    direction: CrossingDirection
    confidence: float
    wall_time: str | None = None
    wall_time_source: WallTimeSource | None = None
    ocr_confidence: float | None = None
    date: str | None = None
    location: str | None = None
    class_id: int | None = None
    raw_class_id: int | None = None
    raw_class_name: str | None = None
    category: str | None = None
    class_type: str | None = None
    sub_class: str | None = None
    norm_area: int | None = None
    anchor_x: float | None = None
    anchor_y: float | None = None


class ReportCrossingEventRow(BaseModel):
    """One row in ``{stem}_events_report.csv`` (``events_report.schema.json``)."""

    model_config = ConfigDict(extra="forbid")

    video_filename: str
    date: str
    time: str
    location: str
    class_id: int
    class_name: str
    category: str
    class_type: str
    direction: CrossingDirection


class Aggregate15MinRow(BaseModel):
    """One row in ``{stem}_15min.csv`` (``events_15min.schema.json``)."""

    model_config = ConfigDict(extra="forbid")

    date: str
    window_start: str
    window_end: str
    location: str | None = None
    class_name: str
    category: str
    class_type: str
    direction: CrossingDirection
    count: int = Field(ge=0)
