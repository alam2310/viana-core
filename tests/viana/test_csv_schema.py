"""Phase 1 — CSV columns match events_*.schema.json."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from viana.config.files import contracts_schemas_dir
from viana.io.csv_schema import (
    EVENTS_15MIN_SCHEMA,
    EVENTS_RAW_SCHEMA,
    Aggregate15MinRow,
    RawCrossingEventRow,
    csv_columns_from_schema,
    events_15min_columns,
    events_raw_columns,
    load_json_schema,
    validate_csv_header,
)


def test_raw_event_fields_match_schema() -> None:
    """Pydantic event row fields equal events_raw.schema.json properties."""
    schema = load_json_schema(EVENTS_RAW_SCHEMA)
    assert set(RawCrossingEventRow.model_fields) == set(schema["properties"])
    assert events_raw_columns() == csv_columns_from_schema(schema)
    assert (contracts_schemas_dir() / EVENTS_RAW_SCHEMA).is_file()


def test_aggregate_fields_match_schema() -> None:
    """Pydantic 15-min row fields equal events_15min.schema.json properties."""
    schema = load_json_schema(EVENTS_15MIN_SCHEMA)
    assert set(Aggregate15MinRow.model_fields) == set(schema["properties"])
    assert events_15min_columns() == csv_columns_from_schema(schema)


def test_validate_csv_header_accepts_schema_order() -> None:
    """Exact schema property order is the CSV header contract."""
    validate_csv_header(events_raw_columns(), events_raw_columns())
    validate_csv_header(events_15min_columns(), events_15min_columns())


def test_validate_csv_header_rejects_mismatch() -> None:
    """Missing or reordered columns are invalid."""
    cols = list(events_raw_columns())
    cols[0], cols[1] = cols[1], cols[0]
    with pytest.raises(ValueError, match="do not match schema"):
        validate_csv_header(cols, events_raw_columns())


def test_raw_event_row_required_fields() -> None:
    """A minimal required-column event row validates."""
    row = RawCrossingEventRow(
        event_id=uuid4(),
        job_id="job_mock_001",
        video_file="clip.mp4",
        track_id=1,
        frame_index=10,
        video_pts_ms=400.0,
        class_name="Car",
        direction="in",
        confidence=0.9,
    )
    assert row.direction == "in"


def test_raw_event_rejects_unknown_column() -> None:
    """Invented CSV fields are forbidden."""
    with pytest.raises(ValidationError):
        RawCrossingEventRow.model_validate(
            {
                "event_id": str(uuid4()),
                "job_id": "j",
                "video_file": "v.mp4",
                "track_id": 1,
                "frame_index": 0,
                "video_pts_ms": 0,
                "class_name": "Car",
                "direction": "out",
                "confidence": 0.8,
                "invented": True,
            }
        )


def test_aggregate_row_count_non_negative() -> None:
    """15-min counts cannot be negative."""
    with pytest.raises(ValidationError):
        Aggregate15MinRow(
            window_start="2026-03-15T09:00:00Z",
            window_end="2026-03-15T09:15:00Z",
            class_name="Car",
            direction="in",
            count=-1,
            partial=False,
        )
