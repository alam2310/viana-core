"""Tests for events report CSV export."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from viana.io.csv_schema import EVENTS_REPORT_CSV_HEADERS, RawCrossingEventRow
from viana.io.events import EventsCsvWriter
from viana.io.events_report import raw_to_report_row, wall_time_to_hms, write_events_report_csv


def _event(**overrides: object) -> RawCrossingEventRow:
    payload: dict[str, object] = {
        "event_id": uuid4(),
        "job_id": "job_1",
        "video_file": "clip.mp4",
        "track_id": 1,
        "frame_index": 10,
        "video_pts_ms": 400.0,
        "class_name": "Car",
        "direction": "in",
        "confidence": 0.9,
        "wall_time": "2026-03-15T09:07:12Z",
        "date": "15-03-2026",
        "location": "NH48",
        "class_id": 0,
        "category": "Passenger",
        "class_type": "Light Fast",
    }
    payload.update(overrides)
    return RawCrossingEventRow.model_validate(payload)


def test_wall_time_to_hms() -> None:
    assert wall_time_to_hms("2026-03-15T09:07:12Z") == "09:07:12"
    assert wall_time_to_hms(None) == ""


def test_write_events_report_csv(tmp_path: Path) -> None:
    events_path = tmp_path / "clip_events.csv"
    report_path = tmp_path / "clip_events_report.csv"
    with EventsCsvWriter(events_path) as writer:
        writer.write_row(_event())
    count = write_events_report_csv(events_path, report_path)
    assert count == 1
    lines = report_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(EVENTS_REPORT_CSV_HEADERS)
    assert "clip.mp4" in lines[1]
    assert "15-03-2026" in lines[1]
    assert "09:07:12" in lines[1]
    assert "Passenger" in lines[1]


def test_raw_to_report_row() -> None:
    row = raw_to_report_row(_event())
    assert row.video_filename == "clip.mp4"
    assert row.time == "09:07:12"
    assert row.category == "Passenger"
