"""Operator reporting export derived from debug ``{stem}_events.csv``."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from viana.io.csv_schema import (
    EVENTS_REPORT_CSV_HEADERS,
    EVENTS_REPORT_FIELD_TO_HEADER,
    RawCrossingEventRow,
    ReportCrossingEventRow,
    events_report_columns,
)
from viana.io.events import iter_events


def wall_time_to_hms(wall_time: str | None) -> str:
    """Format ISO wall_time as ``HH:MM:SS`` (UTC). Empty when unavailable."""
    if not wall_time or not wall_time.strip():
        return ""
    raw = wall_time.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%H:%M:%S")


def raw_to_report_row(row: RawCrossingEventRow) -> ReportCrossingEventRow:
    """Map a debug event row to the operator reporting schema."""
    return ReportCrossingEventRow(
        video_filename=row.video_file,
        date=row.date or "",
        time=wall_time_to_hms(row.wall_time),
        location=row.location or "",
        class_id=row.class_id if row.class_id is not None else -1,
        class_name=row.class_name,
        category=row.category or "",
        class_type=row.class_type or "",
        direction=row.direction,
    )


def write_events_report_csv(events_path: Path, report_path: Path) -> int:
    """Build ``{stem}_events_report.csv`` from the debug events file. Returns row count."""
    if not events_path.is_file():
        raise FileNotFoundError(f"Events CSV not found: {events_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVENTS_REPORT_CSV_HEADERS))
        writer.writeheader()
        for raw in iter_events(events_path):
            report = raw_to_report_row(raw)
            dumped = report.model_dump()
            record = {
                EVENTS_REPORT_FIELD_TO_HEADER[key]: str(dumped[key])
                for key in events_report_columns()
            }
            writer.writerow(record)
            rows_written += 1
    return rows_written
