"""Clock-aligned 15-minute aggregation from events CSV (ADR 001)."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from viana.config.classes import ClassTaxonomy
from viana.io.checkpoint import load_checkpoint
from viana.io.csv_schema import (
    Aggregate15MinRow,
    CrossingDirection,
    RawCrossingEventRow,
    events_15min_columns,
)
from viana.io.events import read_events

WINDOW = timedelta(minutes=15)
DIRECTIONS: tuple[CrossingDirection, CrossingDirection] = ("in", "out")


def parse_wall_time(value: str) -> datetime:
    """Parse an event wall_time string as an aware UTC datetime."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def floor_window(moment: datetime) -> datetime:
    """Return the clock-aligned 15-minute window start for ``moment``."""
    utc = moment.astimezone(timezone.utc)
    minute = (utc.minute // 15) * 15
    return utc.replace(minute=minute, second=0, microsecond=0)


def format_window(moment: datetime) -> str:
    """Format a window edge as HH:MM."""
    utc = moment.astimezone(timezone.utc)
    return utc.strftime("%H:%M")


def _assert_run_complete_or_partial(
    *,
    partial: bool,
    checkpoint_path: Path | None,
) -> bool:
    """Return True if the last window should be marked partial.

    Incomplete checkpoints require ``--partial``; aggregation never resumes inference.
    """
    if checkpoint_path is None or not checkpoint_path.is_file():
        return False
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.is_complete():
        return False
    if not partial:
        raise ValueError(
            "Incomplete run (checkpoint current_frame < total_frames). "
            "Pass --partial to aggregate anyway."
        )
    return True


def build_aggregate_rows(
    events: list[RawCrossingEventRow],
    taxonomy: ClassTaxonomy,
    *,
    last_window_partial: bool = False,
) -> list[Aggregate15MinRow]:
    """Zero-fill a class × direction grid for each 15-minute clock window."""
    aggregatable = taxonomy.aggregatable()
    allowed = {item.name for item in aggregatable}
    timed: list[tuple[datetime, RawCrossingEventRow]] = []
    for event in events:
        if event.class_name not in allowed:
            continue
        if not event.wall_time:
            continue
        timed.append((parse_wall_time(event.wall_time), event))

    if not timed:
        return []

    starts = [floor_window(moment) for moment, _event in timed]
    first = min(starts)
    last = max(starts)
    windows: list[datetime] = []
    cursor = first
    while cursor <= last:
        windows.append(cursor)
        cursor += WINDOW

    counts: Counter[tuple[datetime, str, CrossingDirection]] = Counter()
    meta: dict[datetime, tuple[str | None, str | None]] = {}
    for moment, event in timed:
        window = floor_window(moment)
        counts[(window, event.class_name, event.direction)] += 1
        if window not in meta:
            meta[window] = (event.date, event.location)

    class_by_name = {item.name: item for item in aggregatable}
    rows: list[Aggregate15MinRow] = []
    last_start = windows[-1]
    for window_start in windows:
        window_end = window_start + WINDOW
        date, location = meta.get(window_start, (None, None))
        resolved_date = date or window_start.strftime("%d-%m-%Y")
        is_partial = last_window_partial and window_start == last_start
        for vehicle in aggregatable:
            info = class_by_name[vehicle.name]
            for direction in DIRECTIONS:
                rows.append(
                    Aggregate15MinRow(
                        window_start=format_window(window_start),
                        window_end=format_window(window_end),
                        date=resolved_date,
                        class_name=vehicle.name,
                        direction=direction,
                        count=counts[(window_start, vehicle.name, direction)],
                        partial=is_partial,
                        location=location,
                        category=info.category,
                        class_type=info.class_type,
                        sub_class=info.sub_class,
                    )
                )
    return rows


def write_aggregate_csv(path: Path, rows: list[Aggregate15MinRow]) -> None:
    """Write ``{stem}_15min.csv`` in schema column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(events_15min_columns())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            dumped = row.model_dump()
            record: dict[str, str] = {}
            for column in columns:
                value = dumped[column]
                if value is None:
                    record[column] = ""
                elif isinstance(value, bool):
                    record[column] = "true" if value else "false"
                else:
                    record[column] = str(value)
            writer.writerow(record)


def aggregate_events(
    events_path: Path,
    output_path: Path,
    taxonomy: ClassTaxonomy,
    *,
    partial: bool = False,
    checkpoint_path: Path | None = None,
) -> list[Aggregate15MinRow]:
    """Read events CSV, write 15-minute CSV, return rows.

    Does not run detection. Incomplete jobs need ``partial=True``.
    """
    last_window_partial = _assert_run_complete_or_partial(
        partial=partial, checkpoint_path=checkpoint_path
    )
    events = read_events(events_path)
    rows = build_aggregate_rows(events, taxonomy, last_window_partial=last_window_partial)
    write_aggregate_csv(output_path, rows)
    return rows
