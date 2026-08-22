"""Phase 2 — events CSV writer/reader."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from viana.io.csv_schema import RawCrossingEventRow, events_raw_columns
from viana.io.events import EventsCsvWriter, read_events


def _event(**overrides: object) -> RawCrossingEventRow:
    payload: dict[str, object] = {
        "event_id": uuid4(),
        "job_id": "job_1",
        "video_file": "clip.mp4",
        "track_id": 7,
        "frame_index": 42,
        "video_pts_ms": 1680.0,
        "class_name": "Car",
        "direction": "in",
        "confidence": 0.91,
        "wall_time": "2026-03-15T09:07:12Z",
        "class_id": 0,
    }
    payload.update(overrides)
    return RawCrossingEventRow.model_validate(payload)


def test_events_csv_roundtrip(tmp_path: Path) -> None:
    """Written rows reread with schema column order."""
    path = tmp_path / "clip_events.csv"
    first = _event()
    second = _event(track_id=8, direction="out", class_name="Bus", class_id=6)
    with EventsCsvWriter(path) as writer:
        writer.write_rows([first, second])
    rows = read_events(path)
    assert len(rows) == 2
    assert rows[0].event_id == first.event_id
    assert rows[1].direction == "out"
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == events_raw_columns()


def test_events_csv_append_keeps_single_header(tmp_path: Path) -> None:
    """Resume-style append does not rewrite the header."""
    path = tmp_path / "clip_events.csv"
    with EventsCsvWriter(path) as writer:
        writer.write_row(_event())
    with EventsCsvWriter(path, append=True) as writer:
        writer.write_row(_event(track_id=9))
    text = path.read_text(encoding="utf-8")
    assert text.count("event_id") == 1
    assert len(read_events(path)) == 2


def test_read_events_missing_file(tmp_path: Path) -> None:
    """Missing events CSV fails closed."""
    with pytest.raises(FileNotFoundError):
        read_events(tmp_path / "missing_events.csv")
