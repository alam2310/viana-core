"""CLI aggregate command (Phase 2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from viana.cli import app
from viana.io.csv_schema import RawCrossingEventRow
from viana.io.events import EventsCsvWriter

runner = CliRunner()


def test_cli_aggregate_writes_15min_csv(tmp_path: Path) -> None:
    """viana aggregate reads events and writes a 15-min CSV."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    events_path = tmp_path / "clip_events.csv"
    row = RawCrossingEventRow(
        event_id=uuid4(),
        job_id="job_1",
        video_file="clip.mp4",
        track_id=1,
        frame_index=1,
        video_pts_ms=40.0,
        class_name="Car",
        direction="in",
        confidence=0.8,
        wall_time="2026-03-15T09:01:00Z",
        class_id=0,
    )
    with EventsCsvWriter(events_path) as writer:
        writer.write_row(row)
    result = runner.invoke(
        app,
        [
            "aggregate",
            "--source",
            str(video),
            "--project-id",
            "nh48",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["rows"] > 0
    out_path = tmp_path / "clip_15min.csv"
    assert out_path.is_file()
    with out_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == (
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
        first = next(reader)
    assert first["Date"] == "15-03-2026"
    assert first["Window Start"] == "09:00"
    assert first["Window End"] == "09:15"


def test_cli_aggregate_missing_events(tmp_path: Path) -> None:
    """Missing events CSV exits 1."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    result = runner.invoke(
        app,
        [
            "aggregate",
            "--source",
            str(video),
            "--project-id",
            "nh48",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.stderr.lower()
