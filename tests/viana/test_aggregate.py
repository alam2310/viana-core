"""Phase 2 — 15-minute aggregation from events CSV (ADR 001)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from viana.config.classes import load_class_taxonomy
from viana.io.checkpoint import Checkpoint, save_checkpoint, utc_now_iso
from viana.io.csv_schema import RawCrossingEventRow, events_15min_columns
from viana.io.events import EventsCsvWriter
from viana.stages.aggregate import aggregate_events, build_aggregate_rows


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
        "wall_time": "2026-03-15T09:07:00Z",
        "date": "15-03-2026",
        "location": "NH48",
        "category": "Passenger",
        "class_id": 0,
    }
    payload.update(overrides)
    return RawCrossingEventRow.model_validate(payload)


def test_zero_fill_and_exclude_pedestrian() -> None:
    """Clock window includes all aggregatable classes; Pedestrian is omitted."""
    taxonomy = load_class_taxonomy()
    events = [
        _event(),
        _event(track_id=2, class_name="Car", direction="in"),
        _event(
            track_id=3,
            class_name="Pedestrian",
            class_id=11,
            direction="out",
            wall_time="2026-03-15T09:08:00Z",
        ),
    ]
    rows = build_aggregate_rows(events, taxonomy)
    names = {row.class_name for row in rows}
    assert "Pedestrian" not in names
    assert "Car" in names
    assert "Jeep" in names
    car_in = next(row for row in rows if row.class_name == "Car" and row.direction == "in")
    jeep_in = next(row for row in rows if row.class_name == "Jeep" and row.direction == "in")
    assert car_in.count == 2
    assert jeep_in.count == 0
    assert car_in.window_start == "2026-03-15T09:00:00Z"
    assert car_in.window_end == "2026-03-15T09:15:00Z"
    assert car_in.partial is False
    windows = {row.window_start for row in rows}
    assert windows == {"2026-03-15T09:00:00Z"}
    aggregatable = len(taxonomy.aggregatable())
    assert len(rows) == aggregatable * 2


def test_spans_two_clock_windows() -> None:
    """Events in 09:07 and 09:20 produce two 15-minute bins."""
    taxonomy = load_class_taxonomy()
    events = [
        _event(wall_time="2026-03-15T09:07:00Z"),
        _event(track_id=2, wall_time="2026-03-15T09:20:00Z", direction="out"),
    ]
    rows = build_aggregate_rows(events, taxonomy)
    starts = sorted({row.window_start for row in rows})
    assert starts == ["2026-03-15T09:00:00Z", "2026-03-15T09:15:00Z"]


def test_incomplete_checkpoint_requires_partial(tmp_path: Path) -> None:
    """Do not aggregate an incomplete run unless --partial is set."""
    events_path = tmp_path / "clip_events.csv"
    out_path = tmp_path / "clip_15min.csv"
    ckpt_path = tmp_path / "clip.checkpoint.json"
    with EventsCsvWriter(events_path) as writer:
        writer.write_row(_event())
    save_checkpoint(
        ckpt_path,
        Checkpoint(
            job_id="job_1",
            project_id="nh48",
            source_video_path="/data/clip.mp4",
            video_stem="clip",
            current_frame=10,
            total_frames=100,
            saved_at=utc_now_iso(),
        ),
    )
    taxonomy = load_class_taxonomy()
    with pytest.raises(ValueError, match="--partial"):
        aggregate_events(events_path, out_path, taxonomy, checkpoint_path=ckpt_path)
    rows = aggregate_events(
        events_path, out_path, taxonomy, partial=True, checkpoint_path=ckpt_path
    )
    assert rows
    assert all(row.partial for row in rows if row.window_start == "2026-03-15T09:00:00Z")
    header = out_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == events_15min_columns()
