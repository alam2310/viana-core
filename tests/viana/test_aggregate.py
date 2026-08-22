"""Phase 2 — 15-minute aggregation from events CSV (ADR 001)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from viana.config.classes import load_class_taxonomy
from viana.io.checkpoint import Checkpoint, save_checkpoint, utc_now_iso
from viana.io.csv_schema import RawCrossingEventRow
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


def test_zero_fill_includes_pedestrian() -> None:
    """Clock window includes all aggregatable classes, including Pedestrian."""
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
        _event(
            track_id=4,
            class_name="Pedestrian",
            class_id=11,
            direction="in",
            wall_time="2026-03-15T09:10:00Z",
        ),
        _event(
            track_id=5,
            class_name="Pedestrian",
            class_id=11,
            direction="in",
            wall_time="2026-03-15T09:11:00Z",
        ),
    ]
    rows = build_aggregate_rows(events, taxonomy)
    names = {row.class_name for row in rows}
    assert "Pedestrian" in names
    assert "Car" in names
    assert "Jeep" in names
    car_in = next(row for row in rows if row.class_name == "Car" and row.direction == "in")
    jeep_in = next(row for row in rows if row.class_name == "Jeep" and row.direction == "in")
    ped_in = next(row for row in rows if row.class_name == "Pedestrian" and row.direction == "in")
    ped_out = next(row for row in rows if row.class_name == "Pedestrian" and row.direction == "out")
    assert car_in.count == 2
    assert jeep_in.count == 0
    assert ped_in.count == 2
    assert ped_out.count == 1
    assert car_in.window_start == "09:00"
    assert car_in.window_end == "09:15"
    assert car_in.date == "15-03-2026"
    assert car_in.category == "Passenger"
    assert car_in.class_type == "Light Fast"
    windows = {row.window_start for row in rows}
    assert windows == {"09:00"}
    aggregatable = len(taxonomy.aggregatable())
    assert len(rows) == aggregatable * 2
    # Counts match raw Pedestrian events (2 in + 1 out).
    ped_events = [e for e in events if e.class_name == "Pedestrian"]
    assert sum(1 for e in ped_events if e.direction == "in") == ped_in.count
    assert sum(1 for e in ped_events if e.direction == "out") == ped_out.count


def test_spans_two_clock_windows() -> None:
    """Events in 09:07 and 09:20 produce two 15-minute bins."""
    taxonomy = load_class_taxonomy()
    events = [
        _event(wall_time="2026-03-15T09:07:00Z"),
        _event(track_id=2, wall_time="2026-03-15T09:20:00Z", direction="out"),
    ]
    rows = build_aggregate_rows(events, taxonomy)
    starts = sorted({row.window_start for row in rows})
    assert starts == ["09:00", "09:15"]


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
    header = out_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("Date,Window Start,Window End")
