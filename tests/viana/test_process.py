"""Phase 5 — process loop writes events, checkpoints, and run_result (no 15-min)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from viana.config.classes import load_class_taxonomy
from viana.config.job import JobConfig, JobMetadata, LineSegment, ViAnaTaskParameters
from viana.domain.boxes import Detection
from viana.io.checkpoint import load_checkpoint
from viana.io.events import read_events
from viana.io.paths import artifact_paths, legacy_artifact_paths
from viana.stages.crossing import Crossing
from viana.stages.prescan import VideoMeta
from viana.stages.process import CheckpointExistsError, crossing_to_event, run_moving_count
from viana.stages.render import RecordingRenderer
from viana.stages.time_map import TimeMap, load_time_map, time_map_from_metadata
from viana.stages.track import IoUTracker
from viana.stages.video import VideoFrame


@pytest.fixture(autouse=True)
def _stable_iou_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic 2-frame clips are too short for ByteTrack confirmation (-1 then 0)."""
    monkeypatch.setattr(
        "viana.stages.process.build_tracker",
        lambda frame_rate=30.0: IoUTracker(),
    )


def _job(tmp_path: Path, video: Path) -> JobConfig:
    return JobConfig(
        source_video_path=video,
        project_id="nh48",
        metadata=JobMetadata(
            user_start_time="09:00:00",
            user_start_date="15-03-2026",
            location="NH48 Km42",
        ),
        task_parameters=ViAnaTaskParameters(
            horizon_line=LineSegment(start=(10, 40), end=(190, 50)),
            counting_line=LineSegment(start=(10, 120), end=(190, 120)),
            render_video=False,
            telemetry_detail=True,
        ),
        job_id="job_test_001",
        gpu_device="cuda:0",
        output_dir=tmp_path,
    )


def _job_no_detail(tmp_path: Path, video: Path) -> JobConfig:
    job = _job(tmp_path, video)
    return job.model_copy(
        update={
            "task_parameters": job.task_parameters.model_copy(update={"telemetry_detail": False})
        }
    )


def _meta() -> VideoMeta:
    return VideoMeta(width=200, height=200, fps=25.0, duration_sec=0.12, frame_count=3)


def _frames() -> list[VideoFrame]:
    return [
        VideoFrame(index=0, pts_ms=0.0, width=200, height=200),
        VideoFrame(index=1, pts_ms=40.0, width=200, height=200),
        VideoFrame(index=2, pts_ms=80.0, width=200, height=200),
    ]


def _detect(frame: VideoFrame) -> tuple[list[Detection], list[Detection]]:
    if frame.index == 0:
        box = Detection(20, 90, 80, 115, 0.91, 0)
        return [box], []
    if frame.index == 1:
        box = Detection(22, 95, 82, 140, 0.92, 0)
        return [box], []
    return [], []


def test_run_writes_events_checkpoint_and_run_result(tmp_path: Path) -> None:
    """A two-frame crossing becomes one events row; 15-min CSV is not written."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    telemetry: list[str] = []
    renderer = RecordingRenderer()
    result = run_moving_count(
        job,
        resume=False,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=renderer,
        emit=lambda msg: telemetry.append(msg.telemetry_type),
    )
    assert result.status == "COMPLETED"
    paths = artifact_paths(tmp_path, "clip")
    events = read_events(paths["events"])
    assert len(events) == 1
    assert events[0].class_name == "Car"
    assert events[0].class_id == 0
    assert events[0].direction == "in"
    assert events[0].wall_time_source == "user_fallback"
    assert not paths["aggregate_15min"].exists()
    checkpoint = load_checkpoint(paths["checkpoint"])
    assert checkpoint.is_complete()
    assert checkpoint.events_rows_written == 1
    assert paths["run_result"].is_file()
    assert renderer.frames == [0, 1, 2]
    assert "PROGRESS" in telemetry
    assert "MOVING_EVENT" in telemetry


def test_supplied_frames_do_not_shrink_inflated_total(tmp_path: Path) -> None:
    """Injected iterators (tests) keep header ``total_frames``; live decode may shrink."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    inflated = VideoMeta(width=200, height=200, fps=25.0, duration_sec=40.0, frame_count=1000)
    run_moving_count(
        job,
        resume=False,
        frames=(inflated, _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    checkpoint = load_checkpoint(artifact_paths(tmp_path, "clip")["checkpoint"])
    assert checkpoint.total_frames == 1000
    assert checkpoint.current_frame == 3


def test_moving_event_emitted_without_telemetry_detail(tmp_path: Path) -> None:
    """Crossing events are emitted even when telemetry_detail is false (S14)."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job_no_detail(tmp_path, video)
    telemetry: list[dict[str, object]] = []
    run_moving_count(
        job,
        resume=False,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda msg: telemetry.append({"type": msg.telemetry_type, "data": msg.data}),
    )
    events = [item for item in telemetry if item["type"] == "MOVING_EVENT"]
    assert events
    payload = events[0]["data"]
    assert payload["event_timestamp"] is not None
    assert payload["event_timestamp_source"] in {"ocr_anchor", "user_fallback"}
    assert isinstance(payload["video_pts_ms"], int | float)
    assert "event_timestamp_confidence" in payload


def test_run_refuses_silent_resume(tmp_path: Path) -> None:
    """A second viana run without start_fresh must not continue the checkpoint."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    run_moving_count(
        job,
        resume=False,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    with pytest.raises(CheckpointExistsError):
        run_moving_count(
            job,
            resume=False,
            frames=(_meta(), _frames()),
            detector=_detect,
            renderer=RecordingRenderer(),
            emit=lambda _msg: None,
        )


def test_resume_skips_already_processed_frames(tmp_path: Path) -> None:
    """Explicit resume continues from checkpoint.current_frame without duplicate events."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    run_moving_count(
        job,
        resume=False,
        frames=(_meta(), _frames()[:2]),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    paths = artifact_paths(tmp_path, "clip")
    first = load_checkpoint(paths["checkpoint"])
    assert first.current_frame == 2
    assert not first.is_complete()
    assert len(read_events(paths["events"])) == 1

    job_resume = job.model_copy(update={"resume": True})
    result = run_moving_count(
        job_resume,
        resume=True,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    assert result.status == "COMPLETED"
    assert len(read_events(paths["events"])) == 1
    assert load_checkpoint(paths["checkpoint"]).is_complete()


def test_resume_finds_legacy_flat_checkpoint(tmp_path: Path) -> None:
    """Pre-S29 ``{stem}.checkpoint.json`` must still resume (6.2 / ADR 003)."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    # Partial progress written the old flat way; events already on disk.
    paths = artifact_paths(tmp_path, "clip")
    run_moving_count(
        job,
        resume=False,
        frames=(_meta(), _frames()[:2]),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    # Move checkpoint back to legacy flat path to simulate pre-layout tree.
    canonical = paths["checkpoint"]
    legacy = legacy_artifact_paths(tmp_path, "clip")["checkpoint"]
    legacy.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
    canonical.unlink()
    assert not canonical.is_file()
    assert legacy.is_file()

    job_resume = job.model_copy(update={"resume": True})
    result = run_moving_count(
        job_resume,
        resume=True,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    assert result.status == "COMPLETED"
    assert paths["checkpoint"].is_file()
    assert load_checkpoint(paths["checkpoint"]).is_complete()
    assert not legacy.is_file()  # migrated off legacy after first progress write


def test_cli_run_prints_run_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """viana run prints RunResult JSON when the process loop completes."""
    from typer.testing import CliRunner

    from viana.cli import app
    from viana.io.run_result import RunResultArtifacts, completed_now

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    job = _job(tmp_path, video)
    config = tmp_path / "job.json"
    config.write_text(job.model_dump_json(), encoding="utf-8")

    def fake_run(loaded: JobConfig, *, resume: bool) -> object:
        assert resume is False
        assert loaded.job_id == "job_test_001"
        return completed_now(
            loaded.job_id,
            loaded.source_video_path,
            loaded.source_video_path.stem,
            RunResultArtifacts(events=str(tmp_path / "clip_events.csv")),
        )

    monkeypatch.setattr("viana.stages.process.run_moving_count", fake_run)
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "COMPLETED"
    assert payload["job_id"] == "job_test_001"


def test_crossing_uses_classes_yaml_name(tmp_path: Path) -> None:
    """class_name / class_id come from classes.yaml; hierarchy is not a CSV column (S32)."""
    job = _job(tmp_path, tmp_path / "clip.mp4")
    crossing = Crossing(
        track_id=1,
        class_id=6,
        raw_class_id=7,
        direction="out",
        confidence=0.8,
        norm_area=1000,
        anchor_x=10.0,
        anchor_y=20.0,
        frame_index=3,
        video_pts_ms=120.0,
    )
    row = crossing_to_event(
        job,
        load_class_taxonomy(),
        "clip.mp4",
        crossing,
        TimeMap(job_id=job.job_id, video_stem="clip"),
    )
    assert row.class_name == "Bus"
    assert row.class_id == 6
    dumped = row.model_dump()
    assert "category" not in dumped
    assert "raw_class_name" not in dumped
    assert "anchor_x" not in dumped


def test_crossing_interpolates_confirmed_metadata_clock(tmp_path: Path) -> None:
    """Events CSV uses the confirmed prescan/user clock, not mid-run OSD (I003)."""
    job = _job(tmp_path, tmp_path / "clip.mp4")
    time_map = time_map_from_metadata(job.job_id, "clip", job.metadata)
    crossing = Crossing(
        track_id=1,
        class_id=0,
        raw_class_id=0,
        direction="in",
        confidence=0.9,
        norm_area=1000,
        anchor_x=10.0,
        anchor_y=20.0,
        frame_index=100,
        video_pts_ms=90_000.0,
    )
    row = crossing_to_event(job, load_class_taxonomy(), "clip.mp4", crossing, time_map)
    assert row.wall_time == "2026-03-15T09:01:30Z"
    assert row.wall_time_source == "user_fallback"
    assert row.date == "15-03-2026"
    assert row.location == "NH48 Km42"


def test_process_does_not_init_easyocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Process loop must not construct an OSD reader (prescan-only OCR)."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("process must not init EasyOCR")

    monkeypatch.setattr("viana.stages.ocr.optional_easyocr_reader", boom)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    result = run_moving_count(
        _job(tmp_path, video),
        resume=False,
        frames=(_meta(), _frames()),
        detector=_detect,
        renderer=RecordingRenderer(),
        emit=lambda _msg: None,
    )
    assert result.status == "COMPLETED"
    paths = artifact_paths(tmp_path, "clip")
    time_map = load_time_map(paths["time_map"])
    assert len(time_map.anchors) == 1
    assert time_map.anchors[0].source == "user_fallback"
    assert all(anchor.source != "ocr_recalibrated" for anchor in time_map.anchors)


class _CloseableFeed:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[VideoFrame]:
        return iter(_frames())

    def close(self) -> None:
        self.closed = True


def test_invalid_geometry_releases_owned_frame_feed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S22: VideoCapture iterator must close if geometry validation fails."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    feed = _CloseableFeed()
    monkeypatch.setattr(
        "viana.stages.process._default_frames",
        lambda _source, start_index=0: (_meta(), feed),
    )
    job = _job(tmp_path, video)
    job = job.model_copy(
        update={
            "task_parameters": job.task_parameters.model_copy(
                update={
                    "horizon_line": LineSegment(start=(0, 0), end=(10, 10_000)),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="outside"):
        run_moving_count(
            job,
            resume=False,
            detector=_detect,
            renderer=RecordingRenderer(),
            emit=lambda _msg: None,
        )
    assert feed.closed
