"""Phase 5 — process loop writes events, checkpoints, and run_result (no 15-min)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viana.config.job import JobConfig, JobMetadata, LineSegment, ViAnaTaskParameters
from viana.domain.boxes import Detection
from viana.io.checkpoint import load_checkpoint
from viana.io.events import read_events
from viana.io.paths import artifact_paths
from viana.stages.prescan import VideoMeta
from viana.stages.process import CheckpointExistsError, run_moving_count
from viana.stages.render import RecordingRenderer
from viana.stages.video import VideoFrame


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
        ocr_reader=lambda _frame: [],
    )
    assert result.status == "COMPLETED"
    paths = artifact_paths(tmp_path, "clip")
    events = read_events(paths["events"])
    assert len(events) == 1
    assert events[0].class_name == "Car"
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
        ocr_reader=lambda _frame: [],
    )
    with pytest.raises(CheckpointExistsError):
        run_moving_count(
            job,
            resume=False,
            frames=(_meta(), _frames()),
            detector=_detect,
            renderer=RecordingRenderer(),
            emit=lambda _msg: None,
            ocr_reader=lambda _frame: [],
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
        ocr_reader=lambda _frame: [],
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
        ocr_reader=lambda _frame: [],
    )
    assert result.status == "COMPLETED"
    assert len(read_events(paths["events"])) == 1
    assert load_checkpoint(paths["checkpoint"]).is_complete()


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

    monkeypatch.setattr("viana.cli.run_moving_count", fake_run)
    result = CliRunner().invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "COMPLETED"
    assert payload["job_id"] == "job_test_001"
