"""Phase 2 — checkpoint JSON I/O (explicit resume only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from viana.config.files import repo_root
from viana.io.checkpoint import Checkpoint, load_checkpoint, save_checkpoint, utc_now_iso


def test_load_committed_checkpoint_fixture() -> None:
    """Fixture matches checkpoint.schema.json required fields."""
    path = repo_root() / "packages" / "contracts" / "fixtures" / "checkpoint_resume.json"
    checkpoint = load_checkpoint(path)
    assert checkpoint.schema_version == 1
    assert checkpoint.job_id == "job_mock_paused"
    assert checkpoint.current_frame == 18420
    assert checkpoint.is_complete() is False


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    """Round-trip checkpoint JSON."""
    path = tmp_path / "clip.checkpoint.json"
    original = Checkpoint(
        job_id="job_1",
        project_id="nh48",
        source_video_path="/data/v.mp4",
        video_stem="v",
        current_frame=10,
        total_frames=10,
        saved_at=utc_now_iso(),
        counted_track_ids=[1, 2],
        events_rows_written=3,
    )
    save_checkpoint(path, original)
    loaded = load_checkpoint(path)
    assert loaded.is_complete() is True
    assert loaded.counted_track_ids == [1, 2]
    assert loaded.events_rows_written == 3


def test_checkpoint_rejects_frame_past_total() -> None:
    """current_frame cannot exceed total_frames."""
    with pytest.raises(ValidationError):
        Checkpoint(
            job_id="job_1",
            project_id="nh48",
            source_video_path="/data/v.mp4",
            video_stem="v",
            current_frame=11,
            total_frames=10,
            saved_at=utc_now_iso(),
        )


def test_load_checkpoint_invalid_json_type(tmp_path: Path) -> None:
    """ValueError is raised if JSON root is not a dict."""
    path = tmp_path / "invalid.checkpoint.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match=f"Expected a JSON object in {path}"):
        load_checkpoint(path)
