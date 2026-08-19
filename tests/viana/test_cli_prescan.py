"""CLI prescan command (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from viana.cli import app
from viana.stages.prescan import SampledVideo, VideoMeta

runner = CliRunner()


def test_cli_prescan_missing_video(tmp_path: Path) -> None:
    """Missing source files exit 1."""
    result = runner.invoke(
        app,
        [
            "prescan",
            "--source",
            str(tmp_path / "missing.mp4"),
            "--project-id",
            "nh48",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "not found" in result.stderr.lower()


def test_cli_prescan_json_and_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """viana prescan prints PrescanResponse JSON after a successful sample."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    def sampler(_source: Path, offset: float) -> SampledVideo:
        return SampledVideo(
            meta=VideoMeta(
                width=640,
                height=360,
                fps=25.0,
                duration_sec=8.0,
                frame_count=200,
            ),
            frame_offset_sec=offset,
            frame=None,
        )

    monkeypatch.setattr(
        "viana.stages.prescan.sample_opening_frame",
        lambda _source, **_kwargs: sampler(_source, 0.0),
    )
    monkeypatch.setattr("viana.stages.prescan.sample_video_cv2", sampler)
    monkeypatch.setattr("viana.cli.optional_easyocr_reader", lambda: lambda _f: [])
    result = runner.invoke(
        app,
        [
            "prescan",
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
    assert payload["video_meta"]["width"] == 640
    assert payload["proposed_lines"]["horizon_line"]["start"]
    preview = Path(payload["preview_url"])
    assert preview.is_file()
    assert preview.parent.name == "prescan"
