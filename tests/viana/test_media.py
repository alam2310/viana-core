"""S19 — container duration/fps/frame_count correction."""

from __future__ import annotations

from pathlib import Path

import pytest

from viana.io.media import (
    MIN_PLAUSIBLE_BITRATE_BPS,
    apply_container_timing,
    header_duration_is_plausible,
    implied_bitrate_bps,
    parse_frame_rate,
)


def test_parse_frame_rate_fraction() -> None:
    assert parse_frame_rate("15/1", 25.0) == 15.0
    assert parse_frame_rate("30000/1001", 25.0) == 30000 / 1001
    assert parse_frame_rate("0/0", 15.0) == 15.0
    assert parse_frame_rate(None, 15.0) == 15.0


def test_header_bitrate_rejects_dvr_clock_span() -> None:
    """256 MiB over 21.2 h is ~28 kbps — not 1080p video."""
    size = 268_435_456
    bogus_duration = 76_240.39
    assert implied_bitrate_bps(size, bogus_duration) is not None
    assert implied_bitrate_bps(size, bogus_duration) < MIN_PLAUSIBLE_BITRATE_BPS
    assert not header_duration_is_plausible(size, bogus_duration)
    assert header_duration_is_plausible(size, 2_281.07)


def test_apply_container_timing_recounts_packets_when_bitrate_absurd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "hiv.mp4"
    clip.write_bytes(b"stub")

    def fake_streams(_source: Path) -> dict[str, object]:
        return {
            "streams": [{"avg_frame_rate": "15/1", "r_frame_rate": "15/1", "nb_frames": None}],
            "format": {
                "duration": "76240.392822",
                "size": str(268_435_456),
                "format_name": "mpeg",
            },
        }

    monkeypatch.setattr("viana.io.media._ffprobe_streams", fake_streams)
    monkeypatch.setattr("viana.io.media._ffprobe_packet_count", lambda _source: 34_197)

    fps, frames, duration = apply_container_timing(
        clip, fps=15.0, frame_count=1_143_606, duration_sec=76_240.4
    )
    assert fps == 15.0
    assert frames == 34_197
    assert abs(duration - (34_197 / 15.0)) < 0.01
    assert duration < 2_300


def test_apply_container_timing_keeps_plausible_mp4_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "inframe.mp4"
    clip.write_bytes(b"stub")

    def fake_streams(_source: Path) -> dict[str, object]:
        return {
            "streams": [
                {
                    "avg_frame_rate": "15/1",
                    "r_frame_rate": "15/1",
                    "nb_frames": "2701",
                    "duration": "180.066667",
                }
            ],
            "format": {
                "duration": "180.07",
                "size": "52400000",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            },
        }

    monkeypatch.setattr("viana.io.media._ffprobe_streams", fake_streams)
    monkeypatch.setattr(
        "viana.io.media._ffprobe_packet_count",
        lambda _source: (_ for _ in ()).throw(AssertionError("must not recount")),
    )

    fps, frames, duration = apply_container_timing(
        clip, fps=15.0, frame_count=2701, duration_sec=180.07
    )
    assert fps == 15.0
    assert frames == 2701
    assert abs(duration - 180.07) < 0.01
