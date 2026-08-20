"""Container timing probe (duration / fps / frame count).

OpenCV ``CAP_PROP_FRAME_COUNT`` and MPEG-PS (Hikvision ``.mp4``) headers often
report a huge duration with an implausibly low bitrate. Queue video length and
ETA then inflate because they divide that frame count by processing fps.

Source of truth after this module:

- ``fps`` — stream ``avg_frame_rate`` / ``r_frame_rate``, else OpenCV, else 15
- ``duration_sec`` — container duration when implied bitrate is plausible;
  otherwise ``packet_count / fps``
- ``frame_count`` — ``nb_frames`` when present; otherwise ``round(duration * fps)``
  or demuxed packet count when the header was rejected
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

# 1080p DVR clips at ~1 Mbps are common; header durations that imply tens of
# kbps (256 MiB / 21 h) are not real playback length.
MIN_PLAUSIBLE_BITRATE_BPS = 80_000.0
_FFPROBE_TIMEOUT_SEC = 120.0


def parse_frame_rate(rate: str | None, fallback: float) -> float:
    """Parse ffprobe ``15/1`` or ``30000/1001`` rates."""
    if rate is None:
        return fallback
    text = str(rate).strip()
    if not text or text in {"0/0", "N/A"}:
        return fallback
    try:
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            den = float(den_s)
            return float(num_s) / den if den else fallback
        value = float(text)
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def implied_bitrate_bps(size_bytes: int, duration_sec: float) -> float | None:
    """Return bits/sec from file size and duration, or None if duration is unusable."""
    if duration_sec <= 0 or size_bytes <= 0:
        return None
    return (size_bytes * 8.0) / duration_sec


def header_duration_is_plausible(size_bytes: int, duration_sec: float) -> bool:
    """True when size/duration looks like encoded video, not a DVR clock span."""
    bitrate = implied_bitrate_bps(size_bytes, duration_sec)
    return bitrate is not None and bitrate >= MIN_PLAUSIBLE_BITRATE_BPS


def apply_container_timing(
    source: Path,
    *,
    fps: float,
    frame_count: int,
    duration_sec: float,
) -> tuple[float, int, float]:
    """Correct OpenCV timing using ffprobe when the header bitrate is absurd.

    Returns ``(fps, frame_count, duration_sec)``. OpenCV values are kept when
    ffprobe is missing or fails.
    """
    fps_val = fps if fps > 0 else 15.0
    frames = max(frame_count, 0)
    duration = duration_sec if duration_sec > 0 else 0.0
    if duration <= 0 and fps_val > 0 and frames > 0:
        duration = frames / fps_val

    payload = _ffprobe_streams(source)
    if payload is None:
        return fps_val, frames if frames > 0 else 1, duration

    stream = _first_video_stream(payload)
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    size_bytes = _int_or_zero(fmt.get("size"))
    if size_bytes <= 0:
        try:
            size_bytes = source.stat().st_size
        except OSError:
            size_bytes = 0

    probe_fps = parse_frame_rate(
        stream.get("avg_frame_rate") if stream else None,
        parse_frame_rate(stream.get("r_frame_rate") if stream else None, fps_val),
    )
    if probe_fps > 0:
        fps_val = probe_fps

    header_duration = _float_or_none(fmt.get("duration"))
    if header_duration is None and stream is not None:
        header_duration = _float_or_none(stream.get("duration"))
    if header_duration is not None and header_duration > 0:
        duration = header_duration

    nb_frames = _int_or_zero(stream.get("nb_frames") if stream else None)
    if nb_frames > 0:
        frames = nb_frames
        header_bad = duration <= 0 or not header_duration_is_plausible(size_bytes, duration)
        if fps_val > 0 and header_bad:
            duration = frames / fps_val

    if not header_duration_is_plausible(size_bytes, duration):
        packets = _ffprobe_packet_count(source)
        if packets is not None and packets > 0 and fps_val > 0:
            frames = packets
            duration = packets / fps_val
    elif frames <= 0 and fps_val > 0 and duration > 0:
        frames = max(int(round(duration * fps_val)), 1)

    if frames <= 0:
        frames = 1
    if duration <= 0 and fps_val > 0:
        duration = frames / fps_val
    return fps_val, frames, duration


def _ffprobe_streams(source: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    return _run_ffprobe_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name,width,height",
            "-show_entries",
            "format=duration,size,bit_rate,format_name",
            "-of",
            "json",
            str(source),
        ]
    )


def _ffprobe_packet_count(source: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    payload = _run_ffprobe_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(source),
        ]
    )
    if payload is None:
        return None
    stream = _first_video_stream(payload)
    if stream is None:
        return None
    count = _int_or_zero(stream.get("nb_read_packets"))
    return count if count > 0 else None


def _run_ffprobe_json(args: list[str]) -> dict[str, Any] | None:
    try:
        listed = subprocess.run(  # noqa: S603  # nosec B603
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode != 0 or not listed.stdout.strip():
        return None
    try:
        payload = json.loads(listed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _first_video_stream(payload: dict[str, Any]) -> dict[str, Any] | None:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    first = streams[0]
    return first if isinstance(first, dict) else None


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str) and value.strip() and value.strip() not in {"N/A", "0"}:
        try:
            parsed = int(float(value))
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if float(value) > 0 else None
    if isinstance(value, str) and value.strip() and value.strip() != "N/A":
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None
