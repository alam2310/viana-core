"""Video frame iteration for the process loop (OpenCV when available)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from viana.stages.prescan import VideoMeta


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One sampled frame. ``image`` may be None in CPU tests."""

    index: int
    pts_ms: float
    width: int
    height: int
    image: object | None = None


def iter_cv2_frames(
    source: Path, *, start_index: int = 0
) -> tuple[VideoMeta, Iterator[VideoFrame]]:
    """Open a video and yield frames from ``start_index`` (0-based)."""
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to open video for viana run") from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if width < 1 or height < 1:
        capture.release()
        raise ValueError(f"Invalid frame size {width}x{height} in {source}")
    meta = VideoMeta(
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
        frame_count=frame_count if frame_count > 0 else 1,
    )
    if start_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_index))

    def _iter() -> Iterator[VideoFrame]:
        index = start_index
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if pts <= 0 and fps > 0:
                    pts = index * (1000.0 / fps)
                yield VideoFrame(index=index, pts_ms=pts, width=width, height=height, image=image)
                index += 1
        finally:
            capture.release()

    return meta, _iter()
