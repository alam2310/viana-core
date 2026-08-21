"""Video frame iteration for the process loop (OpenCV when available)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from viana.io.media import apply_container_timing
from viana.stages.prescan import VideoMeta


def open_cv2_capture(source: Path) -> Any:
    """Open ``source`` with OpenCV; always ``release()`` the return value."""
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to open video") from exc
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {source}")
    return capture


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
    capture = open_cv2_capture(source)
    try:
        import cv2
    except ImportError as exc:
        capture.release()
        raise RuntimeError("opencv-python is required to open video for viana run") from exc
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 15.0
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        fps, frame_count, duration_sec = apply_container_timing(
            source, fps=fps, frame_count=frame_count, duration_sec=duration_sec
        )
        if width < 1 or height < 1:
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
        frames = _Cv2FrameIterator(
            capture,
            start_index=start_index,
            width=width,
            height=height,
            fps=fps,
        )
        return meta, frames
    except BaseException:
        capture.release()
        raise


class _Cv2FrameIterator:
    """Close/release VideoCapture even if the caller stops early or errors."""

    def __init__(
        self,
        capture: Any,
        *,
        start_index: int,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self._capture = capture
        self._index = start_index
        self._width = width
        self._height = height
        self._fps = fps
        self._closed = False

    def __iter__(self) -> _Cv2FrameIterator:
        return self

    def __next__(self) -> VideoFrame:
        if self._closed:
            raise StopIteration
        try:
            import cv2
        except ImportError:
            self.close()
            raise
        ok, image = self._capture.read()
        if not ok:
            self.close()
            raise StopIteration
        pts = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
        if pts <= 0 and self._fps > 0:
            pts = self._index * (1000.0 / self._fps)
        frame = VideoFrame(
            index=self._index,
            pts_ms=pts,
            width=self._width,
            height=self._height,
            image=image,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        """Release the underlying ``VideoCapture`` (idempotent)."""
        if self._closed:
            return
        self._closed = True
        release = getattr(self._capture, "release", None)
        if callable(release):
            release()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass
