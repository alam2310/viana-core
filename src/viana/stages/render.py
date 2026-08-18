"""Optional annotated-video writer (FFmpeg or OpenCV). Not used for 15-min CSV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

from viana.config.job import LineSegment
from viana.stages.cv_core import FrameCVResult
from viana.stages.video import VideoFrame


class FrameRenderer(Protocol):
    """Write annotated frames; implementations must be close() safe."""

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Encode or record one processed frame."""

    def close(self) -> None:
        """Flush and release the sink."""


class NullRenderer:
    """No-op renderer when ``render_video`` is false."""

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Ignore the frame."""
        _ = (frame, result)

    def close(self) -> None:
        """Nothing to flush."""


def annotate_bgr(
    image: Any,
    result: FrameCVResult,
    horizon: LineSegment,
    counting_line: LineSegment,
) -> Any:
    """Draw calibration lines and track boxes when OpenCV is available."""
    try:
        import cv2
    except ImportError:
        return image
    canvas = image.copy()
    cv2.line(
        canvas,
        (int(horizon.start[0]), int(horizon.start[1])),
        (int(horizon.end[0]), int(horizon.end[1])),
        (0, 0, 255),
        2,
    )
    cv2.line(
        canvas,
        (int(counting_line.start[0]), int(counting_line.start[1])),
        (int(counting_line.end[0]), int(counting_line.end[1])),
        (0, 255, 0),
        2,
    )
    for item in result.tracked:
        box = item.detection
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            (255, 180, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"#{item.track_id}",
            (int(box.x1), max(0, int(box.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0),
            1,
        )
    return canvas


class FfmpegRenderer:
    """Pipe BGR frames to ``ffmpeg`` H.264 (legacy replaced huge cv2 AVI dumps)."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(  # noqa: S603
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps if fps > 0 else 25),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self._horizon: LineSegment | None = None
        self._counting: LineSegment | None = None

    def set_lines(self, horizon: LineSegment, counting_line: LineSegment) -> None:
        """Store overlay geometry."""
        self._horizon = horizon
        self._counting = counting_line

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Encode one annotated frame."""
        if frame.image is None or self._proc.stdin is None:
            return
        image: Any = frame.image
        if self._horizon is not None and self._counting is not None:
            image = annotate_bgr(image, result, self._horizon, self._counting)
        self._proc.stdin.write(image.tobytes())

    def close(self) -> None:
        """Flush ffmpeg stdin and wait."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


class RecordingRenderer:
    """Test helper that records write counts."""

    def __init__(self) -> None:
        self.frames: list[int] = []

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Record the frame index."""
        _ = result
        self.frames.append(frame.index)

    def close(self) -> None:
        """Nothing to flush."""
