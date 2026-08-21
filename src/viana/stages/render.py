"""Optional annotated-video writer (FFmpeg or OpenCV). Not used for 15-min CSV."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Protocol

from viana.config.job import LineSegment
from viana.io.proc import close_stdio, run_captured, terminate_process_tree
from viana.stages.cv_core import FrameCVResult
from viana.stages.video import VideoFrame

# BGR overlay colors by ITVA class id (not horizon red / counting green).
OVERLAY_BGR: dict[int, tuple[int, int, int]] = {
    0: (255, 144, 30),  # Car — dodger blue
    1: (0, 140, 255),  # Jeep — dark orange
    2: (255, 0, 255),  # Van — magenta
    3: (180, 105, 255),  # MiniBus — pink
    4: (0, 255, 255),  # MTW — yellow
    5: (203, 192, 255),  # Auto — plum
    6: (0, 215, 255),  # Bus — gold
    7: (42, 42, 165),  # Heavy Truck — brown
    8: (19, 69, 139),  # LCV — saddle
    9: (255, 255, 0),  # Cycle — cyan
    10: (160, 160, 160),  # Other — gray
    11: (255, 0, 127),  # Pedestrian — violet
    12: (80, 127, 255),  # MCV — coral
    13: (130, 0, 75),  # Trailer — indigo
    14: (0, 200, 255),  # Taxi — amber
}
_DEFAULT_OVERLAY_BGR = (220, 220, 220)


def overlay_bgr(class_id: int) -> tuple[int, int, int]:
    """Return the processed-video box color for a taxonomy class id."""
    return OVERLAY_BGR.get(class_id, _DEFAULT_OVERLAY_BGR)


# Fragmented MP4 (S13): empty moov + moof/mdat so growing files are streamable.
_FRAG_MP4_ARGS = [
    "-frag_duration",
    "1000000",
    "-movflags",
    "+frag_keyframe+empty_moov+default_base_moof",
]

# H.264 for browser <video> (S20). Prefer NVENC when listed, else libx264.
# HEVC is smaller but Chrome/Firefox on Linux cannot decode hev1/hvc1 in <video>.
_H264_NVENC_ARGS = [
    "-c:v",
    "h264_nvenc",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "p7",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "28",
    "-b:v",
    "0",
    "-g",
    "30",
    "-keyint_min",
    "30",
    "-forced-idr",
    "1",
    *_FRAG_MP4_ARGS,
]

_HEVC_NVENC_ARGS = [
    "-c:v",
    "hevc_nvenc",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "p7",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "42",
    "-b:v",
    "0",
    "-bf",
    "4",
    "-spatial-aq",
    "1",
    "-temporal-aq",
    "1",
    "-g",
    "30",
    "-keyint_min",
    "30",
    "-forced-idr",
    "1",
    *_FRAG_MP4_ARGS,
]


def ffmpeg_video_args(path: Path, *, encoder_list: str) -> list[str]:
    """Pick a browser-playable review encode, then size-oriented HEVC fallbacks.

    Preference (S20): ``h264_nvenc`` → ``libx264`` → ``hevc_nvenc`` → ``libx265``.
    All paths keep fragmented MP4 flags (S13) for in-progress streaming.
    """
    out = str(path)
    if "h264_nvenc" in encoder_list:
        return [*_H264_NVENC_ARGS, out]
    if "libx264" in encoder_list:
        return [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "30",
            "-preset",
            "medium",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            *_FRAG_MP4_ARGS,
            out,
        ]
    if "hevc_nvenc" in encoder_list:
        return [*_HEVC_NVENC_ARGS, out]
    if "libx265" in encoder_list:
        return [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "34",
            "-preset",
            "medium",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            *_FRAG_MP4_ARGS,
            out,
        ]
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "30",
        "-preset",
        "medium",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        *_FRAG_MP4_ARGS,
        out,
    ]


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
    class_names: dict[int, str] | None = None,
) -> Any:
    """Draw calibration lines, track boxes, and class names when OpenCV is available."""
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
    names = class_names or {}
    for item in result.tracked:
        box = item.detection
        class_id = result.class_ids.get(item.track_id, item.raw_class_id)
        name = names.get(class_id, f"class_{class_id}")
        label = f"{name} #{item.track_id}"
        color = overlay_bgr(class_id)
        x1, y1 = int(box.x1), int(box.y1)
        cv2.rectangle(
            canvas,
            (x1, y1),
            (int(box.x2), int(box.y2)),
            color,
            2,
        )
        text_y = y1 - 6 if y1 > 16 else y1 + 16
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return canvas


class FfmpegRenderer:
    """Pipe BGR frames to FFmpeg (H.264 NVENC when available; fragmented MP4)."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        listed = run_captured([ffmpeg, "-hide_banner", "-encoders"], timeout=30.0)
        self._proc = subprocess.Popen(  # noqa: S603  # nosec B603
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
                *ffmpeg_video_args(path, encoder_list=listed.stdout),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._closed = False
        self._horizon: LineSegment | None = None
        self._counting: LineSegment | None = None
        self._class_names: dict[int, str] = {}

    def set_lines(self, horizon: LineSegment, counting_line: LineSegment) -> None:
        """Store overlay geometry."""
        self._horizon = horizon
        self._counting = counting_line

    def set_class_names(self, class_names: dict[int, str]) -> None:
        """YOLO id → display name for box labels."""
        self._class_names = dict(class_names)

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Encode one annotated frame."""
        if frame.image is None or self._proc.stdin is None:
            return
        image: Any = frame.image
        if self._horizon is not None and self._counting is not None:
            image = annotate_bgr(
                image,
                result,
                self._horizon,
                self._counting,
                class_names=self._class_names,
            )
        self._proc.stdin.write(image.tobytes())

    def close(self) -> None:
        """Flush ffmpeg stdin, wait, and kill the process group if it hangs."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self._proc
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, close_pipes=True)
            return
        close_stdio(proc)


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
