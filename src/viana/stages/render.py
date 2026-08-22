"""Optional annotated-video writer (FFmpeg or OpenCV). Not used for 15-min CSV."""

from __future__ import annotations

import queue
import shutil
import subprocess  # nosec B404
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from viana.config.job import LineSegment
from viana.io.proc import close_stdio, run_captured, terminate_process_tree
from viana.stages.cv_core import FrameCVResult
from viana.stages.video import VideoFrame

# Bounded queue: back-pressures the CV loop when annotate/encode lags.
_RENDER_QUEUE_MAX = 30
_RENDER_PUT_TIMEOUT_SEC = 0.25
_RENDER_CLOSE_DEADLINE_SEC = 60.0

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
# Size/speed knobs (review deliverable, not archival):
# - NVENC: p4 (faster than p7) + higher cq → smaller bitstream
# - libx264: veryfast + higher CRF → smaller + less CPU on the writer thread
_H264_NVENC_ARGS = [
    "-c:v",
    "h264_nvenc",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "p4",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "32",
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
    "p4",
    "-tune",
    "hq",
    "-rc",
    "vbr",
    "-cq",
    "44",
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

_LIBX264_ARGS = [
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "34",
    "-preset",
    "veryfast",
    "-g",
    "30",
    "-keyint_min",
    "30",
    "-sc_threshold",
    "0",
    *_FRAG_MP4_ARGS,
]

_LIBX265_ARGS = [
    "-c:v",
    "libx265",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "36",
    "-preset",
    "veryfast",
    "-g",
    "30",
    "-keyint_min",
    "30",
    "-sc_threshold",
    "0",
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
        return [*_LIBX264_ARGS, out]
    if "hevc_nvenc" in encoder_list:
        return [*_HEVC_NVENC_ARGS, out]
    if "libx265" in encoder_list:
        return [*_LIBX265_ARGS, out]
    return [*_LIBX264_ARGS, out]


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
    """Pipe BGR frames to FFmpeg (H.264 NVENC when available; fragmented MP4).

    Annotation and stdin writes run on a background thread so the detect/track
    loop is not blocked on OpenCV draw + FFmpeg I/O. Frames are copied before
    enqueue so the decoder can reuse its capture buffer safely.
    """

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
        self._queue: queue.Queue[tuple[Any, FrameCVResult | None]] = queue.Queue(
            maxsize=_RENDER_QUEUE_MAX
        )
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="viana-ffmpeg-renderer",
            daemon=True,
        )
        self._thread.start()

    def set_lines(self, horizon: LineSegment, counting_line: LineSegment) -> None:
        """Store overlay geometry."""
        self._horizon = horizon
        self._counting = counting_line

    def set_class_names(self, class_names: dict[int, str]) -> None:
        """YOLO id → display name for box labels."""
        self._class_names = dict(class_names)

    def _writer_loop(self) -> None:
        """Annotate and write queued frames until a shutdown sentinel arrives."""
        while True:
            frame_image, result = self._queue.get()
            if frame_image is None:
                break
            try:
                image: Any = frame_image
                if result is not None and self._horizon is not None and self._counting is not None:
                    image = annotate_bgr(
                        image,
                        result,
                        self._horizon,
                        self._counting,
                        class_names=self._class_names,
                    )
                if self._proc.stdin is not None:
                    self._proc.stdin.write(image.tobytes())
            except OSError:
                break

    def write(self, frame: VideoFrame, result: FrameCVResult) -> None:
        """Enqueue one frame for background annotate + encode."""
        if self._closed or frame.image is None or self._proc.stdin is None:
            return
        # Detach from the capture buffer before the next decoder read.
        src: Any = frame.image
        image = src.copy()
        while True:
            if not self._thread.is_alive():
                raise OSError("Background writer thread died unexpectedly")
            try:
                self._queue.put((image, result), timeout=_RENDER_PUT_TIMEOUT_SEC)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        """Drain the writer thread, then flush ffmpeg stdin and wait."""
        if getattr(self, "_closed", False):
            return
        self._closed = True

        if self._thread.is_alive():
            deadline = time.monotonic() + _RENDER_CLOSE_DEADLINE_SEC
            while self._thread.is_alive() and time.monotonic() < deadline:
                try:
                    self._queue.put((None, None), timeout=_RENDER_PUT_TIMEOUT_SEC)
                    break
                except queue.Full:
                    continue
            remaining = max(0.0, deadline - time.monotonic())
            self._thread.join(timeout=remaining)

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
