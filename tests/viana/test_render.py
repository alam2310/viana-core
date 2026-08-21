"""Annotated video overlay (class names on boxes)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.stages.cv_core import FrameCVResult
from viana.stages.render import FfmpegRenderer, annotate_bgr, ffmpeg_video_args, overlay_bgr
from viana.stages.track import TrackedDetection
from viana.stages.video import VideoFrame


def test_annotate_bgr_writes_class_name_above_box() -> None:
    """Processed-video overlay labels the classified vehicle name."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("numpy")
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    tracked = TrackedDetection(
        track_id=7,
        detection=Detection(x1=20, y1=40, x2=80, y2=90, confidence=0.9, class_id=4),
        raw_class_id=4,
    )
    result = FrameCVResult(
        tracked=[tracked],
        crossings=[],
        class_ids={7: 4},
        norm_areas={7: 100},
    )
    horizon = LineSegment(start=(0, 30), end=(159, 0))
    counting = LineSegment(start=(0, 119), end=(159, 0))
    canvas = annotate_bgr(
        image,
        result,
        horizon,
        counting,
        class_names={4: "MTW"},
    )
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    assert int(gray.sum()) > 0
    # Label is drawn above the box (y < 40); the unannotated source was zeros.
    assert int(canvas[34, 22:].sum()) > 0
    assert tuple(int(v) for v in canvas[40, 20]) == overlay_bgr(4)


def test_ffmpeg_prefers_h264_nvenc_for_browser() -> None:
    """Processed MP4 prefers H.264 NVENC over HEVC so browser <video> can decode (S20)."""
    args = ffmpeg_video_args(
        Path("out.mp4"),
        encoder_list="libx264\nhevc_nvenc\nlibx265\nh264_nvenc",
    )
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-cq" in args and "32" in args
    assert "-preset" in args and "p4" in args
    assert "-movflags" in args
    assert "+frag_keyframe+empty_moov+default_base_moof" in args
    assert "-frag_duration" in args and "1000000" in args


def test_ffmpeg_prefers_libx264_before_hevc() -> None:
    """Without h264_nvenc, software H.264 beats HEVC for browser playback."""
    args = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx264\nhevc_nvenc\nlibx265")
    assert args[:2] == ["-c:v", "libx264"]
    assert "-crf" in args and "34" in args
    assert "-preset" in args and "veryfast" in args
    assert "+frag_keyframe+empty_moov+default_base_moof" in args


def test_ffmpeg_fallbacks_keep_fragmented_mp4_flags() -> None:
    """HEVC fallbacks still write streamable fragmented MP4 (S13)."""
    hevc = ffmpeg_video_args(Path("out.mp4"), encoder_list="hevc_nvenc")
    x265 = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx265")
    x264 = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx264")
    for args in (hevc, x265, x264):
        assert "-movflags" in args
        assert "+frag_keyframe+empty_moov+default_base_moof" in args
        assert "-frag_duration" in args and "1000000" in args


class _FakeStdin:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.chunks.append(data)
        return len(data)

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return 0


class _FakeImage:
    """Minimal buffer with copy()/tobytes() like a NumPy BGR frame."""

    def __init__(self, fill: int, *, nbytes: int = 12) -> None:
        self.fill = fill
        self.nbytes = nbytes

    def copy(self) -> _FakeImage:
        return _FakeImage(self.fill, nbytes=self.nbytes)

    def tobytes(self) -> bytes:
        return bytes([self.fill] * self.nbytes)


def _empty_result() -> FrameCVResult:
    return FrameCVResult(tracked=[], crossings=[], class_ids={}, norm_areas={})


def test_ffmpeg_renderer_copies_frame_before_enqueue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Decoder buffer mutations after write() must not corrupt queued frames."""
    fake = _FakeProc()
    monkeypatch.setattr("viana.stages.render.shutil.which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "viana.stages.render.run_captured",
        lambda *_a, **_k: SimpleNamespace(stdout="h264_nvenc"),
    )
    monkeypatch.setattr("viana.stages.render.subprocess.Popen", lambda *_a, **_k: fake)
    monkeypatch.setattr("viana.stages.render.close_stdio", lambda _proc: None)

    renderer = FfmpegRenderer(tmp_path / "out.mp4", width=2, height=2, fps=25.0)
    image = _FakeImage(7)
    frame = VideoFrame(index=0, pts_ms=0.0, width=2, height=2, image=image)
    renderer.write(frame, _empty_result())
    image.fill = 99  # simulate capture buffer reuse
    renderer.close()

    assert fake.stdin.closed is True
    assert len(fake.stdin.chunks) == 1
    assert fake.stdin.chunks[0] == bytes([7] * 12)


def test_ffmpeg_renderer_drains_queue_on_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """close() waits for queued frames before shutting down ffmpeg stdin."""
    fake = _FakeProc()
    monkeypatch.setattr("viana.stages.render.shutil.which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "viana.stages.render.run_captured",
        lambda *_a, **_k: SimpleNamespace(stdout="libx264"),
    )
    monkeypatch.setattr("viana.stages.render.subprocess.Popen", lambda *_a, **_k: fake)
    monkeypatch.setattr("viana.stages.render.close_stdio", lambda _proc: None)

    renderer = FfmpegRenderer(tmp_path / "out.mp4", width=2, height=2, fps=25.0)
    for index in range(5):
        image = _FakeImage(index + 1)
        frame = VideoFrame(index=index, pts_ms=float(index), width=2, height=2, image=image)
        renderer.write(frame, _empty_result())
    renderer.close()

    assert len(fake.stdin.chunks) == 5
    assert fake.stdin.chunks[0] == bytes([1] * 12)
    assert fake.stdin.chunks[-1] == bytes([5] * 12)
    assert renderer._thread.is_alive() is False
