"""Annotated video overlay (class names on boxes)."""

from __future__ import annotations

from viana.config.job import LineSegment
from viana.domain.boxes import Detection
from viana.stages.cv_core import FrameCVResult
from viana.stages.render import annotate_bgr, ffmpeg_video_args, overlay_bgr
from viana.stages.track import TrackedDetection


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
    from pathlib import Path

    args = ffmpeg_video_args(
        Path("out.mp4"),
        encoder_list="libx264\nhevc_nvenc\nlibx265\nh264_nvenc",
    )
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-cq" in args and "28" in args
    assert "-movflags" in args
    assert "+frag_keyframe+empty_moov+default_base_moof" in args
    assert "-frag_duration" in args and "1000000" in args


def test_ffmpeg_prefers_libx264_before_hevc() -> None:
    """Without h264_nvenc, software H.264 beats HEVC for browser playback."""
    from pathlib import Path

    args = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx264\nhevc_nvenc\nlibx265")
    assert args[:2] == ["-c:v", "libx264"]
    assert "+frag_keyframe+empty_moov+default_base_moof" in args


def test_ffmpeg_fallbacks_keep_fragmented_mp4_flags() -> None:
    """HEVC fallbacks still write streamable fragmented MP4 (S13)."""
    from pathlib import Path

    hevc = ffmpeg_video_args(Path("out.mp4"), encoder_list="hevc_nvenc")
    x265 = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx265")
    x264 = ffmpeg_video_args(Path("out.mp4"), encoder_list="libx264")
    for args in (hevc, x265, x264):
        assert "-movflags" in args
        assert "+frag_keyframe+empty_moov+default_base_moof" in args
        assert "-frag_duration" in args and "1000000" in args
