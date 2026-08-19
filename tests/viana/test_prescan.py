"""Phase 4 — line proposal, profiles, OCR hits, prescan."""

from __future__ import annotations

from pathlib import Path

from viana.config.job import LineSegment
from viana.domain.geometry import scale_line
from viana.io.profiles import CalibrationProfile, list_profiles, save_profile
from viana.stages.lines import GEOMETRIC_CONFIDENCE, PROFILE_CONFIDENCE, propose_lines
from viana.stages.ocr import (
    OCR_FALLBACK_SCALE,
    OCR_ROI_SCALE,
    parse_corner_osd_hits,
    parse_frame_corner_osd,
    parse_osd_hits,
)
from viana.stages.prescan import SampledVideo, VideoMeta, preview_jpeg_path, run_prescan


def test_geometric_lines_stay_in_frame() -> None:
    """Default proposal endpoints are inside the sampled frame."""
    proposed = propose_lines(1920, 1080)
    proposed.horizon_line.assert_within_frame(1920, 1080, "horizon_line")
    proposed.counting_line.assert_within_frame(1920, 1080, "counting_line")
    assert proposed.confidence == GEOMETRIC_CONFIDENCE
    assert proposed.horizon_line.start[1] < proposed.counting_line.start[1]


def test_geometric_lines_span_frame_width() -> None:
    """Proposed lines pin X to the left and right pixel edges."""
    proposed = propose_lines(1920, 1080)
    assert proposed.horizon_line.start == (0, 648)
    assert proposed.horizon_line.end == (1919, 0)
    assert proposed.counting_line.start == (0, 1079)
    assert proposed.counting_line.end == (1919, 0)


def test_scale_line_maps_profile_resolution() -> None:
    """Profile lines scale then clamp like the UI canvas rule."""
    line = LineSegment(start=(100, 200), end=(1900, 400))
    scaled = scale_line(line, (1920, 1080), (1280, 720))
    scaled.assert_within_frame(1280, 720, "scaled")
    assert scaled.start[0] < scaled.end[0]


def test_matching_profile_overrides_geometry(tmp_path: Path) -> None:
    """A same-aspect profile is scaled onto the preview frame."""
    profile = CalibrationProfile(
        profile_id="morning_northbound",
        profile_name="Morning NB",
        reference_resolution=(1920, 1080),
        horizon_line=LineSegment(start=(120, 400), end=(1800, 520)),
        counting_line=LineSegment(start=(80, 900), end=(1840, 780)),
        source="user_drawn",
    )
    save_profile(tmp_path, profile)
    proposed = propose_lines(1280, 720, list_profiles(tmp_path))
    assert proposed.confidence == PROFILE_CONFIDENCE
    proposed.horizon_line.assert_within_frame(1280, 720, "horizon_line")
    proposed.counting_line.assert_within_frame(1280, 720, "counting_line")


def test_frame_guided_lines_used_without_profile() -> None:
    """When no profile exists, frame cues can raise proposal confidence."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("numpy")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (0, 320), (1279, 280), (255, 255, 255), 4)
    cv2.line(frame, (0, 560), (1279, 540), (255, 255, 255), 4)
    proposed = propose_lines(1280, 720, [], frame=frame)
    assert proposed.confidence > GEOMETRIC_CONFIDENCE
    proposed.horizon_line.assert_within_frame(1280, 720, "horizon_line")
    proposed.counting_line.assert_within_frame(1280, 720, "counting_line")
    assert 280 <= proposed.horizon_line.start[1] <= 360
    assert 500 <= proposed.counting_line.start[1] <= 620


def test_frame_guided_lines_are_deterministic() -> None:
    """Same frame should produce the same proposal every time."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("numpy")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (0, 330), (1279, 260), (255, 255, 255), 5)
    cv2.line(frame, (0, 600), (1279, 520), (255, 255, 255), 5)
    first = propose_lines(1280, 720, [], frame=frame)
    second = propose_lines(1280, 720, [], frame=frame)
    assert first.horizon_line == second.horizon_line
    assert first.counting_line == second.counting_line
    assert first.confidence == second.confidence


def test_invalid_frame_shape_falls_back_to_geometric() -> None:
    """Mismatched frame dimensions should not crash frame-guided path."""
    np = __import__("numpy")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    proposed = propose_lines(1280, 720, [], frame=frame)
    assert proposed.confidence == GEOMETRIC_CONFIDENCE


def test_parse_osd_hits_respects_min_confidence() -> None:
    """Low-prob EasyOCR strings are dropped before time/date parse."""
    parsed, mean = parse_osd_hits(
        [("09:00:00 15-03-2026 NH48", 0.9), ("garbage", 0.1)],
        min_confidence=0.6,
    )
    assert parsed.time == "09:00:00"
    assert parsed.date == "15-03-2026"
    assert parsed.location == "NH48"
    assert mean is not None
    assert mean == 0.9


def test_parse_corner_osd_hits_splits_metadata_and_location() -> None:
    """Metadata and location ROIs are parsed independently."""
    parsed, mean = parse_corner_osd_hits(
        [("18-10-2024 Fri 07 21 26", 0.82)],
        [("L11-BARABANKI", 0.78)],
        min_confidence=0.6,
    )
    assert parsed.date == "18-10-2024"
    assert parsed.time == "07:21:26"
    assert parsed.location == "L11-BARABANKI"
    assert mean is not None


def test_run_prescan_writes_preview_and_response(tmp_path: Path) -> None:
    """Injected sampler/OCR produce a contract-shaped payload plus JPEG."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")

    def sampler(_source: Path, offset: float) -> SampledVideo:
        return SampledVideo(
            meta=VideoMeta(
                width=1920,
                height=1080,
                fps=25.0,
                duration_sec=3600.0,
                frame_count=90_000,
            ),
            frame_offset_sec=offset,
            frame=None,
            preview_jpeg=None,
        )

    def ocr_reader(_frame: object) -> list[tuple[str, float]]:
        return [("09:00:00 15-03-2026 NH48 Km42", 0.82)]

    result = run_prescan(
        video,
        "nh48",
        frame_offset_sec=0.0,
        output_dir=tmp_path,
        sampler=sampler,
        ocr_reader=ocr_reader,
        prescan_id="prescan_test_001",
        auto_skip_dark_frames=False,
    )
    assert result.prescan_id == "prescan_test_001"
    assert result.video_meta.width == 1920
    assert result.ocr.time == "09:00:00"
    assert result.ocr.location == "NH48 Km42"
    assert result.proposed_lines is not None
    preview = preview_jpeg_path(tmp_path, "prescan_test_001")
    assert preview.is_file()
    assert preview.read_bytes()[:3] == b"\xff\xd8\xff"
    assert result.preview_url == str(preview)


def test_find_best_frame_offset_respects_explicit_scrub() -> None:
    """User scrub offset bypasses dark-frame auto-skip (G7)."""
    from viana.stages.prescan import find_best_frame_offset

    offset = find_best_frame_offset(
        Path("/nonexistent.mp4"),
        requested_offset_sec=12.5,
        scan_sec=30.0,
        step_sec=1.0,
        luminance_threshold=28.0,
    )
    assert offset == 12.5


def test_osd_band_score_is_high_on_striped_metadata_roi() -> None:
    """Top-left contrast (OSD) scores above the dark-frame min."""
    np = __import__("pytest").importorskip("numpy")
    from viana.stages.prescan import osd_band_score

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = 40
    frame[0:16, 0:144] = 20
    frame[0:16, 0:144:4] = 255
    assert osd_band_score(frame) > 20.0


def test_sample_opening_frame_skips_dark_then_blank_osd(tmp_path: Path) -> None:
    """Opening scan waits for a bright frame with top-band OSD texture (G7/S08)."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("pytest").importorskip("numpy")
    from viana.stages.prescan import sample_opening_frame

    path = tmp_path / "opening.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        __import__("pytest").skip("OpenCV VideoWriter MJPG unavailable")
    try:
        for index in range(40):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            if index >= 20:
                frame[:] = 80
                frame[0:16, 0:144] = 20
                frame[0:16, 0:144:4] = 255
            writer.write(frame)
    finally:
        writer.release()

    sampled = sample_opening_frame(
        path,
        requested_offset_sec=0.0,
        scan_sec=4.0,
        step_sec=1.0,
        luminance_threshold=28.0,
        min_osd_score=20.0,
        probe_start_sec=1.0,
    )
    assert sampled.frame is not None
    assert sampled.frame_offset_sec == __import__("pytest").approx(2.0, abs=0.15)
    assert sampled.meta.width == 320
    assert sampled.meta.height == 240


def test_parse_frame_corner_osd_skips_fallback_when_fast_path_hits() -> None:
    """S08 fast ROI is used when time and date already parse."""

    class FastOnlyReader:
        def readtext(self, _rgb: object, **_kwargs: object) -> list[object]:
            if _kwargs.get("allowlist"):
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "L11-BARABANKI", 0.9]]
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "18-10-2024 Fri 07 21 26", 0.9]]

    np = __import__("pytest").importorskip("numpy")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    parsed, mean = parse_frame_corner_osd(frame, FastOnlyReader(), 0.6)
    assert parsed.time == "07:21:26"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "L11-BARABANKI"
    assert mean is not None


def test_parse_frame_corner_osd_retries_wide_roi_when_fast_path_misses() -> None:
    """Wide 4× pass fills time/date when the tight crop returns junk."""

    class TwoPassReader:
        def __init__(self) -> None:
            self.calls = 0

        def readtext(self, _rgb: object, **_kwargs: object) -> list[object]:
            self.calls += 1
            if self.calls <= 2:
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "xxx", 0.9]]
            if _kwargs.get("allowlist"):
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "L11-BARABANKI", 0.9]]
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "18-10-2024 Fri 07 21 26", 0.9]]

    np = __import__("pytest").importorskip("numpy")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    reader = TwoPassReader()
    parsed, _mean = parse_frame_corner_osd(frame, reader, 0.6)
    assert reader.calls == 4
    assert parsed.time == "07:21:26"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "L11-BARABANKI"
    assert OCR_ROI_SCALE == 2.0
    assert OCR_FALLBACK_SCALE == 4.0
