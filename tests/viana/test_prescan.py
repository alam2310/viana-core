"""Phase 4 — line proposal, profiles, OCR hits, prescan."""

from __future__ import annotations

from pathlib import Path

from viana.config.job import LineSegment
from viana.domain.geometry import scale_line
from viana.io.profiles import CalibrationProfile, list_profiles, save_profile
from viana.stages.lines import GEOMETRIC_CONFIDENCE, PROFILE_CONFIDENCE, propose_lines
from viana.stages.ocr import (
    OCR_BAND_SCALE,
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
    np = __import__("pytest").importorskip("numpy")
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


def test_frame_guided_lines_keep_parallel_direction() -> None:
    """Horizon/counting should follow one dominant road direction."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("numpy")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (0, 260), (1279, 430), (255, 255, 255), 4)
    cv2.line(frame, (0, 500), (1279, 670), (255, 255, 255), 4)
    proposed = propose_lines(1280, 720, [], frame=frame)
    h_dy = proposed.horizon_line.end[1] - proposed.horizon_line.start[1]
    c_dy = proposed.counting_line.end[1] - proposed.counting_line.start[1]
    assert h_dy > 0
    assert c_dy > 0
    assert abs(h_dy - c_dy) <= 80


def test_invalid_frame_shape_falls_back_to_geometric() -> None:
    """Mismatched frame dimensions should not crash frame-guided path."""
    np = __import__("pytest").importorskip("numpy")
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    proposed = propose_lines(1280, 720, [], frame=frame)
    assert proposed.confidence == GEOMETRIC_CONFIDENCE


def test_frame_guided_lines_prefer_road_band_over_rooflines() -> None:
    """Upper-frame building edges must not invert the road slope (S10)."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("pytest").importorskip("numpy")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (0, 90), (1279, 240), (255, 255, 255), 7)
    cv2.line(frame, (0, 130), (1279, 280), (255, 255, 255), 7)
    cv2.line(frame, (0, 390), (1279, 250), (255, 255, 255), 4)
    cv2.line(frame, (0, 610), (1279, 470), (255, 255, 255), 4)
    proposed = propose_lines(1280, 720, [], frame=frame)
    h_dy = proposed.horizon_line.end[1] - proposed.horizon_line.start[1]
    c_dy = proposed.counting_line.end[1] - proposed.counting_line.start[1]
    assert h_dy < 0
    assert c_dy < 0
    assert 300 <= proposed.horizon_line.start[1] <= 450
    assert proposed.counting_line.start[1] >= proposed.horizon_line.start[1] + 80
    proposed.horizon_line.assert_within_frame(1280, 720, "horizon_line")
    proposed.counting_line.assert_within_frame(1280, 720, "counting_line")


def test_matching_profile_overrides_frame_guided(tmp_path: Path) -> None:
    """Saved profile still wins even when a sampled frame is present (S10)."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("pytest").importorskip("numpy")
    profile = CalibrationProfile(
        profile_id="s10_override",
        profile_name="S10 override",
        reference_resolution=(1280, 720),
        horizon_line=LineSegment(start=(0, 200), end=(1279, 180)),
        counting_line=LineSegment(start=(0, 500), end=(1279, 480)),
        source="user_drawn",
    )
    save_profile(tmp_path, profile)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (0, 390), (1279, 250), (255, 255, 255), 4)
    cv2.line(frame, (0, 610), (1279, 470), (255, 255, 255), 4)
    proposed = propose_lines(1280, 720, list_profiles(tmp_path), frame=frame)
    assert proposed.confidence == PROFILE_CONFIDENCE
    assert proposed.horizon_line.start == (0, 200)
    assert proposed.counting_line.start == (0, 500)


def test_hiv000001_inframe_proposal_near_review_geometry() -> None:
    """S10: no-profile proposal on the parity clip stays near geometry C/D."""
    pytest = __import__("pytest")
    pytest.importorskip("cv2")
    from viana.config.defaults import load_engine_defaults
    from viana.stages.prescan import sample_opening_frame

    clip = Path("/data/raw/hiv000001_inframe.mp4")
    if not clip.is_file():
        clip = Path("/home/mushaffa/Work/ViAna/data/raw/hiv000001_inframe.mp4")
    if not clip.is_file():
        pytest.skip("hiv000001_inframe.mp4 not available")
    defaults = load_engine_defaults()
    sampled = sample_opening_frame(
        clip,
        requested_offset_sec=0.0,
        scan_sec=defaults.prescan.dark_frame_scan_sec,
        step_sec=defaults.prescan.dark_frame_step_sec,
        luminance_threshold=defaults.prescan.dark_frame_luminance_threshold,
        min_osd_score=defaults.prescan.osd_min_score,
        probe_start_sec=defaults.prescan.osd_probe_start_sec,
    )
    assert sampled.frame is not None
    proposed = propose_lines(sampled.meta.width, sampled.meta.height, [], frame=sampled.frame)
    proposed.horizon_line.assert_within_frame(1920, 1080, "horizon_line")
    proposed.counting_line.assert_within_frame(1920, 1080, "counting_line")
    assert proposed.horizon_line.start[0] == 0
    assert proposed.horizon_line.end[0] == 1919
    assert proposed.counting_line.start[0] == 0
    assert proposed.counting_line.end[0] == 1919
    # Geometry C/D: horizon left ~500–540, right ~200–325; counting left ~775–850.
    assert 430 <= proposed.horizon_line.start[1] <= 650
    assert 150 <= proposed.horizon_line.end[1] <= 430
    assert proposed.horizon_line.end[1] < proposed.horizon_line.start[1]
    assert 680 <= proposed.counting_line.start[1] <= 980
    assert proposed.counting_line.start[1] > proposed.horizon_line.start[1]
    assert proposed.confidence > GEOMETRIC_CONFIDENCE


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


def test_parse_osd_hits_does_not_prefer_date_year_as_time() -> None:
    """Corner allowlist can glue month+year; band clock with colons must win."""
    parsed, _mean = parse_osd_hits(
        [
            ("29-07 2026 WE 0982713", 0.9),
            ("Bangalorebypass_J2", 0.9),
            ("29-07-2026 Wed 09:27:32", 0.9),
        ],
        min_confidence=0.6,
    )
    assert parsed.time == "09:27:32"
    assert parsed.date == "29-07-2026"
    assert parsed.location == "Bangalorebypass_J2"


def test_parse_osd_hits_salvages_clock_appended_to_location() -> None:
    """Date can parse while a spaced-colon clock is stuck on the location line."""
    parsed, _mean = parse_osd_hits(
        [("28-07-2026 Tue", 0.9), ("Bangalorebypassjz 06 :44:35", 0.9)],
        min_confidence=0.6,
    )
    assert parsed.time == "06:44:35"
    assert parsed.date == "28-07-2026"
    assert parsed.location == "Bangalorebypassjz"


def test_parse_osd_hits_plus_separator_clock_and_unhyphenated_location() -> None:
    """test_video.mp4: EasyOCR uses + as a colon and repeats the location code."""
    parsed, _mean = parse_osd_hits(
        [
            ("18-10-2024 Fri 08:38+31", 0.9),
            ("I3TRARARANKT", 0.8),
            ("L3TRARARANKT", 0.8),
            ("I37NARARAN80", 0.8),
        ],
        min_confidence=0.6,
    )
    assert parsed.time == "08:38:31"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "L3TRARARANKT"
    assert " " not in (parsed.location or "")


def test_parse_osd_hits_does_not_join_hyphenated_locations() -> None:
    """hiv000001: mixed-polarity location OCR must pick one label, not all of them."""
    parsed, _mean = parse_osd_hits(
        [
            ("18-10 2074 Fri 02 21.25", 0.9),
            ("LITO-RARARANKI", 0.8),
            ("L1TO-RARARANKI", 0.8),
            ("LIT-BRBNKI", 0.8),
        ],
        min_confidence=0.6,
    )
    assert parsed.time == "02:21:25"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "L1TO-RARARANKI"
    assert " " not in (parsed.location or "")


def test_parse_osd_hits_prefers_barabanki_over_rararanki() -> None:
    """Mixed-polarity location OCR can emit both a junk and a BANKI reading."""
    parsed, _mean = parse_osd_hits(
        [
            ('19 10-2024 Sat 05:34"04', 0.9),
            ("L3TORARARANKI", 0.8),
            ("LZTBARABANKI", 0.8),
        ],
        min_confidence=0.6,
    )
    assert parsed.time == "05:34:04"
    assert parsed.date == "19-10-2024"
    assert parsed.location == "LZTBARABANKI"


def test_crop_has_mixed_text_polarity_on_bicolor_osd() -> None:
    """White-on-dark plus black-on-light strokes should trip the mixed-polarity gate."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("pytest").importorskip("numpy")
    from viana.stages.ocr import crop_has_mixed_text_polarity

    crop = np.zeros((40, 160, 3), dtype=np.uint8)
    crop[:, 80:] = 220
    cv2.putText(crop, "AB", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(crop, "CD", (90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    assert crop_has_mixed_text_polarity(crop) is True
    assert crop_has_mixed_text_polarity(np.zeros((40, 160, 3), dtype=np.uint8)) is False


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


def test_osd_band_score_detects_bottom_left_osd() -> None:
    """S21: OSD contrast in the bottom-left band is scored even if the top is blank."""
    np = __import__("pytest").importorskip("numpy")
    from viana.stages.prescan import osd_band_score

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = 40
    frame[210:240, 0:150] = 20
    frame[210:240, 0:150:4] = 255
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
    assert OCR_BAND_SCALE == 2.0


def test_parse_frame_corner_osd_uses_bands_when_location_is_elsewhere() -> None:
    """S21: missing location after a time/date hit skips 4× corners and scans bands."""

    class LocationElsewhereReader:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, int]] = []

        def readtext(self, rgb: object, **kwargs: object) -> list[object]:
            np = __import__("numpy")
            height, width = np.asarray(rgb).shape[:2]
            self.shapes.append((height, width))
            if kwargs.get("allowlist"):
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "xxx", 0.9]]
            if self.shapes and len(self.shapes) == 1:
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "18-10-2024 Fri 07 21 26", 0.9]]
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "CAM-TOPCENTER", 0.9]]

    np = __import__("pytest").importorskip("numpy")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    reader = LocationElsewhereReader()
    parsed, _mean = parse_frame_corner_osd(frame, reader, 0.6)
    assert parsed.time == "07:21:26"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "CAM-TOPCENTER"
    assert (24, 464) not in reader.shapes
    assert len(reader.shapes) == 4


def test_parse_frame_corner_osd_layout_variant_bands() -> None:
    """S21: timestamp bottom-left + location top-center are recovered from bands."""
    np = __import__("pytest").importorskip("numpy")

    class ColorBandReader:
        def readtext(self, rgb: object, **kwargs: object) -> list[object]:
            arr = np.asarray(rgb)
            red_frac = float((arr[:, :, 0] > 200).mean()) if arr.size else 0.0
            blue_frac = float((arr[:, :, 2] > 200).mean()) if arr.size else 0.0
            if red_frac > 0.02:
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "LITO-TOPCENTER", 0.92]]
            if blue_frac > 0.02:
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "18-10-2024 Fri 07 21 26", 0.91]]
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]], "xxx", 0.9]]

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[8:40, 400:620] = (0, 0, 255)
    frame[320:355, 20:300] = (255, 0, 0)
    parsed, _mean = parse_frame_corner_osd(frame, ColorBandReader(), 0.6)
    assert parsed.time == "07:21:26"
    assert parsed.date == "18-10-2024"
    assert parsed.location == "LITO-TOPCENTER"


def _paint_layout_variant_frame(width: int = 640, height: int = 360) -> object:
    cv2 = __import__("cv2")
    np = __import__("numpy")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        __import__("pytest").skip("Pillow is required to paint a readable OSD fixture")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 32)
    except OSError:
        __import__("pytest").skip(f"OSD fixture font missing: {font_path}")
    image = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.text((360, 8), "LITO-TOPCENTER", font=font, fill=(255, 255, 255))
    draw.text((16, 300), "18-10-2024", font=font, fill=(255, 255, 255))
    draw.text((16, 332), "07:21:26", font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def test_layout_variant_clip_easyocr_recovers_osd(tmp_path: Path) -> None:
    """S21: EasyOCR on a clip with top-center location and bottom-left clock."""
    pytest = __import__("pytest")
    pytest.importorskip("easyocr")
    cv2 = pytest.importorskip("cv2")
    from viana.stages.ocr import CornerOsdReader

    path = tmp_path / "osd_layout_variant.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 360))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter MJPG unavailable")
    try:
        frame = _paint_layout_variant_frame()
        for _ in range(8):
            writer.write(frame)
    finally:
        writer.release()
    assert path.is_file()
    capture = cv2.VideoCapture(str(path))
    ok, sampled = capture.read()
    capture.release()
    assert ok
    parsed, _mean = CornerOsdReader(gpu=False).parse(sampled, 0.6)
    assert parsed.time == "07:21:26"
    assert parsed.date == "18-10-2024"
    assert parsed.location is not None
    assert "TOPCENTER" in parsed.location.replace(" ", "")
