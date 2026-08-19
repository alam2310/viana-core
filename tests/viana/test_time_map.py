"""Phase 3 — time map OCR parse, interpolation, user fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from viana.config.files import repo_root
from viana.config.job import JobMetadata
from viana.stages.time_map import (
    TimeAnchor,
    TimeMap,
    extract_ocr_time,
    load_time_map,
    next_boundary_delta_ms,
    normalize_ocr_date,
    parse_ocr_texts,
    time_map_from_metadata,
)


def test_parse_ocr_texts() -> None:
    """Time, date, and location are split out of OSD-like strings."""
    parsed = parse_ocr_texts(["09:00:12 15-03-2026 NH48 Km42", "Monday"])
    assert parsed.time == "09:00:12"
    assert parsed.date == "15-03-2026"
    assert parsed.location == "NH48 Km42"


def test_normalize_ocr_date_repairs_spaces_and_year() -> None:
    """OCR spacing and 2074→2024 year drift are normalized."""
    assert normalize_ocr_date("18-10 2074 Fri") == "18-10-2024"
    assert extract_ocr_time(["18-10 2074 Fri 07 21 26"]) == "07:21:26"
    assert extract_ocr_time(["18-10 2074 Fri 02 21.26"]) == "02:21:26"
    assert extract_ocr_time(["18-10 2074 Fri 02 2125"]) == "02:21:25"


def test_user_fallback_when_no_ocr() -> None:
    """Job metadata supplies wall clock when OCR is missing."""
    time_map = time_map_from_metadata(
        "job_1",
        "clip",
        JobMetadata(user_start_date="15-03-2026", user_start_time="09:00:00"),
    )
    wall, source, _conf = time_map.resolve(60_000)
    assert source == "user_fallback"
    assert wall == "2026-03-15T09:01:00Z"


def test_interpolate_between_ocr_anchors() -> None:
    """PTS between two OCR anchors is linearly interpolated."""
    time_map = TimeMap(
        job_id="job_1",
        video_stem="clip",
        anchors=[
            TimeAnchor(
                video_pts_ms=0,
                wall_time="2026-03-15T09:00:00Z",
                source="ocr_anchor",
                ocr_confidence=0.9,
            ),
            TimeAnchor(
                video_pts_ms=60_000,
                wall_time="2026-03-15T09:01:00Z",
                source="ocr_recalibrated",
                ocr_confidence=0.8,
            ),
        ],
    )
    wall, source, _conf = time_map.resolve(30_000)
    assert wall == "2026-03-15T09:00:30Z"
    assert source == "ocr_recalibrated"


def test_load_time_map_fixture() -> None:
    """Committed fixture matches time_map.schema.json."""
    path = repo_root() / "packages" / "contracts" / "fixtures" / "time_map.json"
    loaded = load_time_map(path)
    assert loaded.schema_version == 1
    assert loaded.anchors[0].source == "ocr_anchor"


def test_next_clock_boundary_delta() -> None:
    """Next 15-minute clock mark matches legacy TimeSyncEngine math."""
    now = datetime(2026, 3, 15, 9, 7, 0, tzinfo=timezone.utc)
    assert next_boundary_delta_ms(now) == 8 * 60 * 1000


def test_unavailable_without_anchors_or_user(tmp_path: Path) -> None:
    """No OCR and no user time → wall_time unavailable (not invented)."""
    time_map = TimeMap(job_id="job_1", video_stem="clip")
    wall, source, conf = time_map.resolve(0)
    assert wall is None
    assert source == "unavailable"
    assert conf is None
    _ = tmp_path
