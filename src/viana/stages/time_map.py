"""Map video PTS to wall clock (OCR anchors + user fallback). No 15-min CSV here."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from viana.config.job import JobMetadata
from viana.io.csv_schema import WallTimeSource

TIME_PATTERN = re.compile(r"\d{2}:\d{2}:\d{2}")
DATE_PATTERN = re.compile(r"\d{2}[-/]\d{2}[-/]\d{2,4}")
DATE_LOOSE_PATTERN = re.compile(r"\d{2}[-/ ]\d{2}[-/ ]\d{4}")
TIME_SPACED_PATTERN = re.compile(r"\b(\d{2})\s+(\d{2})\s+(\d{2})\b")
TIME_PARTIAL_PATTERN = re.compile(r"\b(\d{2})\s+(\d{2}):(\d{2})\b")
TIME_DOT_PATTERN = re.compile(r"\b(\d{2})\s+(\d{2})[.:](\d{2})\b")
TIME_SEP_PATTERN = re.compile(r"\b(\d{2})[.:](\d{2})[.:](\d{2})\b")
TIME_FLEX_PATTERN = re.compile(r"\b(\d{2})\s*[:.\"'`*;+]\s*(\d{2})\s*[:.\"'`*;+]\s*(\d{2})\b")
TIME_COLON_SPACE_PATTERN = re.compile(r"\b(\d{2})[:.](\d{2})\s+(\d{2})\b")
TIME_COMPACT_PATTERN = re.compile(r"\b(\d{2})\s+(\d{4})\b")
TIME_DOT_GLUED_PATTERN = re.compile(r"\b(\d{2})[.:](\d{4})\b")
TIME_GLUED_PATTERN = re.compile(r"\b(\d{6})\b")
LOCATION_NOISE = {
    "f",
    "u",
    "s",
    "ue",
    "we",
    "fri",
    "mon",
    "tue",
    "wed",
    "thu",
    "sat",
    "sun",
}
IGNORE_WORDS = {
    "mon",
    "tue",
    "wed",
    "we",
    "thu",
    "fri",
    "sat",
    "sun",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


class ParsedOcr(BaseModel):
    """Structured OCR fields parsed from on-screen text (EasyOCR runs in Phase 4)."""

    model_config = ConfigDict(extra="forbid")

    time: str | None = None
    date: str | None = None
    location: str | None = None


class TimeAnchor(BaseModel):
    """One PTS ↔ wall-clock sample."""

    model_config = ConfigDict(extra="forbid")

    video_pts_ms: float
    wall_time: str
    source: WallTimeSource
    ocr_confidence: float | None = None
    date: str | None = None
    location: str | None = None


class TimeMap(BaseModel):
    """``{stem}.time_map.json`` — interpolate wall time for crossing events."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str
    video_stem: str
    anchors: list[TimeAnchor] = Field(default_factory=list)
    user_start_date: str | None = None
    user_start_time: str | None = None

    def resolve(self, video_pts_ms: float) -> tuple[str | None, WallTimeSource, float | None]:
        """Return ``(wall_time_iso, source, ocr_confidence)`` for a video timestamp."""
        if self.anchors:
            return self._from_anchors(video_pts_ms)
        fallback = parse_user_datetime(self.user_start_date, self.user_start_time)
        if fallback is None:
            return None, "unavailable", None
        wall = fallback + timedelta(milliseconds=video_pts_ms)
        return format_wall_time(wall), "user_fallback", None

    def _from_anchors(self, video_pts_ms: float) -> tuple[str | None, WallTimeSource, float | None]:
        ordered = sorted(self.anchors, key=lambda item: item.video_pts_ms)
        if len(ordered) == 1:
            anchor = ordered[0]
            base = parse_wall_time(anchor.wall_time)
            wall = base + timedelta(milliseconds=video_pts_ms - anchor.video_pts_ms)
            return format_wall_time(wall), anchor.source, anchor.ocr_confidence
        left = ordered[0]
        right = ordered[-1]
        for item in ordered:
            if item.video_pts_ms <= video_pts_ms:
                left = item
            if item.video_pts_ms >= video_pts_ms:
                right = item
                break
        if left.video_pts_ms == right.video_pts_ms:
            return left.wall_time, left.source, left.ocr_confidence
        span = right.video_pts_ms - left.video_pts_ms
        t = (video_pts_ms - left.video_pts_ms) / span
        left_dt = parse_wall_time(left.wall_time)
        right_dt = parse_wall_time(right.wall_time)
        wall = left_dt + (right_dt - left_dt) * t
        source: WallTimeSource = left.source
        if left.source in ("ocr_anchor", "ocr_recalibrated"):
            source = "ocr_recalibrated"
        conf = left.ocr_confidence
        return format_wall_time(wall), source, conf


def format_wall_time(moment: datetime) -> str:
    """UTC ISO-8601 with a trailing Z."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_wall_time(value: str) -> datetime:
    """Parse an ISO wall_time string as aware UTC."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_user_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    """Combine job metadata date + time into a UTC datetime (naive treated as UTC)."""
    if not date_str or not time_str:
        return None
    time_part = time_str.strip()
    date_part = date_str.strip().replace("/", "-")
    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%y %H:%M:%S"):
        try:
            parsed = datetime.strptime(f"{date_part} {time_part}", fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def normalize_ocr_date(raw: str) -> str | None:
    """Normalize OCR date fragments to DD-MM-YYYY."""
    match = DATE_PATTERN.search(raw) or DATE_LOOSE_PATTERN.search(raw)
    if match is None:
        return None
    normalized = match.group().replace("/", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    parts = normalized.split("-")
    if len(parts) != 3:
        return None
    day, month, year = parts
    year = _repair_ocr_year(year)
    if not _plausible_calendar_date(day, month, year):
        return None
    return f"{day}-{month}-{year}"


def _repair_ocr_year(year: str) -> str:
    """Map common 2↔7 OSD substitutions (2074, 7074) back to 20xx."""
    if len(year) == 4 and year.startswith("70"):
        year = "20" + year[2:]
    if year == "2074":
        return "2024"
    if len(year) == 4 and year[0] in "67" and year.endswith("074"):
        return "2024"
    return year


def _plausible_calendar_date(day: str, month: str, year: str) -> bool:
    try:
        day_n, month_n, year_n = int(day), int(month), int(year)
    except ValueError:
        return False
    return 1 <= day_n <= 31 and 1 <= month_n <= 12 and 2000 <= year_n <= 2039


def is_valid_clock_time(value: str) -> bool:
    """Return True when ``value`` is a plausible HH:MM:SS clock reading."""
    parts = value.split(":")
    if len(parts) != 3:
        return False
    try:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return False
    return 0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59


def is_plausible_ocr_date(value: str | None) -> bool:
    """True when ``value`` is DD-MM-YYYY with a 2000–2039 year."""
    if not value:
        return False
    parts = value.split("-")
    if len(parts) != 3:
        return False
    return _plausible_calendar_date(parts[0], parts[1], parts[2])


def _four_digits_look_like_year(digits: str) -> bool:
    """True when compact OCR digits are a calendar year, not MMSS."""
    if len(digits) != 4 or not digits.isdigit():
        return False
    year = int(digits)
    return year == 2074 or 1900 <= year <= 2099


def _strip_date_tokens(text: str) -> str:
    cleaned = DATE_PATTERN.sub(" ", text)
    cleaned = DATE_LOOSE_PATTERN.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _best_clock_in_text(text: str) -> tuple[int, str] | None:
    """Return ``(quality, HH:MM:SS)`` for the strongest clock in one OCR line."""
    haystack = _strip_date_tokens(text) or text
    best: tuple[int, str] | None = None

    def consider(score: int, candidate: str) -> None:
        nonlocal best
        if not is_valid_clock_time(candidate):
            return
        if best is None or score > best[0]:
            best = (score, candidate)

    match = TIME_PATTERN.search(haystack)
    if match:
        consider(6, match.group())
    for pattern, score in (
        (TIME_FLEX_PATTERN, 5),
        (TIME_SEP_PATTERN, 5),
        (TIME_COLON_SPACE_PATTERN, 4),
        (TIME_PARTIAL_PATTERN, 3),
        (TIME_DOT_PATTERN, 3),
        (TIME_SPACED_PATTERN, 2),
    ):
        found = pattern.search(haystack)
        if found is None:
            continue
        consider(score, f"{found.group(1)}:{found.group(2)}:{found.group(3)}")
    for compact in TIME_COMPACT_PATTERN.finditer(haystack):
        digits = compact.group(2)
        if _four_digits_look_like_year(digits):
            continue
        consider(1, f"{compact.group(1)}:{digits[0:2]}:{digits[2:4]}")
    for glued_dot in TIME_DOT_GLUED_PATTERN.finditer(haystack):
        digits = glued_dot.group(2)
        if _four_digits_look_like_year(digits):
            continue
        consider(2, f"{glued_dot.group(1)}:{digits[0:2]}:{digits[2:4]}")
    for glued in TIME_GLUED_PATTERN.finditer(haystack):
        digits = glued.group(1)
        consider(1, f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}")
    return best


def extract_ocr_time(texts: list[str]) -> str | None:
    """Extract HH:MM:SS from OCR strings, including common spacing errors."""
    scored: list[tuple[int, str]] = []
    for raw in texts:
        found = _best_clock_in_text(raw)
        if found is None:
            continue
        scored.append(found)
    if not scored:
        return None
    best_score = max(item[0] for item in scored)
    for score, clock in scored:
        if score == best_score:
            return clock
    return None


def parse_metadata_texts(texts: list[str]) -> ParsedOcr:
    """Parse date first, then time from the remainder so years are not clocks."""
    date_str: str | None = None
    for raw in texts:
        candidate = normalize_ocr_date(raw)
        if candidate is not None:
            date_str = candidate
            break
    remainder = [_strip_date_tokens(raw) for raw in texts]
    parsed_time = extract_ocr_time(remainder)
    return ParsedOcr(time=parsed_time, date=date_str, location=None)


def _location_rank(part: str) -> tuple[int, ...]:
    """Prefer a single camera-id/place reading over concatenated OCR guesses."""
    upper = part.upper()
    return (
        0 if " " in part else 1,
        1 if "BARA" in upper else 0,
        1 if "BANKI" in upper else 0,
        1 if "BYPASS" in upper else 0,
        1 if upper.startswith("L") else 0,
        1 if re.search(r"L\d", upper) else 0,
        sum(ch.isalpha() for ch in part),
        1 if "-" in part or "_" in part else 0,
        1 if any(ch.isdigit() for ch in part) else 0,
        len(part),
    )


def _is_camera_code(part: str) -> bool:
    compact = re.sub(r"[^\w]", "", part)
    return len(compact) >= 8


def parse_location_texts(texts: list[str]) -> str | None:
    """Parse camera location label from the bottom-left ROI."""
    location_parts: list[str] = []
    for raw in texts:
        text_clean = _strip_metadata_tokens(raw.strip())
        if not text_clean:
            continue
        lowered = text_clean.lower()
        if lowered in IGNORE_WORDS or lowered in LOCATION_NOISE:
            continue
        loc_cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", text_clean)
        if not loc_cleaned or loc_cleaned.lower() in LOCATION_NOISE:
            continue
        if loc_cleaned.replace(" ", "").isdigit():
            continue
        if re.fullmatch(r"[A-Za-z]{1,3}\s+\d{4,}", loc_cleaned):
            continue
        if _mostly_numeric_osd(loc_cleaned):
            continue
        for token in loc_cleaned.split():
            if len(token) < 3:
                continue
            if token in location_parts:
                continue
            if token.lower() in IGNORE_WORDS or token.lower() in LOCATION_NOISE:
                continue
            location_parts.append(token)
    marked = [part for part in location_parts if "-" in part or "_" in part]
    long_codes = [part for part in location_parts if _is_camera_code(part)]
    if marked or len(long_codes) >= 2:
        pool = list(marked)
        for part in long_codes:
            if part not in pool:
                pool.append(part)
        return max(pool, key=_location_rank)
    landmark = [
        part
        for part in location_parts
        if any(token in part.upper() for token in ("BANKI", "BARA", "BYPASS"))
    ]
    if landmark:
        return max(landmark, key=len)
    return " ".join(location_parts) if location_parts else None


def _mostly_numeric_osd(text: str) -> bool:
    """Reject date/clock leftovers like 07-71576 that still contain a hyphen."""
    compact = re.sub(r"[^\w]", "", text)
    if not compact:
        return True
    digits = sum(ch.isdigit() for ch in compact)
    letters = sum(ch.isalpha() for ch in compact)
    return digits >= 4 and digits > letters


def _strip_metadata_tokens(text: str) -> str:
    """Remove time/date/day tokens so location can be parsed from combined OSD lines."""
    cleaned = TIME_PATTERN.sub("", text).strip()
    cleaned = DATE_PATTERN.sub("", cleaned).strip()
    cleaned = DATE_LOOSE_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_FLEX_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_COLON_SPACE_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_PARTIAL_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_DOT_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_SEP_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_SPACED_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_COMPACT_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_DOT_GLUED_PATTERN.sub("", cleaned).strip()
    cleaned = TIME_GLUED_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"\b\d{2}[.:]\d{3,6}\b", "", cleaned).strip()
    for word in IGNORE_WORDS:
        cleaned = re.sub(rf"\b{word}\b", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def parse_ocr_texts(texts: list[str]) -> ParsedOcr:
    """Extract time/date/location from OCR strings (legacy TimeSyncEngine.extract_metadata)."""
    meta = parse_metadata_texts(texts)
    location = parse_location_texts(texts)
    return ParsedOcr(
        time=meta.time,
        date=meta.date,
        location=location,
    )


def next_boundary_delta_ms(current_time: datetime) -> float:
    """Milliseconds until the next clock :00/:15/:30/:45 (legacy interval helper)."""
    minutes_to_add = 15 - (current_time.minute % 15)
    if minutes_to_add == 0:
        minutes_to_add = 15
    next_boundary = (current_time + timedelta(minutes=minutes_to_add)).replace(
        second=0, microsecond=0
    )
    return (next_boundary - current_time).total_seconds() * 1000.0


def time_map_from_metadata(
    job_id: str,
    video_stem: str,
    metadata: JobMetadata,
    *,
    ocr: ParsedOcr | None = None,
    video_pts_ms: float = 0.0,
    ocr_confidence: float | None = None,
) -> TimeMap:
    """Build an initial time map from OCR and/or user fallback metadata."""
    time_map = TimeMap(
        job_id=job_id,
        video_stem=video_stem,
        user_start_date=metadata.user_start_date,
        user_start_time=metadata.user_start_time,
    )
    wall: datetime | None = None
    source: WallTimeSource = "unavailable"
    date = metadata.user_start_date
    location = metadata.location
    if ocr and ocr.time:
        date = ocr.date or date
        user_date = date or metadata.user_start_date
        wall = parse_user_datetime(user_date, ocr.time)
        if wall is not None:
            source = "ocr_anchor"
            location = ocr.location or location
    if wall is None:
        wall = parse_user_datetime(metadata.user_start_date, metadata.user_start_time)
        if wall is not None:
            source = "user_fallback"
    if wall is not None:
        time_map.anchors.append(
            TimeAnchor(
                video_pts_ms=video_pts_ms,
                wall_time=format_wall_time(wall),
                source=source,
                ocr_confidence=(
                    ocr_confidence if source in ("ocr_anchor", "ocr_recalibrated") else None
                ),
                date=date,
                location=location,
            )
        )
    return time_map


def load_time_map(path: Path) -> TimeMap:
    """Load ``{stem}.time_map.json``."""
    if not path.is_file():
        raise FileNotFoundError(f"Time map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return TimeMap.model_validate(payload)


def save_time_map(path: Path, time_map: TimeMap) -> None:
    """Write ``{stem}.time_map.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(time_map.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
