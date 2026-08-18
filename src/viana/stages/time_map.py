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
IGNORE_WORDS = {
    "mon",
    "tue",
    "wed",
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


def parse_ocr_texts(texts: list[str]) -> ParsedOcr:
    """Extract time/date/location from OCR strings (legacy TimeSyncEngine.extract_metadata)."""
    parsed_time: str | None = None
    date_str: str | None = None
    location_parts: list[str] = []
    for raw in texts:
        text_clean = raw.strip()
        time_match = TIME_PATTERN.search(text_clean)
        if time_match:
            parsed_time = time_match.group()
            text_clean = TIME_PATTERN.sub("", text_clean).strip()
        date_match = DATE_PATTERN.search(text_clean)
        if date_match:
            date_str = date_match.group()
            text_clean = DATE_PATTERN.sub("", text_clean).strip()
        if text_clean and text_clean.lower() not in IGNORE_WORDS:
            loc_cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", text_clean)
            if loc_cleaned:
                location_parts.append(loc_cleaned)
    return ParsedOcr(
        time=parsed_time,
        date=date_str,
        location=" ".join(location_parts) if location_parts else None,
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
