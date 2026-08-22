"""Tests for calibration profile I/O and utilities."""

from datetime import datetime, timezone

from viana.io.profiles import parse_created_at


def test_parse_created_at_missing() -> None:
    """Missing or empty values sort as oldest."""
    expected = datetime.min.replace(tzinfo=timezone.utc)
    assert parse_created_at(None) == expected
    assert parse_created_at("") == expected


def test_parse_created_at_valid() -> None:
    """Valid ISO strings are parsed and converted to UTC."""
    # With 'Z' suffix
    dt_z = parse_created_at("2023-10-10T10:00:00Z")
    assert dt_z.tzinfo == timezone.utc
    assert dt_z.year == 2023
    assert dt_z.month == 10
    assert dt_z.day == 10
    assert dt_z.hour == 10

    # With numeric offset
    dt_offset = parse_created_at("2023-10-10T10:00:00+02:00")
    assert dt_offset.tzinfo == timezone.utc
    assert dt_offset.hour == 8  # Converted to UTC

    # Without timezone (assumes UTC)
    dt_naive = parse_created_at("2023-10-10T10:00:00")
    assert dt_naive.tzinfo == timezone.utc
    assert dt_naive.hour == 10
