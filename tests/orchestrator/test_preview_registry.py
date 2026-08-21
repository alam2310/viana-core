"""Tests for preview registry."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator.preview_registry import (
    _PREVIEW_FILES,
    register_preview,
    resolve_preview_path,
)


@pytest.fixture(autouse=True)
def setup_preview_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear registry and set output_parent to tmp_path before each test."""
    _PREVIEW_FILES.clear()
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(tmp_path))
    yield
    _PREVIEW_FILES.clear()


def test_resolve_from_registry_success(tmp_path: Path) -> None:
    """Resolve an existing file from the registry successfully."""
    prescan_id = "test_id_1"
    preview_file = tmp_path / "test_id_1_preview.jpg"
    preview_file.write_bytes(b"image data")

    register_preview(prescan_id, preview_file)

    resolved = resolve_preview_path(prescan_id)
    assert resolved == preview_file.resolve()


def test_resolve_registry_file_missing(tmp_path: Path) -> None:
    """Return None if registered file does not exist on disk."""
    prescan_id = "test_id_2"
    preview_file = tmp_path / "test_id_2_preview.jpg"

    register_preview(prescan_id, preview_file)

    # File is not created, so it falls back to search and fails.
    resolved = resolve_preview_path(prescan_id)
    assert resolved is None


def test_resolve_not_relative_to_output_parent(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Reject a file outside of the output_parent directory."""
    prescan_id = "test_id_3"
    outside_dir = tmp_path_factory.mktemp("outside_dir")
    preview_file = outside_dir / "test_id_3_preview.jpg"
    preview_file.write_bytes(b"image data")

    register_preview(prescan_id, preview_file)

    # It rejects the registry hit, then fallback searches tmp_path and finds nothing.
    resolved = resolve_preview_path(prescan_id)
    assert resolved is None


def test_resolve_fallback_search_success(tmp_path: Path) -> None:
    """Search and find a file not in registry but present in output_parent."""
    prescan_id = "test_id_4"
    sub_dir = tmp_path / "some_project"
    sub_dir.mkdir()
    preview_file = sub_dir / "test_id_4_preview.jpg"
    preview_file.write_bytes(b"image data")

    # Not registered yet
    assert prescan_id not in _PREVIEW_FILES

    resolved = resolve_preview_path(prescan_id)
    assert resolved == preview_file.resolve()

    # Should be registered now
    assert prescan_id in _PREVIEW_FILES
    assert _PREVIEW_FILES[prescan_id] == preview_file


def test_resolve_fallback_search_not_found(tmp_path: Path) -> None:
    """Return None if file not in registry and not found on disk."""
    prescan_id = "test_id_5"
    resolved = resolve_preview_path(prescan_id)
    assert resolved is None


def test_resolve_fallback_search_symlink_outside(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Reject a symlinked file found in search if it points outside output_parent."""
    prescan_id = "test_id_6"
    outside_dir = tmp_path_factory.mktemp("outside_dir_symlink")
    real_file = outside_dir / "test_id_6_preview.jpg"
    real_file.write_bytes(b"image data")

    symlink_path = tmp_path / "test_id_6_preview.jpg"
    os.symlink(real_file, symlink_path)

    resolved = resolve_preview_path(prescan_id)
    assert resolved is None


def test_resolve_output_parent_not_a_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Return None safely if output_parent is not a directory."""
    non_existent_dir = tmp_path / "missing_dir"
    monkeypatch.setenv("VIANA_OUTPUT_PARENT", str(non_existent_dir))

    prescan_id = "test_id_7"
    resolved = resolve_preview_path(prescan_id)
    assert resolved is None
