"""Disk preview JPEG registry — maps prescan_id to HTTP preview URLs."""

from __future__ import annotations

from pathlib import Path

from orchestrator.settings import output_parent

_PREVIEW_FILES: dict[str, Path] = {}


def register_preview(prescan_id: str, disk_path: Path) -> None:
    """Remember a preview JPEG path for later HTTP serving."""
    _PREVIEW_FILES[prescan_id] = disk_path


def preview_http_url(prescan_id: str) -> str:
    """Return the orchestrator-relative preview URL."""
    return f"/utils/prescan/{prescan_id}/preview.jpg"


def resolve_preview_path(prescan_id: str) -> Path | None:
    """Return the on-disk preview path when registered and still present."""
    path = _PREVIEW_FILES.get(prescan_id)
    if path is None or not path.is_file():
        return None
    resolved = path.resolve()
    if not resolved.is_relative_to(output_parent().resolve()):
        return None
    return resolved


def rewrite_preview_url(payload: dict[str, object]) -> dict[str, object]:
    """Replace disk ``preview_url`` with an HTTP path when possible."""
    prescan_id = payload.get("prescan_id")
    disk = payload.get("preview_url")
    if isinstance(prescan_id, str) and isinstance(disk, str) and disk:
        register_preview(prescan_id, Path(disk))
        payload = dict(payload)
        payload["preview_url"] = preview_http_url(prescan_id)
    return payload
