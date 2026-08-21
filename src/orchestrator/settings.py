"""Orchestrator runtime settings (output root and intake mounts)."""

from __future__ import annotations

import os
from pathlib import Path

from viana.config.defaults import load_engine_defaults

CONTAINER_DATA_ROOT = "/data"
CONTAINER_REPO_ROOT = "/app/ViAna"
DEFAULT_INTAKE_ROOTS = (CONTAINER_DATA_ROOT, CONTAINER_REPO_ROOT)


def output_parent() -> Path:
    """Return artifact root: ``VIANA_OUTPUT_PARENT`` or engine defaults."""
    env = os.environ.get("VIANA_OUTPUT_PARENT")
    if env:
        return Path(env)
    return load_engine_defaults().output.parent_dir


def project_dir(project_id: str) -> Path:
    """Return ``{output_parent}/{project_id}/``."""
    from viana.io.paths import project_output_dir

    return project_output_dir(output_parent(), project_id)


def resolve_output_dir(project_id: str, override: Path | None = None) -> Path:
    """Return project output dir, optionally overridden per intake/submit."""
    if override is not None:
        return override
    return project_dir(project_id)


def _posix(path: str) -> str:
    text = path.replace("\\", "/").strip()
    if text and text != "/":
        text = text.rstrip("/")
    return text


def intake_roots() -> tuple[str, ...]:
    """Container prefixes the API may read (``VIANA_INTAKE_ROOTS``, colon-separated)."""
    raw = os.environ.get("VIANA_INTAKE_ROOTS", "").strip()
    if not raw:
        return DEFAULT_INTAKE_ROOTS
    parts = tuple(_posix(item) for item in raw.split(":") if item.strip())
    return parts or DEFAULT_INTAKE_ROOTS


def _absolute_host_prefix(prefix: str, host_repo: str) -> str:
    """Resolve a possibly relative host prefix against the host repo root."""
    normalized = _posix(prefix)
    if normalized.startswith("/"):
        return normalized
    if host_repo:
        return _posix(str(Path(host_repo) / prefix))
    return normalized


def intake_path_maps() -> tuple[tuple[str, str], ...]:
    """Host→container prefix pairs for intake rewrite (longest host wins)."""
    host_repo = _posix(os.environ.get("VIANA_HOST_REPO_ROOT", "").strip())
    host_data = os.environ.get("VIANA_HOST_DATA_ROOT", "").strip()
    extra = os.environ.get("VIANA_PATH_MAPS", "").strip()
    maps: list[tuple[str, str]] = []
    if host_data:
        maps.append((_absolute_host_prefix(host_data, host_repo), CONTAINER_DATA_ROOT))
    if host_repo:
        maps.append((host_repo, CONTAINER_REPO_ROOT))
    if extra:
        for pair in extra.split(";"):
            item = pair.strip()
            if not item or "->" not in item:
                continue
            host, container = item.split("->", 1)
            host_n = _posix(host.strip())
            container_n = _posix(container.strip())
            if host_n and container_n:
                maps.append((host_n, container_n))
    return tuple(maps)
