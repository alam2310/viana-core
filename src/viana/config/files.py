"""Locate committed YAML config files."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_CONFIGS_DIR = "VIANA_CONFIGS_DIR"


def repo_root() -> Path:
    """Return the monorepo root (parent of ``src/``) from this package path."""
    return Path(__file__).resolve().parents[3]


def repo_configs_dir() -> Path:
    """Return the configs directory next to this source tree when present.

    Layout: ``<repo>/src/viana/config/files.py`` → ``<repo>/configs``.
    """
    return repo_root() / "configs"


def contracts_schemas_dir() -> Path:
    """Return ``packages/contracts/schemas`` under the monorepo root."""
    return repo_root() / "packages" / "contracts" / "schemas"


def resolve_configs_dir(explicit: Path | None = None) -> Path:
    """Resolve the directory that holds ``classes.yaml`` and ``engine_defaults.yaml``.

    Lookup order: ``explicit`` → ``VIANA_CONFIGS_DIR`` → repo ``configs/`` →
    ``./configs`` under the current working directory.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_dir = os.environ.get(_ENV_CONFIGS_DIR)
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_configs_dir())
    candidates.append(Path.cwd() / "configs")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        f"Could not find a configs directory. Pass an explicit path or set {_ENV_CONFIGS_DIR}."
    )
