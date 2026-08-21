"""Normalize and reject intake paths the processing container cannot read.

Mirrors host→container translation in ``apps/web/src/lib/container-paths.ts``.
UI already maps browsed host paths; this defends direct API clients (S09 / F006).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from orchestrator.settings import intake_path_maps, intake_roots

INTAKE_PATH_HINT = (
    "Videos must live under a bind-mount. Default docker-compose: host ./data → "
    "/data, repo → /app/ViAna. Mount the drive and add the container path to "
    "VIANA_INTAKE_ROOTS (optional VIANA_PATH_MAPS=host->container)."
)


class IntakePathError(ValueError):
    """Intake path is outside configured mounts and cannot be rewritten."""


def posix_norm(path: Path | str) -> str:
    """Return a POSIX path without a trailing slash (except root)."""
    text = str(path).replace("\\", "/").strip()
    if not text:
        return text
    if text != "/":
        text = text.rstrip("/")
    return text


def is_under_root(path: str, root: str) -> bool:
    """Return True if ``path`` is ``root`` or a descendant."""
    normalized = posix_norm(path)
    base = posix_norm(root)
    if not normalized or not base:
        return False
    return normalized == base or normalized.startswith(f"{base}/")


def is_under_any_root(path: str, roots: Sequence[str]) -> bool:
    """Return True if ``path`` is under any intake root."""
    return any(is_under_root(path, root) for root in roots)


def apply_host_maps(path: str, maps: Sequence[tuple[str, str]]) -> str:
    """Rewrite a host prefix to the matching container prefix (longest host first)."""
    normalized = posix_norm(path)
    sorted_maps = sorted(maps, key=lambda item: len(posix_norm(item[0])), reverse=True)
    for host, container in sorted_maps:
        host_n = posix_norm(host)
        container_n = posix_norm(container)
        if not host_n:
            continue
        if normalized == host_n or normalized.startswith(f"{host_n}/"):
            suffix = normalized[len(host_n) :]
            return posix_norm(container_n + suffix)
    return normalized


def suffix_rewrite(path: str, roots: Sequence[str]) -> str | None:
    """Map ``.../<root-name>/rest`` onto ``<root>/rest``, preferring the later match."""
    normalized = posix_norm(path)
    best_idx = -1
    best: str | None = None
    for root in roots:
        base = posix_norm(root)
        name = Path(base).name
        if not name:
            continue
        marker = f"/{name}/"
        idx = normalized.rfind(marker)
        if idx == -1:
            continue
        after = normalized[idx + len(marker) - 1 :]  # "/rest..."
        candidate = posix_norm(base + after)
        if idx > best_idx:
            best_idx = idx
            best = candidate
    return best


def relative_to_root(path: str, roots: Sequence[str]) -> str | None:
    """Map ``data/raw/clip.mp4`` onto ``/data/raw/clip.mp4`` when ``data`` is a root name."""
    normalized = posix_norm(path)
    if normalized.startswith("/"):
        return None
    for root in sorted((posix_norm(item) for item in roots), key=len, reverse=True):
        name = Path(root).name
        if normalized == name or normalized.startswith(f"{name}/"):
            suffix = normalized[len(name) :]
            return posix_norm(root + suffix)
    return None


def resolve_intake_path(
    raw: Path | str,
    *,
    roots: Sequence[str] | None = None,
    maps: Sequence[tuple[str, str]] | None = None,
) -> Path:
    """Return a container-readable path, or raise ``IntakePathError``."""
    allowed = tuple(posix_norm(item) for item in (roots if roots is not None else intake_roots()))
    host_maps = maps if maps is not None else intake_path_maps()
    original = posix_norm(raw)
    if not original:
        raise IntakePathError(f"Intake path is empty. {INTAKE_PATH_HINT}")

    relative = relative_to_root(original, allowed)
    if relative is not None:
        original = relative

    mapped = apply_host_maps(original, host_maps)
    rewritten = suffix_rewrite(original, allowed)
    candidates: list[str] = []
    for item in (mapped, rewritten, original):
        if item and item not in candidates:
            candidates.append(item)

    for item in candidates:
        if is_under_any_root(item, allowed) and Path(item).is_file():
            return Path(item)
    if Path(original).is_file():
        return Path(original)
    for item in candidates:
        if is_under_any_root(item, allowed):
            return Path(item)

    display = str(raw) if str(raw) != mapped else mapped
    extra = f" (normalized {mapped})" if mapped != original else ""
    raise IntakePathError(
        f"Intake path is not readable inside the processing container: {display}{extra}. "
        f"{INTAKE_PATH_HINT}"
    )


def resolve_intake_paths(
    paths: Sequence[Path | str],
    *,
    roots: Sequence[str] | None = None,
    maps: Sequence[tuple[str, str]] | None = None,
) -> list[Path]:
    """Resolve every path, failing the whole batch if any path is unreadable."""
    resolved: list[Path] = []
    errors: list[str] = []
    for path in paths:
        try:
            resolved.append(resolve_intake_path(path, roots=roots, maps=maps))
        except IntakePathError as exc:
            errors.append(str(exc))
    if errors:
        raise IntakePathError(" ".join(errors))
    return resolved
