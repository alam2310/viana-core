"""Output path resolution under configured parent directory.

S29 / ADR 003: operator deliverables stay flat under ``{project}/``;
resume and orchestrator sidecars live under ``{project}/_meta/``.
Legacy flat ``{stem}.checkpoint.json`` (etc.) remain readable for PAUSED resume.
"""

from __future__ import annotations

from pathlib import Path

# Non-deliverable keys stored under ``_meta/{stem}/``.
_META_KEYS = frozenset({"manifest", "time_map", "checkpoint", "run_result"})

_LEGACY_META_SUFFIX: dict[str, str] = {
    "manifest": ".manifest.json",
    "time_map": ".time_map.json",
    "checkpoint": ".checkpoint.json",
    "run_result": ".run_result.json",
}


def project_output_dir(parent_dir: Path, project_id: str) -> Path:
    """Resolve the per-project output directory.

    Args:
        parent_dir: Configured artifact root (e.g. `/data/viana-outputs`).
        project_id: Project slug validated by `JobSubmitRequest`.

    Returns:
        Absolute path `{parent_dir}/{project_id}/`.
    """
    return parent_dir / project_id


def stem_meta_dir(output_dir: Path, video_stem: str) -> Path:
    """Return ``{output_dir}/_meta/{video_stem}/`` (checkpoint / time_map / run_result)."""
    return output_dir / "_meta" / video_stem


def jobs_meta_dir(output_dir: Path) -> Path:
    """Return ``{output_dir}/_meta/jobs/`` for orchestrator job JSON."""
    return output_dir / "_meta" / "jobs"


def job_config_path(output_dir: Path, job_id: str) -> Path:
    """Canonical path for ``{job_id}.job.json`` under ``_meta/jobs/``."""
    return jobs_meta_dir(output_dir) / f"{job_id}.job.json"


def artifact_paths(output_dir: Path, video_stem: str) -> dict[str, Path]:
    """Build standard artifact paths for a processed video stem.

    Operator deliverables stay at the project root. Resume / audit sidecars
    go under ``_meta/{stem}/`` (ADR 003).

    Args:
        output_dir: Project output directory.
        video_stem: Filename without extension.

    Returns:
        Mapping of logical artifact names to absolute paths (canonical write targets).
    """
    meta = stem_meta_dir(output_dir, video_stem)
    return {
        "events": output_dir / f"{video_stem}_events.csv",
        "events_report": output_dir / f"{video_stem}_events_report.csv",
        "aggregate_15min": output_dir / f"{video_stem}_15min.csv",
        "processed_video": output_dir / f"{video_stem}_processed.mp4",
        "manifest": meta / "manifest.json",
        "time_map": meta / "time_map.json",
        "checkpoint": meta / "checkpoint.json",
        "run_result": meta / "run_result.json",
    }


def legacy_artifact_paths(output_dir: Path, video_stem: str) -> dict[str, Path]:
    """Pre-S29 flat sidecar paths (compat for existing PAUSED checkpoints)."""
    return {
        key: output_dir / f"{video_stem}{suffix}" for key, suffix in _LEGACY_META_SUFFIX.items()
    }


def resolve_artifact(output_dir: Path, video_stem: str, key: str) -> Path:
    """Path for reading ``key``: canonical if present, else legacy flat, else canonical.

    Deliverables (events / 15min / processed) have no legacy alternate.
    Incomplete checkpoints under the legacy flat path must remain findable for
    Step 6.2 PAUSED resume — never require a mass migrate before resume.
    """
    paths = artifact_paths(output_dir, video_stem)
    canonical = paths[key]
    if key not in _META_KEYS:
        return canonical
    if canonical.is_file():
        return canonical
    legacy = legacy_artifact_paths(output_dir, video_stem)[key]
    if legacy.is_file():
        return legacy
    return canonical


def wipe_run_sidecars(output_dir: Path, video_stem: str) -> None:
    """Delete deliverables + meta sidecars for ``start_fresh`` (canonical and legacy)."""
    paths = artifact_paths(output_dir, video_stem)
    legacy = legacy_artifact_paths(output_dir, video_stem)
    for key in ("events", "events_report", "processed_video", "aggregate_15min"):
        target = paths[key]
        if target.is_file():
            target.unlink()
    for key in _META_KEYS:
        for target in (paths[key], legacy[key]):
            if target.is_file():
                target.unlink()


def profiles_dir(output_dir: Path) -> Path:
    """Return the calibration profiles directory for a project."""
    return output_dir / "profiles"


def prescan_dir(output_dir: Path) -> Path:
    """Return the prescan preview directory for a project."""
    return output_dir / "prescan"
