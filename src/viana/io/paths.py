"""Output path resolution under configured parent directory."""

from __future__ import annotations

from pathlib import Path


def project_output_dir(parent_dir: Path, project_id: str) -> Path:
    return parent_dir / project_id


def artifact_paths(output_dir: Path, video_stem: str) -> dict[str, Path]:
    """Standard artifact filenames for a processed video."""
    return {
        "events": output_dir / f"{video_stem}_events.csv",
        "aggregate_15min": output_dir / f"{video_stem}_15min.csv",
        "processed_video": output_dir / f"{video_stem}_processed.mp4",
        "manifest": output_dir / f"{video_stem}.manifest.json",
        "time_map": output_dir / f"{video_stem}.time_map.json",
        "checkpoint": output_dir / f"{video_stem}.checkpoint.json",
        "run_result": output_dir / f"{video_stem}.run_result.json",
    }


def profiles_dir(output_dir: Path) -> Path:
    return output_dir / "profiles"


def prescan_dir(output_dir: Path) -> Path:
    return output_dir / "prescan"
