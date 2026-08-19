"""ViAna CLI — prescan, run, resume, aggregate (Phase 1+ implementation)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from viana.config.defaults import load_engine_defaults
from viana.config.job import PROJECT_ID_PATTERN, JobConfig, load_job_config
from viana.io.paths import artifact_paths, project_output_dir
from viana.io.run_result import RunResult
from viana.stages.ocr import optional_easyocr_reader
from viana.stages.prescan import run_prescan

app = typer.Typer(
    name="viana",
    help="ViAna moving-count analytics engine (CLI-first).",
    no_args_is_help=True,
)


def _load_job_config_or_exit(config: Path) -> JobConfig:
    """Parse JobConfig JSON or exit 1 on missing/invalid files."""
    try:
        return load_job_config(config)
    except FileNotFoundError:
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(code=1) from None
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        typer.echo(f"Invalid JobConfig: {exc}", err=True)
        raise typer.Exit(code=1) from None


def _emit_run_result(result: RunResult) -> None:
    """Print RunResult JSON; non-COMPLETED jobs exit 1."""
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    if result.status != "COMPLETED":
        raise typer.Exit(code=1)


def _run_pipeline(job: JobConfig, *, resume: bool) -> None:
    """Execute the moving-count loop and print RunResult JSON."""
    from viana.stages.process import (
        CheckpointExistsError,
        MissingCheckpointError,
        run_moving_count,
    )

    try:
        result = run_moving_count(job, resume=resume)
    except (CheckpointExistsError, MissingCheckpointError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Run failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    _emit_run_result(result)


@app.command()
def prescan(
    source: Path = typer.Option(..., "--source", "-s", help="Absolute path to input video."),
    project_id: str = typer.Option(..., "--project-id", "-p", help="Project slug [a-z0-9_-]+."),
    frame_offset: float = typer.Option(
        0.0, "--frame-offset", help="Seconds into video for preview frame."
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Project output directory. Defaults to {parent_dir}/{project_id}.",
    ),
) -> None:
    """Sample video, OCR metadata, propose calibration lines, write preview JPEG."""
    if not PROJECT_ID_PATTERN.match(project_id):
        typer.echo("project_id must match [a-z0-9][a-z0-9_-]*", err=True)
        raise typer.Exit(code=1)
    if not source.is_file():
        typer.echo(f"Video not found: {source}", err=True)
        raise typer.Exit(code=1)
    resolved_output = output_dir
    if resolved_output is None:
        defaults = load_engine_defaults()
        resolved_output = project_output_dir(defaults.output.parent_dir, project_id)
    try:
        result = run_prescan(
            source,
            project_id,
            frame_offset_sec=frame_offset,
            output_dir=resolved_output,
            ocr_reader=optional_easyocr_reader(),
        )
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"Prescan failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to JobConfig JSON file."),
) -> None:
    """Run full moving-count pipeline (events CSV; no inline 15-min bins)."""
    job = _load_job_config_or_exit(config)
    if job.resume:
        typer.echo("Use `viana resume` when resume is true.", err=True)
        raise typer.Exit(code=1)
    _run_pipeline(job, resume=False)


@app.command()
def resume(
    config: Path = typer.Option(..., "--config", "-c", help="JobConfig JSON with resume intent."),
) -> None:
    """Resume from checkpoint (explicit trigger only)."""
    job = _load_job_config_or_exit(config)
    if not job.resume:
        typer.echo("viana resume requires resume=true in JobConfig.", err=True)
        raise typer.Exit(code=1)
    _run_pipeline(job, resume=True)


@app.command()
def aggregate(
    source: Path = typer.Option(
        ..., "--source", "-s", help="Source video path (stem locates events CSV)."
    ),
    project_id: str = typer.Option(..., "--project-id", "-p", help="Project slug."),
    partial: bool = typer.Option(False, "--partial", help="Allow aggregation on incomplete run."),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Project output directory. Defaults to {parent_dir}/{project_id}.",
    ),
) -> None:
    """Build 15-minute CSV from raw events (no inference)."""
    if not PROJECT_ID_PATTERN.match(project_id):
        typer.echo("project_id must match [a-z0-9][a-z0-9_-]*", err=True)
        raise typer.Exit(code=1)
    resolved_output = output_dir
    if resolved_output is None:
        defaults = load_engine_defaults()
        resolved_output = project_output_dir(defaults.output.parent_dir, project_id)
    paths = artifact_paths(resolved_output, source.stem)
    if not paths["events"].is_file():
        typer.echo(f"Events CSV not found: {paths['events']}", err=True)
        raise typer.Exit(code=1)
    from viana.config.classes import load_class_taxonomy
    from viana.stages.aggregate import aggregate_events

    try:
        rows = aggregate_events(
            paths["events"],
            paths["aggregate_15min"],
            load_class_taxonomy(),
            partial=partial,
            checkpoint_path=paths["checkpoint"],
        )
    except (OSError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"Aggregation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "command": "aggregate",
                "rows": len(rows),
                "events": str(paths["events"]),
                "aggregate_15min": str(paths["aggregate_15min"]),
                "partial": partial,
            },
            indent=2,
        )
    )


def main() -> None:
    """Console script entrypoint for the `viana` CLI."""
    app()


if __name__ == "__main__":
    main()
