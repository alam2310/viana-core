"""ViAna CLI — prescan, run, resume, aggregate (Phase 1+ implementation)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="viana",
    help="ViAna moving-count analytics engine (CLI-first).",
    no_args_is_help=True,
)


@app.command()
def prescan(
    source: Path = typer.Option(..., "--source", "-s", help="Absolute path to input video."),
    project_id: str = typer.Option(..., "--project-id", "-p", help="Project slug [a-z0-9_-]+."),
    frame_offset: float = typer.Option(
        0.0, "--frame-offset", help="Seconds into video for preview frame."
    ),
) -> None:
    """Sample video, OCR metadata, propose calibration lines (Phase 4)."""
    typer.echo(
        json.dumps(
            {
                "status": "not_implemented",
                "phase": 0,
                "command": "prescan",
                "source": str(source),
                "project_id": project_id,
                "frame_offset": frame_offset,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=2)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to JobConfig JSON file."),
) -> None:
    """Run full moving-count pipeline (Phase 3+)."""
    if not config.is_file():
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        json.dumps({"status": "not_implemented", "phase": 0, "config": str(config)}, indent=2)
    )
    raise typer.Exit(code=2)


@app.command()
def resume(
    config: Path = typer.Option(..., "--config", "-c", help="JobConfig JSON with resume intent."),
) -> None:
    """Resume from checkpoint (explicit trigger only)."""
    typer.echo(
        json.dumps({"status": "not_implemented", "phase": 0, "config": str(config)}, indent=2)
    )
    raise typer.Exit(code=2)


@app.command()
def aggregate(
    source: Path = typer.Option(
        ..., "--source", "-s", help="Source video path (stem locates events CSV)."
    ),
    project_id: str = typer.Option(..., "--project-id", "-p", help="Project slug."),
    partial: bool = typer.Option(False, "--partial", help="Allow aggregation on incomplete run."),
) -> None:
    """Build 15-minute CSV from raw events (Phase 5)."""
    typer.echo(
        json.dumps(
            {
                "status": "not_implemented",
                "phase": 0,
                "source": str(source),
                "project_id": project_id,
                "partial": partial,
            },
            indent=2,
        )
    )
    raise typer.Exit(code=2)


def main() -> None:
    """Console script entrypoint for the `viana` CLI."""
    app()


if __name__ == "__main__":
    main()
