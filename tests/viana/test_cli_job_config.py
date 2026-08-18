"""CLI JobConfig validation (pipeline still not implemented)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from viana.cli import app
from viana.config.files import repo_root

runner = CliRunner()


def test_run_validates_job_config_then_not_implemented() -> None:
    """viana run accepts a valid JobConfig and exits 2 until Phase 3."""
    fixture = repo_root() / "packages" / "contracts" / "fixtures" / "job_config.json"
    result = runner.invoke(app, ["run", "--config", str(fixture)])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_implemented"
    assert payload["job_id"] == "job_mock_001"
    assert payload["command"] == "run"


def test_run_rejects_missing_config(tmp_path: Path) -> None:
    """Missing config files exit 1."""
    result = runner.invoke(app, ["run", "--config", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "not found" in result.stderr.lower()


def test_run_rejects_resume_flag(tmp_path: Path) -> None:
    """Explicit resume belongs on viana resume."""
    src = repo_root() / "packages" / "contracts" / "fixtures" / "job_config.json"
    payload = json.loads(src.read_text())
    payload["resume"] = True
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(path)])
    assert result.exit_code == 1
    assert "resume" in result.stderr.lower()


def test_resume_requires_resume_true() -> None:
    """viana resume refuses JobConfig with resume=false."""
    fixture = repo_root() / "packages" / "contracts" / "fixtures" / "job_config.json"
    result = runner.invoke(app, ["resume", "--config", str(fixture)])
    assert result.exit_code == 1
    assert "resume=true" in result.stderr
