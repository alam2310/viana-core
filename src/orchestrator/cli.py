"""Spawn ``python -m viana`` — no CV in this module."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from collections.abc import Sequence


def viana_command(args: Sequence[str]) -> list[str]:
    """Build ``python -m viana …`` argv (list form, never shell)."""
    return [sys.executable, "-m", "viana", *args]


def run_viana(
    args: Sequence[str], *, timeout: float | None = 120.0
) -> subprocess.CompletedProcess[str]:
    """Run a short CLI command (prescan / aggregate) and capture stdio."""
    return subprocess.run(  # noqa: S603  # nosec B603
        viana_command(args),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def start_viana_process(args: Sequence[str]) -> subprocess.Popen[str]:
    """Start ``viana run`` / ``viana resume`` with streamed stdio."""
    return subprocess.Popen(  # noqa: S603  # nosec B603
        viana_command(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
