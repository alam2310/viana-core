"""Spawn ``python -m viana`` — no CV in this module."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from collections.abc import Callable, Sequence

from viana.io.proc import close_stdio, popen_session, terminate_process_tree

SpawnHook = Callable[[subprocess.Popen[str]], None]


def viana_command(args: Sequence[str]) -> list[str]:
    """Build ``python -m viana …`` argv (list form, never shell)."""
    return [sys.executable, "-m", "viana", *args]


def run_viana(
    args: Sequence[str],
    *,
    timeout: float | None = 120.0,
    on_spawn: SpawnHook | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a short CLI command (prescan / aggregate) and capture stdio.

    Always closes pipes. On timeout, kills the process group (OpenCV/ffprobe
    grandchildren) so FDs cannot leak into the orchestrator after ``TimeoutExpired``.
    """
    proc = start_viana_process(args)
    if on_spawn is not None:
        on_spawn(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, close_pipes=True)
            raise
        return subprocess.CompletedProcess(
            args=viana_command(args),
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    finally:
        close_stdio(proc)


def start_viana_process(args: Sequence[str]) -> subprocess.Popen[str]:
    """Start ``viana run`` / ``viana resume`` with streamed stdio in a new session."""
    return popen_session(viana_command(args), bufsize=1)
