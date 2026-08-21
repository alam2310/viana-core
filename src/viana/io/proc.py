"""Subprocess and FD helpers — close pipes and process groups on every exit."""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def open_fd_count() -> int | None:
    """Return this process's open FD count, or None when ``/proc`` is unavailable."""
    fd_dir = Path("/proc/self/fd")
    try:
        return len(os.listdir(fd_dir))
    except OSError:
        return None


def close_stdio(proc: Any) -> None:
    """Close stdin/stdout/stderr pipes if present (idempotent)."""
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, name, None)
        if stream is None:
            continue
        closer = getattr(stream, "close", None)
        if not callable(closer):
            continue
        try:
            closer()
        except OSError:
            pass


def terminate_process_tree(
    proc: Any,
    *,
    grace_sec: float = 2.0,
    close_pipes: bool = True,
) -> None:
    """SIGTERM then SIGKILL the session (or PID), then optionally close pipes.

    Workers are started with ``start_new_session=True`` so this does not signal
    the orchestrator. Call with ``close_pipes=False`` when another thread is
    still draining stdio.
    """
    if getattr(proc, "poll", None) is not None:
        try:
            if proc.poll() is not None:
                if close_pipes:
                    close_stdio(proc)
                return
        except OSError:
            pass
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0:
        _signal_group(pid, signal.SIGTERM)
    else:
        _call(proc, "terminate")
    if not _wait(proc, grace_sec):
        if isinstance(pid, int) and pid > 0:
            _signal_group(pid, signal.SIGKILL)
        else:
            _call(proc, "kill")
        _wait(proc, grace_sec)
    if close_pipes:
        close_stdio(proc)


def popen_session(
    args: Sequence[str],
    *,
    stdin: Any = subprocess.DEVNULL,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    text: bool = True,
    bufsize: int = 1,
) -> subprocess.Popen[str]:
    """``Popen`` in a new session so timeout/cancel can kill grandchildren."""
    return subprocess.Popen(  # noqa: S603  # nosec B603
        list(args),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
        bufsize=bufsize,
        start_new_session=True,
    )


def run_captured(
    args: Sequence[str],
    *,
    timeout: float | None,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` equivalent that always closes pipes and kills the group."""
    proc = popen_session(args, text=text, bufsize=-1 if not text else 1)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc, close_pipes=True)
            raise
        return subprocess.CompletedProcess(
            args=list(args),
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    finally:
        close_stdio(proc)


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _call(proc: Any, method: str) -> None:
    func = getattr(proc, method, None)
    if not callable(func):
        return
    try:
        func()
    except OSError:
        pass


def _wait(proc: Any, timeout: float) -> bool:
    wait = getattr(proc, "wait", None)
    if not callable(wait):
        return True
    try:
        wait(timeout=timeout)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False
    except TypeError:
        try:
            wait()
            return True
        except OSError:
            return False
