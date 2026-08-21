"""S22 — subprocess pipes and process groups must not leak FDs."""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from viana.io.proc import close_stdio, open_fd_count, run_captured, terminate_process_tree


def test_run_captured_loop_does_not_grow_fds() -> None:
    """Repro loop: repeated short subprocesses leave FD count stable."""
    baseline = open_fd_count()
    if baseline is None:
        return
    for _ in range(8):
        result = run_captured(["python3", "-c", "print('ok')"], timeout=5.0)
        assert result.returncode == 0
        assert "ok" in result.stdout
    after = open_fd_count()
    assert after is not None
    assert after <= baseline + 4


def test_terminate_process_tree_closes_pipes() -> None:
    """Killing a piped child must drop stdout/stderr FDs."""
    sleep = shutil.which("sleep")
    if sleep is None:
        pytest.skip("sleep not on PATH")
    baseline = open_fd_count()
    if baseline is None:
        return
    for _ in range(4):
        proc = subprocess.Popen(  # noqa: S603  # nosec B603
            [sleep, "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        terminate_process_tree(proc, close_pipes=True)
        assert proc.poll() is not None
        close_stdio(proc)
    time.sleep(0.05)
    after = open_fd_count()
    assert after is not None
    assert after <= baseline + 4
