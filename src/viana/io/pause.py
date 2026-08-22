"""Cooperative operator pause — SIGINT sets a flag checked between frames."""

from __future__ import annotations

import signal

_pause_requested = False


def install_cooperative_pause_handlers() -> None:
    """Arm SIGINT to request pause without relying on interrupt during native GPU work."""
    global _pause_requested
    _pause_requested = False
    signal.signal(signal.SIGINT, _on_sigint)


def _on_sigint(signum: int, frame: object | None) -> None:
    del signum, frame
    signal_pause_pending()


def signal_pause_pending() -> None:
    """Mark that the operator requested pause (SIGINT handler or tests)."""
    global _pause_requested
    _pause_requested = True


def consume_pause_request() -> None:
    """Raise ``KeyboardInterrupt`` when a pause was requested since the last frame."""
    if _pause_requested:
        raise KeyboardInterrupt
