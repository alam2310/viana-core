"""Thread-safe WebSocket fan-out for engine stderr telemetry."""

from __future__ import annotations

import asyncio
from typing import Any


class TelemetryHub:
    """Publish telemetry dicts to connected ``/ws/jobs`` clients."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the FastAPI event loop so worker threads can enqueue."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a subscriber queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Drop a subscriber queue."""
        if queue in self._queues:
            self._queues.remove(queue)

    def publish(self, message: dict[str, Any]) -> None:
        """Enqueue a telemetry payload for all subscribers (thread-safe)."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        for queue in list(self._queues):
            loop.call_soon_threadsafe(queue.put_nowait, message)


hub = TelemetryHub()
