"""Tests for the orchestrator telemetry hub."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from orchestrator.hub import TelemetryHub


def test_telemetry_hub_subscribe_publish_unsubscribe() -> None:
    """Subscribe, publish to all queues, and unsubscribe safely."""

    async def _test() -> None:
        hub = TelemetryHub()
        hub.bind_loop(asyncio.get_running_loop())

        q1 = hub.subscribe()
        q2 = hub.subscribe()
        assert len(hub._queues) == 2

        message = {"status": "running"}
        hub.publish(message)

        start_time = time.monotonic()
        while q1.empty() and time.monotonic() - start_time < 2.0:
            await asyncio.sleep(0.01)

        assert q1.get_nowait() == message
        assert q2.get_nowait() == message

        hub.unsubscribe(q1)
        assert len(hub._queues) == 1
        hub.unsubscribe(q1)  # idempotent
        assert len(hub._queues) == 1
        hub.unsubscribe(q2)
        assert len(hub._queues) == 0

    asyncio.run(_test())


def test_publish_no_loop() -> None:
    """Publish is a no-op when no loop is bound."""
    hub = TelemetryHub()
    q1 = hub.subscribe()
    hub.publish({"status": "running"})
    assert q1.empty()


def test_publish_loop_not_running() -> None:
    """Publish returns early when the bound loop is not running."""
    hub = TelemetryHub()
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = False
    hub.bind_loop(mock_loop)
    hub.publish({"msg": "test"})
    mock_loop.call_soon_threadsafe.assert_not_called()


def test_publish_happy_path_threadsafe() -> None:
    """Publish schedules put_nowait on the bound running loop for each queue."""
    hub = TelemetryHub()
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = True
    hub.bind_loop(mock_loop)

    q1 = hub.subscribe()
    q2 = hub.subscribe()
    hub.publish({"msg": "test"})

    assert mock_loop.call_soon_threadsafe.call_count == 2
    mock_loop.call_soon_threadsafe.assert_any_call(q1.put_nowait, {"msg": "test"})
    mock_loop.call_soon_threadsafe.assert_any_call(q2.put_nowait, {"msg": "test"})

    hub.unsubscribe(q1)
    mock_loop.reset_mock()
    hub.publish({"msg": "test2"})
    assert mock_loop.call_soon_threadsafe.call_count == 1
    mock_loop.call_soon_threadsafe.assert_any_call(q2.put_nowait, {"msg": "test2"})
