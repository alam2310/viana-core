"""Tests for the orchestrator telemetry hub."""

import asyncio
import time

from orchestrator.hub import TelemetryHub


def test_telemetry_hub() -> None:
    """Test telemetry hub subscribe, unsubscribe, and publish."""

    async def _test() -> None:
        hub = TelemetryHub()
        loop = asyncio.get_running_loop()
        hub.bind_loop(loop)

        q1 = hub.subscribe()
        q2 = hub.subscribe()

        assert len(hub._queues) == 2

        message = {"status": "running"}
        hub.publish(message)

        # Poll briefly instead of sleeping blindly
        start_time = time.monotonic()
        while q1.empty() and time.monotonic() - start_time < 2.0:
            await asyncio.sleep(0.01)

        msg1 = q1.get_nowait()
        msg2 = q2.get_nowait()

        assert msg1 == message
        assert msg2 == message

        hub.unsubscribe(q1)
        assert len(hub._queues) == 1

        # Safe to unsubscribe if not in list
        hub.unsubscribe(q1)
        assert len(hub._queues) == 1

        hub.unsubscribe(q2)
        assert len(hub._queues) == 0

    asyncio.run(_test())


def test_publish_no_loop() -> None:
    """Test publish safely returns when no loop is bound."""

    async def _test() -> None:
        hub = TelemetryHub()
        # Bind no loop intentionally

        q1 = hub.subscribe()

        # Loop is not bound, publish should silently return
        hub.publish({"status": "running"})

        # Queue should be empty
        assert q1.empty()

    asyncio.run(_test())
