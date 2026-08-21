import asyncio
from unittest.mock import MagicMock

from orchestrator.hub import TelemetryHub


def test_publish_loop_is_none() -> None:
    """Test publish returns early when loop is None."""
    hub = TelemetryHub()
    # No loop bound
    hub.publish({"msg": "test"})
    # Should not raise an error


def test_publish_loop_not_running() -> None:
    """Test publish returns early when loop is not running."""
    hub = TelemetryHub()
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_running.return_value = False

    hub.bind_loop(mock_loop)
    hub.publish({"msg": "test"})

    # call_soon_threadsafe should not be called
    mock_loop.call_soon_threadsafe.assert_not_called()


def test_publish_happy_path() -> None:
    """Test publish calls call_soon_threadsafe for each subscriber queue."""
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
