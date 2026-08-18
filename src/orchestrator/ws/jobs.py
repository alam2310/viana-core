"""WebSocket telemetry — engine stderr NDJSON bridged from the worker pool."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from orchestrator.hub import hub
from orchestrator.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])


@router.websocket("/ws/jobs")
async def jobs_telemetry(websocket: WebSocket) -> None:
    """Keep the socket open and forward TelemetryMessage payloads."""
    await websocket.accept()
    queue = hub.subscribe()
    logger.info("ws_jobs_connected")
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("ws_jobs_disconnected")
    finally:
        hub.unsubscribe(queue)
