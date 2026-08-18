"""WebSocket telemetry stub. Payloads match telemetry.schema.json."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from orchestrator.logging_config import get_logger
from orchestrator.models import TelemetryMessage

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])

ENGINE_STUB_LOG = "Telemetry bridge not implemented; requires Phase 5 python -m viana run stdout."


@router.websocket("/ws/jobs")
async def jobs_telemetry(websocket: WebSocket) -> None:
    """Accept a telemetry subscription and close; engine stdout bridge is Phase 5+."""
    await websocket.accept()
    logger.info("ws_jobs_stub")
    message = TelemetryMessage(
        job_id="",
        status="PENDING",
        telemetry_type="LOG",
        data={
            "message": ENGINE_STUB_LOG,
        },
    )
    await websocket.send_json(message.model_dump())
    await websocket.close(code=1001)
