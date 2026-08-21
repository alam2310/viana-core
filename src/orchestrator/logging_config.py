"""Structured logging for the orchestrator."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def redact_pii_processor(
    _logger: logging.Logger, _name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Scrub sensitive PII fields (like location) from logs."""
    sensitive_keys = {"location", "user_start_time", "user_start_date"}

    def _redact(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: "***REDACTED***" if k in sensitive_keys else _redact(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [_redact(item) for item in data]
        return data

    return _redact(event_dict)  # type: ignore[no-any-return]


def configure_logging(*, json_logs: bool = False) -> None:
    """Configure structlog for development (console) or production (JSON).

    Args:
        json_logs: When True, emit JSON lines suitable for log aggregation.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        redact_pii_processor,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a module-scoped structured logger."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
