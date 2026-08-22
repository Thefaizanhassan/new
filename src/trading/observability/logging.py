"""Structured logging.

``logger.info("order_rejected", rule="max_exposure", observed=0.72, limit=0.60)``
is greppable and analysable; a formatted string is neither.  Phase 0 §16.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        # UTC everywhere, including in logs — a log in local time is a log you
        # cannot line up against an exchange session.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
