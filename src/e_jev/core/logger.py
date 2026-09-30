"""core.logger: structlog configured once, at the composition root — PROD JSON via msgspec, DEV flat text."""

import logging
import sys
from typing import Any, Final, Literal

import msgspec
import structlog
from structlog.typing import Processor

SHARED: Final[tuple[Processor, ...]] = (
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
)


def encode_json(value: Any, **_: Any) -> bytes:
    """msgspec as structlog's serializer; structlog's stdlib-json kwargs (`default`) do not apply."""
    return msgspec.json.encode(value)


def setup_logger(env: Literal["prod", "dev"], level: str) -> None:
    match env:
        case "prod":
            renderer: Processor = structlog.processors.JSONRenderer(serializer=encode_json)
            factory = structlog.BytesLoggerFactory(file=sys.stdout.buffer)
        case "dev":
            renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
            factory = structlog.PrintLoggerFactory(file=sys.stdout)
    structlog.configure(
        processors=[*SHARED, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=factory,
        cache_logger_on_first_use=True,
    )
