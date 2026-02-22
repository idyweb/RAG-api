import logging
import sys
from contextvars import ContextVar
from typing import Optional

from api.core.config import settings

# Context variable for correlation ID (for distributed tracing)
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def _resolve_log_level(level: str) -> int:
    """Convert string log level to logging constant."""
    level_name = level.upper()
    return getattr(logging, level_name, logging.INFO)


class CorrelationIdFilter(logging.Filter):
    """Attach the active correlation ID to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_context.get() or "N/A"
        return True


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure structured logging with correlation ID support."""

    log_level = level or settings.LOG_LEVEL
    service_name = settings.APP_NAME

    logger = logging.getLogger(service_name)
    logger.setLevel(_resolve_log_level(log_level))
    logger.propagate = False

    # Prevent duplicate handlers when running under reloaders
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_resolve_log_level(log_level))

    formatter = logging.Formatter(
        (
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"correlation_id": "%(correlation_id)s", "service": "'
            + service_name
            + '", '
            '"message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s"}'
        )
    )
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationIdFilter())

    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Return a logger instance properly named for the module."""
    return logger.getChild(name)


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set correlation ID for current context."""
    if correlation_id is None:
        clear_correlation_id()
        return
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from current context."""
    return correlation_id_context.get()


def clear_correlation_id() -> None:
    """Reset the correlation ID for the current context."""
    correlation_id_context.set(None)


__all__ = [
    "logger",
    "set_correlation_id",
    "get_correlation_id",
    "clear_correlation_id",
]