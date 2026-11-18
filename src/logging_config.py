"""
Structured JSON logging configuration for MP190 prediction service.

Ensures log records are formatted as single-line JSON objects
emitted strictly to stdout without writing sensitive data to files.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter outputting logs as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            # Exclude personal/sensitive keys if any appear
            sanitized = {
                k: v for k, v in record.extra_fields.items()
                if not any(sub in k.lower() for sub in ["password", "secret", "token", "personal", "name", "email"])
            }
            log_entry.update(sanitized)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logger(name: str = "mp190", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with JSON formatting writing to stdout.

    Args:
        name: Logger identifier.
        level: Minimum logging level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
