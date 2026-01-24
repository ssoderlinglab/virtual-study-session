"""Logging configuration for Virtual Lab.

This module provides structured logging with configurable output and levels.
It supports both console and file output, with optional structured metadata.

Usage:
    from virtual_lab.logging_config import get_logger, setup_logging

    # Setup logging once at application start
    setup_logging(level="INFO", log_file="virtual_lab.log")

    # Get a logger for your module
    logger = get_logger(__name__)

    # Log messages with optional structured data
    logger.info("Starting meeting", extra={"phase": "team_selection"})
    logger.error("API call failed", extra={"status_code": 500})

Environment Variables:
    VIRTUAL_LAB_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    VIRTUAL_LAB_LOG_FILE: Set log file path (optional, defaults to stdout only)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Default format strings
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

# Module-level logger instance cache
_loggers: dict[str, logging.Logger] = {}
_initialized = False


class ContextFilter(logging.Filter):
    """Filter that adds context information to log records.

    This filter adds a 'context' field to log records that combines any
    extra fields passed to the logging call into a formatted string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context field to log record.

        Args:
            record: The log record to filter.

        Returns:
            True (always allows the record through).
        """
        # Collect extra fields that aren't standard LogRecord attributes
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'taskName', 'message', 'asctime'
        }

        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith('_')
        }

        if extra_fields:
            record.context = " | " + " | ".join(f"{k}={v}" for k, v in extra_fields.items())
        else:
            record.context = ""

        return True


def setup_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
    format_string: str | None = None,
    detailed: bool = False,
) -> None:
    """Configure logging for the Virtual Lab application.

    This function should be called once at application startup to configure
    logging behavior. Subsequent calls will update the configuration.

    Args:
        level: Logging level as string (DEBUG, INFO, etc.) or int constant.
            Defaults to VIRTUAL_LAB_LOG_LEVEL env var or INFO.
        log_file: Optional path to log file. If provided, logs will be written
            to both the file and stdout.
        format_string: Custom format string. If None, uses default format.
        detailed: If True, uses detailed format with file/line info.

    Example:
        setup_logging(level="DEBUG", log_file="app.log", detailed=True)
    """
    global _initialized

    # Determine log level
    if level is None:
        level = os.environ.get("VIRTUAL_LAB_LOG_LEVEL", "INFO")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Determine log file
    if log_file is None:
        log_file = os.environ.get("VIRTUAL_LAB_LOG_FILE")

    # Determine format
    if format_string is None:
        format_string = DETAILED_FORMAT if detailed else DEFAULT_FORMAT

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger for virtual_lab
    root_logger = logging.getLogger("virtual_lab")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add context filter
    context_filter = ContextFilter()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with the specified name.

    If logging has not been set up yet, initializes with default configuration.
    Logger instances are cached to avoid creating duplicates.

    Args:
        name: Logger name, typically __name__ of the calling module.
            If None, returns the root virtual_lab logger.

    Returns:
        Configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"items": 100})
    """
    global _initialized

    if not _initialized:
        setup_logging()

    if name is None:
        name = "virtual_lab"
    elif not name.startswith("virtual_lab"):
        # Ensure all loggers are under the virtual_lab namespace
        name = f"virtual_lab.{name}"

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log an exception with structured context.

    This is a convenience function for logging exceptions with consistent
    formatting and optional additional context.

    Args:
        logger: Logger instance to use.
        exc: The exception to log.
        message: Message prefix for the log entry.
        level: Logging level (defaults to ERROR).
        extra: Additional context to include in the log.

    Example:
        try:
            risky_operation()
        except Exception as e:
            log_exception(logger, e, "Operation failed", extra={"op": "risky"})
    """
    extra = extra or {}
    extra["exception_type"] = type(exc).__name__
    extra["exception_message"] = str(exc)

    logger.log(level, f"{message}: {exc}", extra=extra, exc_info=True)


class LogContext:
    """Context manager for adding consistent context to a block of logs.

    This context manager temporarily adds fields to all log messages within
    its scope.

    Example:
        with LogContext(logger, phase="team_selection", round=1):
            logger.info("Starting round")  # Will include phase and round
            do_work()
            logger.info("Round complete")  # Will include phase and round
    """

    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        """Initialize the log context.

        Args:
            logger: Logger instance to apply context to.
            **context: Key-value pairs to add to all log messages.
        """
        self.logger = logger
        self.context = context
        self._old_factory: Any = None

    def __enter__(self) -> "LogContext":
        """Enter the context, installing the custom record factory."""
        old_factory = logging.getLogRecordFactory()
        context = self.context

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        self._old_factory = old_factory
        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, restoring the original record factory."""
        if self._old_factory is not None:
            logging.setLogRecordFactory(self._old_factory)
