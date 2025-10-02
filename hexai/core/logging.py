"""Centralized logging configuration for hexDAG.

Provides consistent logging across the framework with support for:
- Multiple output formats (console, JSON, structured)
- Environment-based configuration (dev/prod)
- Performance monitoring
- Structured context logging
- Integration with observability systems

Examples
--------
Basic usage:

>>> from hexai.core.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Pipeline started", extra={"pipeline_id": "123"})

Configure logging globally:

>>> from hexai.core.logging import configure_logging
>>> configure_logging(level="DEBUG", format="json")

Custom configuration:

>>> from hexai.core.logging import configure_logging
>>> configure_logging(
...     level="INFO",
...     format="structured",
...     output_file="hexdag.log",
...     include_context=True
... )
"""

import logging
import sys
from pathlib import Path
from typing import Literal

# Global flag to track if logging has been configured
_LOGGING_CONFIGURED = False

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json", "structured"]


class StructuredFormatter(logging.Formatter):
    """Enhanced formatter with structured output and color support.

    Provides clean, readable logs with optional color coding for terminals.
    Includes timestamp, level, logger name, and message with context.
    """

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_color: bool = True, include_timestamp: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structure and optional color."""
        # Build timestamp
        timestamp = ""
        if self.include_timestamp:
            timestamp = f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "

        # Build level with optional color
        level = record.levelname
        if self.use_color:
            color = self.COLORS.get(level, self.COLORS["RESET"])
            level_colored = f"{color}{level:8s}{self.COLORS['RESET']}"
        else:
            level_colored = f"{level:8s}"

        # Build logger name (shorten if too long)
        logger_name = record.name
        if len(logger_name) > 30:
            parts = logger_name.split(".")
            if len(parts) > 2:
                logger_name = f"{parts[0]}...{parts[-1]}"

        # Format message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        # Build final output
        return f"{timestamp}[{level_colored}] {logger_name:30s} | {message}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging and log aggregation systems.

    Outputs logs as JSON for easy parsing by log aggregation tools
    (e.g., ELK stack, Datadog, CloudWatch).
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        import json

        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra context from record
        log_data.update({
            key: value
            for key, value in record.__dict__.items()
            if key
            not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]
        })

        return json.dumps(log_data)


def configure_logging(
    level: LogLevel = "INFO",
    format: LogFormat = "structured",
    output_file: str | Path | None = None,
    use_color: bool = True,
    include_timestamp: bool = True,
    force_reconfigure: bool = False,
) -> None:
    """Configure global logging for hexDAG framework.

    This should be called once at application startup to set up consistent
    logging across the entire framework.

    Parameters
    ----------
    level : LogLevel, default="INFO"
        Minimum log level to output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format : LogFormat, default="structured"
        Output format:
        - "console": Simple console output
        - "json": JSON format for log aggregation
        - "structured": Enhanced structured format with colors
    output_file : str | Path | None, default=None
        Optional file path to write logs to (in addition to console)
    use_color : bool, default=True
        Use ANSI color codes in structured format (auto-disabled for non-TTY)
    include_timestamp : bool, default=True
        Include timestamp in log output
    force_reconfigure : bool, default=False
        Force reconfiguration even if already configured

    Examples
    --------
    Development setup:

    >>> configure_logging(level="DEBUG", format="structured", use_color=True)

    Production setup:

    >>> configure_logging(
    ...     level="INFO",
    ...     format="json",
    ...     output_file="/var/log/hexdag/app.log"
    ... )

    Testing setup:

    >>> configure_logging(level="WARNING", format="console")
    """
    global _LOGGING_CONFIGURED

    # Check if already configured
    if _LOGGING_CONFIGURED and not force_reconfigure:
        return

    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, level)
    root_logger.setLevel(log_level)

    # Create formatter based on format type
    formatter: logging.Formatter
    if format == "json":
        formatter = JSONFormatter()
    elif format == "structured":
        formatter = StructuredFormatter(use_color=use_color, include_timestamp=include_timestamp)
    else:  # console
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler (stderr for better separation from stdout)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(output_path)
        # Always use JSON for file output for easier parsing
        file_handler.setFormatter(JSONFormatter() if format == "json" else formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Set library loggers to WARNING by default to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("concurrent").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    This is the recommended way to get loggers in hexDAG. It ensures
    consistent configuration and lazy initialization.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ from the calling module

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting orchestrator")
    >>> logger.debug("Processing node", extra={"node_id": "abc123"})

    Notes
    -----
    - If configure_logging() hasn't been called, uses Python's default config
    - Logger names follow the module hierarchy (e.g., "hexai.core.orchestrator")
    - Extra context can be passed via the 'extra' parameter
    """
    return logging.getLogger(name)


def get_logger_for_component(component_type: str, component_name: str) -> logging.Logger:
    """Get a logger for a specific component instance.

    Useful for adapters, nodes, and other components that need
    instance-specific logging.

    Parameters
    ----------
    component_type : str
        Type of component (e.g., "adapter", "node", "observer")
    component_name : str
        Specific component name (e.g., "openai_llm", "postgres_db")

    Returns
    -------
    logging.Logger
        Logger with hierarchical name like "hexai.adapter.openai_llm"

    Examples
    --------
    >>> logger = get_logger_for_component("adapter", "openai_llm")
    >>> logger.info("LLM adapter initialized")
    """
    logger_name = f"hexai.{component_type}.{component_name}"
    return logging.getLogger(logger_name)


# Auto-configure with sensible defaults if not explicitly configured
# This ensures logging works even if configure_logging() is never called
def _ensure_default_config() -> None:
    """Ensure logging has at least basic configuration."""
    global _LOGGING_CONFIGURED

    if not _LOGGING_CONFIGURED:
        # Check if root logger has handlers
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # No handlers - configure with defaults
            configure_logging(level="INFO", format="structured")


# Auto-configure on module import
_ensure_default_config()
