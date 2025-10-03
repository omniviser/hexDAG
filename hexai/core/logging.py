"""Centralized logging configuration for hexDAG using Loguru.

Provides consistent logging across the framework with support for:
- Multiple output formats (console, JSON, structured)
- Environment-based configuration (dev/prod)
- Performance monitoring
- Structured context logging
- Integration with observability systems
- Idempotent configuration

Examples
--------
Basic usage:

>>> from hexai.core.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Pipeline started", pipeline_id="123")

Configure logging globally:

>>> from hexai.core.logging import configure_logging
>>> configure_logging(level="DEBUG", format="json")

Custom configuration:

>>> from hexai.core.logging import configure_logging
>>> configure_logging(
...     level="INFO",
...     format="structured",
...     output_file="hexdag.log",
... )
"""

import logging
import sys
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import types

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json", "structured"]

# Store configured state to enable idempotent reconfiguration
_CURRENT_CONFIG: dict | None = None
# Track handler IDs added by configure_logging for safe cleanup
_HANDLER_IDS: list[int] = []


def configure_logging(
    level: LogLevel = "INFO",
    format: LogFormat = "structured",
    output_file: str | Path | None = None,
    use_color: bool = True,
    include_timestamp: bool = True,
    force_reconfigure: bool = False,
) -> None:
    """Configure global logging for hexDAG framework.

    This function is idempotent - calling it multiple times with the same
    configuration will not duplicate handlers or change settings.

    Parameters
    ----------
    level : LogLevel, default="INFO"
        Minimum log level to output (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format : LogFormat, default="structured"
        Output format:
        - "console": Simple console output
        - "json": JSON format for log aggregation (uses orjson for performance)
        - "structured": Enhanced structured format with colors
    output_file : str | Path | None, default=None
        Optional file path to write logs to (in addition to console)
    use_color : bool, default=True
        Use ANSI color codes in structured format (auto-disabled for non-TTY)
    include_timestamp : bool, default=True
        Include timestamp in log output
    force_reconfigure : bool, default=False
        Force reconfiguration even if already configured with same settings

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
    global _CURRENT_CONFIG, _HANDLER_IDS

    # Check if already configured with same settings (idempotent)
    current_config = {
        "level": level,
        "format": format,
        "output_file": str(output_file) if output_file else None,
        "use_color": use_color,
        "include_timestamp": include_timestamp,
    }

    if not force_reconfigure and current_config == _CURRENT_CONFIG:
        return

    # Remove only our handlers, not all handlers (safer for testing/integration)
    for handler_id in _HANDLER_IDS:
        with suppress(ValueError):
            # Handler may have been removed elsewhere, ignore ValueError
            logger.remove(handler_id)
    _HANDLER_IDS.clear()

    # Prepare format strings and track handler IDs
    if format == "json":
        # JSON format with orjson serialization
        handler_id = logger.add(
            sys.stderr,
            level=level,
            serialize=True,  # JSON output
            backtrace=True,
            diagnose=True,
        )
        _HANDLER_IDS.append(handler_id)
    elif format == "structured":
        # Structured format with optional colors
        timestamp_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> " if include_timestamp else ""
        color_level = (
            "<level>{level: <8}</level>" if use_color and sys.stderr.isatty() else "{level: <8}"
        )
        structured_format = f"{timestamp_fmt}[{color_level}]"
        structured_format += "<cyan>{{name}}:{{function}}:{{line}}</cyan> "
        structured_format += "| <level>{{message}}</level>"

        handler_id = logger.add(
            sys.stderr,
            level=level,
            format=structured_format,
            colorize=use_color and sys.stderr.isatty(),
            backtrace=True,
            diagnose=True,
        )
        _HANDLER_IDS.append(handler_id)
    else:  # console
        # Simple console format
        timestamp_fmt = "{time:YYYY-MM-DD HH:mm:ss} " if include_timestamp else ""
        console_format = f"{timestamp_fmt}{{level: <8}} | {{name}} | {{message}}"

        handler_id = logger.add(
            sys.stderr,
            level=level,
            format=console_format,
            colorize=False,
            backtrace=True,
            diagnose=False,
        )
        _HANDLER_IDS.append(handler_id)

    # Add file handler if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # File output always uses JSON for easier parsing
        handler_id = logger.add(
            output_path,
            level=level,
            serialize=True,  # JSON format for files
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip",  # Compress rotated logs
            backtrace=True,
            diagnose=True,
        )
        _HANDLER_IDS.append(handler_id)

    # Set library loggers to WARNING by default to reduce noise
    # (Loguru intercepts stdlib logging via logging bridge if needed)
    _CURRENT_CONFIG = current_config


@lru_cache(maxsize=256)
def get_logger(name: str) -> Any:  # Returns loguru.Logger
    """Get a logger instance with the given name (cached for performance).

    This is the recommended way to get loggers in hexDAG. Logger instances
    are bound with the module name for better tracking.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ from the calling module

    Returns
    -------
    loguru.Logger
        Configured logger instance bound with the module name

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting orchestrator")
    >>> logger.debug("Processing node", node_id="abc123")

    Notes
    -----
    - If configure_logging() hasn't been called, initializes with sensible defaults
    - Logger names follow the module hierarchy (e.g., "hexai.core.orchestrator")
    - Extra context can be passed as keyword arguments
    - Logger instances are cached for performance
    """
    # Lazy initialization - only configure if not already done
    _ensure_configured()
    return logger.bind(module=name)


@lru_cache(maxsize=128)
def get_logger_for_component(
    component_type: str, component_name: str
) -> Any:  # Returns loguru.Logger
    """Get a logger for a specific component instance (cached for performance).

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
    loguru.Logger
        Logger bound with hierarchical name like "hexai.adapter.openai_llm"

    Examples
    --------
    >>> logger = get_logger_for_component("adapter", "openai_llm")
    >>> logger.info("LLM adapter initialized")
    """
    # Lazy initialization - only configure if not already done
    _ensure_configured()
    logger_name = f"hexai.{component_type}.{component_name}"
    return logger.bind(
        module=logger_name, component_type=component_type, component_name=component_name
    )


def enable_stdlib_logging_bridge() -> None:
    """Enable interception of stdlib logging for third-party libraries.

    This redirects all stdlib logging.Logger calls to Loguru, ensuring
    consistent formatting across all logs including from dependencies.

    Examples
    --------
    >>> from hexai.core.logging import configure_logging, enable_stdlib_logging_bridge
    >>> configure_logging(level="INFO", format="structured")
    >>> enable_stdlib_logging_bridge()  # Now all stdlib logging goes through loguru
    """

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding Loguru level if it exists
            level: str | int
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame: types.FrameType | None = sys._getframe(6)
            depth = 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Intercept all stdlib logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def _ensure_configured() -> None:
    """Ensure logging has at least basic configuration (lazy initialization).

    This is called automatically by get_logger() if no configuration exists.
    Users can call configure_logging() explicitly for custom settings.
    """
    global _CURRENT_CONFIG

    if _CURRENT_CONFIG is None:
        # Default configuration - minimal, non-intrusive
        configure_logging(level="INFO", format="structured")
