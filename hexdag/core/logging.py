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

>>> from hexdag.core.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Pipeline started", pipeline_id="123")

Configure logging globally::

    from hexdag.core.logging import configure_logging
    configure_logging(level="DEBUG", format="json")

Custom configuration:

>>> from hexdag.core.logging import configure_logging
>>> configure_logging(
...     level="INFO",
...     format="structured",
...     output_file="hexdag.log",
... )
"""

import contextvars
import logging
import os
import sys
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import types

    from hexdag.core.types import Logger

from loguru import logger
from rich.logging import RichHandler

LogLevel = Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["console", "json", "structured", "rich", "dual"]

_CURRENT_CONFIG: dict | None = None
_HANDLER_IDS: list[int] = []

# Correlation ID context variable for request tracing
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")


def configure_logging(
    level: LogLevel = "INFO",
    format: LogFormat = "structured",
    output_file: str | Path | None = None,
    use_color: bool = True,
    include_timestamp: bool = True,
    force_reconfigure: bool = False,
    use_rich: bool = False,
    dual_sink: bool = False,
    enable_stdlib_bridge: bool = False,
    backtrace: bool = True,
    diagnose: bool = True,
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
        - "console": Simple console output (no colors, basic format)
        - "json": JSON format for log aggregation (uses orjson for performance)
        - "structured": Enhanced structured format with colors (Loguru native)
        - "rich": Rich console handler with beautiful formatting
        - "dual": Dual-sink mode (Rich to stderr + JSON to stdout)
    output_file : str | Path | None, default=None
        Optional file path to write logs to (in addition to console)
    use_color : bool, default=True
        Use ANSI color codes in structured format (auto-disabled for non-TTY)
    include_timestamp : bool, default=True
        Include timestamp in log output
    force_reconfigure : bool, default=False
        Force reconfiguration even if already configured with same settings
    use_rich : bool, default=False
        Use Rich library for enhanced console output (overrides format if True)
    dual_sink : bool, default=False
        Enable dual-sink: Rich console (stderr) + JSON (stdout) simultaneously
    enable_stdlib_bridge : bool, default=False
        Enable interception of stdlib logging for third-party libraries
    backtrace : bool, default=True
        Enable backtrace for debugging (disable in production for security)
    diagnose : bool, default=True
        Enable diagnose mode with variable values (disable in production for security)

    Examples
    --------
    Development setup with Rich::

        configure_logging(level="DEBUG", format="rich", use_rich=True)

    Dual-sink setup (Rich console + JSON for aggregation)::

        configure_logging(level="INFO", dual_sink=True, use_rich=True)

    Production setup::

        configure_logging(
            level="INFO",
            format="json",
            output_file="/var/log/hexdag/app.log",
            backtrace=True,
            diagnose=False,  # Disable for security
        )

    Testing setup::

        configure_logging(level="WARNING", format="console")
    """
    global _CURRENT_CONFIG, _HANDLER_IDS

    current_config = {
        "level": level,
        "format": format,
        "output_file": str(output_file) if output_file else None,
        "use_color": use_color,
        "include_timestamp": include_timestamp,
        "use_rich": use_rich,
        "dual_sink": dual_sink,
        "enable_stdlib_bridge": enable_stdlib_bridge,
        "backtrace": backtrace,
        "diagnose": diagnose,
    }

    if not force_reconfigure and current_config == _CURRENT_CONFIG:
        return

    # Remove only our previously added handlers (not external ones)
    # This ensures we don't interfere with pytest or other framework handlers
    for handler_id in _HANDLER_IDS:
        with suppress(ValueError):
            logger.remove(handler_id)
    _HANDLER_IDS.clear()

    # Prepare format strings and track handler IDs
    if dual_sink or format == "dual":
        # Dual-sink mode: Rich console (stderr) + JSON (stdout)
        # 1. Rich handler for human-readable output to stderr
        rich_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=include_timestamp,
            show_level=True,
            show_path=True,
        )
        handler_id = logger.add(
            sink=rich_handler,
            level=level,
            format="{message}",
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

        # 2. JSON handler for machine-readable output to stdout
        handler_id = logger.add(
            sink=sys.stdout,
            level=level,
            serialize=True,  # JSON output
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    elif use_rich or format == "rich":
        # Rich-only mode
        rich_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=include_timestamp,
            show_level=True,
            show_path=True,
        )
        handler_id = logger.add(
            sink=rich_handler,
            level=level,
            format="{message}",
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    elif format == "json":
        # JSON format with orjson serialization
        handler_id = logger.add(
            sink=sys.stderr,
            level=level,
            serialize=True,  # JSON output
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    elif format == "structured":
        # Structured format with optional colors
        timestamp_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> " if include_timestamp else ""
        color_level = (
            "<level>{level: <8}</level>" if use_color and sys.stderr.isatty() else "{level: <8}"
        )
        # Note: cid (correlation ID) is injected via .bind() in get_logger()
        structured_format = (
            f"{timestamp_fmt}[{color_level}]"
            "<cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>"
        )

        handler_id = logger.add(
            sink=sys.stderr,
            level=level,
            format=structured_format,
            colorize=use_color and sys.stderr.isatty(),
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    else:  # console
        # Simple console format
        timestamp_fmt = "{time:YYYY-MM-DD HH:mm:ss} " if include_timestamp else ""
        console_format = f"{timestamp_fmt}{{level: <8}} | {{name}} | {{message}}"

        handler_id = logger.add(
            sink=sys.stderr,
            level=level,
            format=console_format,
            colorize=False,
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # File output always uses JSON for easier parsing
        handler_id = logger.add(
            sink=output_path,
            level=level,
            serialize=True,  # JSON format for files
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip",  # Compress rotated logs
            backtrace=backtrace,
            diagnose=diagnose,
        )
        _HANDLER_IDS.append(handler_id)

    # Enable stdlib logging bridge if requested
    if enable_stdlib_bridge:
        enable_stdlib_logging_bridge()

    _CURRENT_CONFIG = current_config


@lru_cache(maxsize=256)
def get_logger(name: str) -> "Logger":
    """Get a logger instance with the given name (cached for performance).

    This is the recommended way to get loggers in hexDAG. Logger instances
    are bound with the module name for better tracking.

    Correlation IDs are dynamically included via ContextVar when logging.

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
    >>> from hexdag.core.logging import get_logger, set_correlation_id
    >>> set_correlation_id("req-123")
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting orchestrator")  # Includes cid=req-123 in context
    >>> logger.debug("Processing node", node_id="abc123")

    Notes
    -----
    - If configure_logging() hasn't been called, initializes with sensible defaults
    - Logger names follow the module hierarchy (e.g., "hexdag.core.orchestrator")
    - Correlation ID is automatically included from ContextVar when set
    - Extra context can be passed as keyword arguments
    - Logger instances are cached for performance
    """
    # Lazy initialization - only configure if not already done
    _ensure_configured()
    return logger.bind(module=name)


@lru_cache(maxsize=128)
def get_logger_for_component(component_type: str, component_name: str) -> "Logger":
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
        Logger bound with hierarchical name like "hexdag.adapter.openai_llm"

    Examples
    --------
    >>> logger = get_logger_for_component("adapter", "openai_llm")
    >>> logger.info("LLM adapter initialized")
    """
    # Lazy initialization - only configure if not already done
    _ensure_configured()
    logger_name = f"hexdag.{component_type}.{component_name}"
    return logger.bind(
        module=logger_name, component_type=component_type, component_name=component_name
    )


def enable_stdlib_logging_bridge() -> None:
    """Enable interception of stdlib logging for third-party libraries.

    This redirects all stdlib logging.Logger calls to Loguru, ensuring
    consistent formatting across all logs including from dependencies.

    Examples
    --------
    >>> from hexdag.core.logging import configure_logging, enable_stdlib_logging_bridge
    >>> configure_logging(level="INFO", format="structured")
    >>> enable_stdlib_logging_bridge()  # Now all stdlib logging goes through loguru
    """

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
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


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for the current context.

    This ID will be automatically included in all log records emitted
    within the current async context or thread.

    Parameters
    ----------
    cid : str
        Correlation ID (e.g., request ID, trace ID, session ID)

    Examples
    --------
    >>> from hexdag.core.logging import set_correlation_id, get_logger
    >>> set_correlation_id("req-abc-123")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing request")  # Logs will include cid=req-abc-123
    """
    correlation_id.set(cid)


def get_correlation_id() -> str:
    """Get the current correlation ID.

    Returns
    -------
    str
        Current correlation ID, or "-" if not set

    Examples
    --------
    >>> from hexdag.core.logging import get_correlation_id, set_correlation_id
    >>> get_correlation_id()
    '-'
    >>> set_correlation_id('req-abc-123')
    >>> get_correlation_id()
    'req-abc-123'
    """
    return correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context.

    Examples
    --------
    >>> from hexdag.core.logging import clear_correlation_id
    >>> clear_correlation_id()
    """
    correlation_id.set("-")


def _ensure_configured() -> None:
    """Ensure logging has at least basic configuration (lazy initialization).

    This is called automatically by get_logger() if no configuration exists.
    Users can call configure_logging() explicitly for custom settings.
    """

    global _CURRENT_CONFIG

    if _CURRENT_CONFIG is None:
        # Default configuration - check environment variables first
        level = os.getenv("HEXDAG_LOG_LEVEL", "INFO").upper()
        format_type = os.getenv("HEXDAG_LOG_FORMAT", "structured").lower()
        configure_logging(level=level, format=format_type)  # type: ignore
