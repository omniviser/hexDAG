"""Runtime warnings for synchronous I/O operations in async contexts.

This module provides utilities to detect and warn about blocking I/O operations
at runtime when executing within async functions or coroutines.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

from hexdag.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

T = TypeVar("T")


class AsyncIOWarning(UserWarning):
    """Warning emitted when sync I/O is detected in async context."""

    pass


def _is_in_async_context() -> bool:
    """Check if code is running in an async context.

    Returns
    -------
        True if running in async context (event loop is running)
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def warn_sync_io(operation: str, suggestion: str | None = None) -> None:
    """Emit a warning about synchronous I/O in async context.

    Args
    ----
        operation: Description of the blocking operation
        suggestion: Optional suggestion for async alternative
    """
    if not _is_in_async_context():
        return

    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        filename = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        function_name = caller_frame.f_code.co_name

        message = (
            f"Blocking I/O operation '{operation}' detected in async context "
            f"(function: {function_name}, {filename}:{line_number})"
        )

        if suggestion:
            message += f". {suggestion}"

        warnings.warn(message, AsyncIOWarning, stacklevel=3)
        logger.warning(message)


def warn_if_async[T](func: Callable[..., T]) -> Callable[..., T]:
    """Warn if a synchronous function is called in async context.

    Args
    ----
        func: Function to wrap

    Returns
    -------
        Wrapped function that warns if called in async context

    Example
    -------
        >>> @warn_if_async
        ... def sync_database_query(sql: str) -> list:
        ...     return connection.execute(sql).fetchall()
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if _is_in_async_context():
            warn_sync_io(
                f"{func.__name__}()",
                f"Consider using an async version of {func.__name__}",
            )
        return func(*args, **kwargs)

    return wrapper


class SyncIOMonitor:
    """Context manager to monitor and warn about sync I/O operations.

    This can be used to wrap code sections that should use async I/O but might
    accidentally use blocking operations.

    Example
    -------
        >>> async def process_data():
        ...     with SyncIOMonitor("data processing"):
        ...         # Any blocking I/O here will trigger warnings
        ...         data = process_file()  # Would warn if this blocks
    """

    def __init__(self, context_name: str = "code block") -> None:
        """Initialize the monitor.

        Args
        ----
            context_name: Name of the context being monitored
        """
        self.context_name = context_name
        self.is_async = False

    def __enter__(self) -> SyncIOMonitor:
        """Enter the monitoring context."""
        self.is_async = _is_in_async_context()
        return self

    def __exit__(
        self,
        _exc_type: Any,  # noqa: ARG002
        _exc_val: Any,  # noqa: ARG002
        _exc_tb: Any,  # noqa: ARG002
    ) -> None:
        """Exit the monitoring context."""
        pass

    def check_operation(self, operation: str, suggestion: str | None = None) -> None:
        """Check and warn about an operation if in async context.

        Args
        ----
            operation: Description of the operation
            suggestion: Optional suggestion for async alternative
        """
        if self.is_async:
            warn_sync_io(operation, suggestion)


# Commonly monitored operations
def warn_file_open(path: str) -> None:
    """Warn about sync file open in async context.

    Args
    ----
        path: File path being opened
    """
    warn_sync_io(
        f"open('{path}')",
        "Use aiofiles.open() for async file I/O",
    )


def warn_sqlite_connect(db_path: str) -> None:
    """Warn about sync SQLite connection in async context.

    Args
    ----
        db_path: Database path
    """
    warn_sync_io(
        f"sqlite3.connect('{db_path}')",
        "Use aiosqlite.connect() for async database operations",
    )


def warn_requests_call(method: str, url: str) -> None:
    """Warn about sync HTTP request in async context.

    Args
    ----
        method: HTTP method (GET, POST, etc.)
        url: Request URL
    """
    warn_sync_io(
        f"requests.{method.lower()}('{url}')",
        "Use aiohttp.ClientSession for async HTTP requests",
    )


def warn_time_sleep(seconds: float) -> None:
    """Warn about sync sleep in async context.

    Args
    ----
        seconds: Sleep duration
    """
    warn_sync_io(
        f"time.sleep({seconds})",
        "Use await asyncio.sleep() in async functions",
    )
