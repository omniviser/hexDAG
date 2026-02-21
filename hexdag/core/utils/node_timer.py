"""Shared timing context manager for node execution.

Eliminates the repeated timing boilerplate across all node wrappers:
    start_time = time.perf_counter()
    ...
    duration_ms = (time.perf_counter() - start_time) * 1000
"""

import time
from collections.abc import Generator
from contextlib import contextmanager


class Timer:
    """Lightweight timer that tracks elapsed milliseconds.

    Examples
    --------
    >>> with node_timer() as t:
    ...     pass  # do work
    >>> assert t.duration_ms >= 0
    """

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        """Elapsed time in milliseconds since the timer started."""
        return (time.perf_counter() - self._start) * 1000

    @property
    def duration_str(self) -> str:
        """Elapsed time formatted as a string with 2 decimal places."""
        return f"{self.duration_ms:.2f}"


@contextmanager
def node_timer() -> Generator[Timer, None, None]:
    """Time an operation and provide elapsed milliseconds.

    Yields a ``Timer`` whose ``duration_ms`` property returns the
    elapsed time at any point during (or after) the block.

    Examples
    --------
    >>> with node_timer() as t:
    ...     pass  # do work
    >>> assert t.duration_ms >= 0
    """
    yield Timer()
