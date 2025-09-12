"""Thread-safe locking mechanisms for the registry.

Note: After bootstrap, the registry is effectively read-only,
so locking becomes less critical. We keep it for safety during
the bootstrap phase and for development mode.
"""

from __future__ import annotations

from threading import RLock
from types import TracebackType
from typing import Literal


class ReaderContext:
    """Context manager for read locks."""

    def __init__(self, lock: ReadWriteLock):
        self._lock = lock

    def __enter__(self) -> ReaderContext:
        """Enter the read lock context."""
        self._lock.acquire_read()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the read lock context."""
        self._lock.release_read()
        return False


class WriterContext:
    """Context manager for write locks."""

    def __init__(self, lock: ReadWriteLock):
        self._lock = lock

    def __enter__(self) -> WriterContext:
        """Enter the write lock context."""
        self._lock.acquire_write()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit the write lock context."""
        self._lock.release_write()
        return False


class ReadWriteLock:
    """Thread-safe read/write lock (sync only).

    - Multiple readers OR one writer.
    - Not async-aware.
    - Writers may starve if readers dominate.

    Post-bootstrap note: Since the registry becomes read-only after
    bootstrap, this lock is primarily used during the bootstrap phase.
    After that, reads could theoretically be lock-free.
    """

    def __init__(self) -> None:
        self._readers = 0
        self._read_ready = RLock()
        self._write_ready = RLock()

    def acquire_read(self) -> None:
        """Acquire a read lock."""
        with self._read_ready:
            self._readers += 1
            if self._readers == 1:
                self._write_ready.acquire()

    def release_read(self) -> None:
        """Release a read lock."""
        with self._read_ready:
            if self._readers == 0:
                raise RuntimeError("release_read without matching acquire_read")
            self._readers -= 1
            if self._readers == 0:
                self._write_ready.release()

    def acquire_write(self) -> None:
        """Acquire a write lock."""
        self._write_ready.acquire()

    def release_write(self) -> None:
        """Release a write lock."""
        self._write_ready.release()

    def read(self) -> ReaderContext:
        """Get a reader context manager."""
        return ReaderContext(self)

    def write(self) -> WriterContext:
        """Get a writer context manager."""
        return WriterContext(self)
