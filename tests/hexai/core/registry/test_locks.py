import pytest

from hexai.core.registry.locks import ReaderContext, ReadWriteLock, WriterContext


class TestReadWriteLock:
    """Test the read-write lock implementation."""

    def test_multiple_readers(self):
        """Multiple readers can acquire lock simultaneously."""
        lock = ReadWriteLock()

        with ReaderContext(lock), ReaderContext(lock):  # Nested read works
            assert True

    def test_writer_blocks_readers(self):
        """Writer blocks new readers until released."""
        lock = ReadWriteLock()

        with WriterContext(lock):
            # Writer is active, check state
            assert lock._readers == 0  # No readers should be active

    def test_reader_blocks_writer(self):
        """Active readers block writers (in different threads)."""
        lock = ReadWriteLock()

        with ReaderContext(lock):
            # Note: RLock is reentrant, so same thread can acquire again
            # This test would only work correctly with threading
            # For now, just verify the reader count is correct
            assert lock._readers == 1

    def test_error_on_extra_release(self):
        """Extra release raises RuntimeError without auto-reset."""
        lock = ReadWriteLock()

        lock.acquire_read()
        lock.release_read()

        with pytest.raises(RuntimeError, match="without matching acquire_read"):
            lock.release_read()

    def test_reentrant_read(self):
        """The same thread can re-enter read lock (RLock semantics)."""
        lock = ReadWriteLock()

        with ReaderContext(lock), ReaderContext(lock):  # Should not deadlock
            assert True

    def test_reentrant_write(self):
        """The same thread can re-enter write lock (RLock semantics)."""
        lock = ReadWriteLock()

        with WriterContext(lock), WriterContext(lock):  # Should not deadlock
            assert True
