"""Tests for async_warnings module."""

from __future__ import annotations

import warnings

import pytest

from hexdag.core.utils.async_warnings import (
    AsyncIOWarning,
    SyncIOMonitor,
    _is_in_async_context,
    warn_file_open,
    warn_if_async,
    warn_requests_call,
    warn_sqlite_connect,
    warn_sync_io,
    warn_time_sleep,
)


class TestAsyncContextDetection:
    """Test async context detection."""

    def test_is_in_async_context_sync(self) -> None:
        """Test that sync context is detected correctly."""
        assert not _is_in_async_context()

    async def test_is_in_async_context_async(self) -> None:
        """Test that async context is detected correctly."""
        assert _is_in_async_context()


class TestWarnSyncIO:
    """Test warn_sync_io function."""

    async def test_warn_sync_io_in_async_context(self) -> None:
        """Test that warning is emitted in async context."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_sync_io("test_operation", "Use async alternative")

            assert len(w) == 1
            assert issubclass(w[0].category, AsyncIOWarning)
            assert "test_operation" in str(w[0].message)
            assert "Use async alternative" in str(w[0].message)

    def test_warn_sync_io_in_sync_context(self) -> None:
        """Test that no warning is emitted in sync context."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_sync_io("test_operation", "Use async alternative")

            assert len(w) == 0

    async def test_warn_sync_io_without_suggestion(self) -> None:
        """Test warning without suggestion."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_sync_io("test_operation")

            assert len(w) == 1
            assert "test_operation" in str(w[0].message)


class TestWarnIfAsync:
    """Test warn_if_async decorator."""

    async def test_warn_if_async_in_async_context(self) -> None:
        """Test that decorator warns when called in async context."""

        @warn_if_async
        def sync_function() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = sync_function()

            assert result == "result"
            assert len(w) == 1
            assert issubclass(w[0].category, AsyncIOWarning)
            assert "sync_function" in str(w[0].message)

    def test_warn_if_async_in_sync_context(self) -> None:
        """Test that decorator doesn't warn in sync context."""

        @warn_if_async
        def sync_function() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = sync_function()

            assert result == "result"
            assert len(w) == 0

    async def test_warn_if_async_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @warn_if_async
        def my_function() -> str:
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestSyncIOMonitor:
    """Test SyncIOMonitor context manager."""

    async def test_sync_io_monitor_in_async_context(self) -> None:
        """Test that monitor detects async context."""
        with SyncIOMonitor("test context") as monitor:
            assert monitor.is_async

    def test_sync_io_monitor_in_sync_context(self) -> None:
        """Test that monitor detects sync context."""
        with SyncIOMonitor("test context") as monitor:
            assert not monitor.is_async

    async def test_sync_io_monitor_check_operation(self) -> None:
        """Test check_operation method."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with SyncIOMonitor("test context") as monitor:
                monitor.check_operation("file_read", "Use aiofiles")

            assert len(w) == 1
            assert "file_read" in str(w[0].message)
            assert "Use aiofiles" in str(w[0].message)

    async def test_sync_io_monitor_exception_handling(self) -> None:
        """Test that monitor handles exceptions properly."""
        with pytest.raises(ValueError):
            with SyncIOMonitor("test context"):
                raise ValueError("Test error")


class TestConvenienceFunctions:
    """Test convenience warning functions."""

    async def test_warn_file_open(self) -> None:
        """Test warn_file_open function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_file_open("/path/to/file")

            assert len(w) == 1
            assert "open" in str(w[0].message)
            assert "/path/to/file" in str(w[0].message)
            assert "aiofiles" in str(w[0].message)

    async def test_warn_sqlite_connect(self) -> None:
        """Test warn_sqlite_connect function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_sqlite_connect("db.sqlite")

            assert len(w) == 1
            assert "sqlite3.connect" in str(w[0].message)
            assert "db.sqlite" in str(w[0].message)
            assert "aiosqlite" in str(w[0].message)

    async def test_warn_requests_call(self) -> None:
        """Test warn_requests_call function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_requests_call("GET", "http://example.com")

            assert len(w) == 1
            assert "requests.get" in str(w[0].message)
            assert "http://example.com" in str(w[0].message)
            assert "aiohttp" in str(w[0].message)

    async def test_warn_time_sleep(self) -> None:
        """Test warn_time_sleep function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            warn_time_sleep(1.5)

            assert len(w) == 1
            assert "time.sleep" in str(w[0].message)
            assert "1.5" in str(w[0].message)
            assert "asyncio.sleep" in str(w[0].message)


class TestWarningConfiguration:
    """Test warning configuration options."""

    async def test_warning_as_error(self) -> None:
        """Test converting warnings to errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", AsyncIOWarning)

            with pytest.raises(AsyncIOWarning):
                warn_sync_io("test_operation")

    async def test_warning_ignore(self) -> None:
        """Test ignoring warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", AsyncIOWarning)

            warn_sync_io("test_operation")

            assert len(w) == 0


class TestIntegration:
    """Integration tests for async warnings."""

    async def test_real_world_scenario(self) -> None:
        """Test a real-world scenario with multiple warnings."""

        @warn_if_async
        def sync_helper() -> str:
            return "data"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Multiple operations that should warn
            result = sync_helper()
            warn_file_open("config.yaml")
            warn_sqlite_connect("app.db")

            assert result == "data"
            assert len(w) == 3

            # Check all warnings are AsyncIOWarning
            assert all(issubclass(warning.category, AsyncIOWarning) for warning in w)

    async def test_nested_async_calls(self) -> None:
        """Test warnings in nested async functions."""

        async def outer() -> str:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                warn_sync_io("outer_operation")

                async def inner() -> str:
                    warn_sync_io("inner_operation")
                    return "result"

                result = await inner()

                assert len(w) == 2
                return result

        result = await outer()
        assert result == "result"
