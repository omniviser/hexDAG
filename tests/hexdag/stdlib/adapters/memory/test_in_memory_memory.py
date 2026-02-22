"""Tests for InMemoryMemory implementation."""

import asyncio

import pytest

from hexdag.stdlib.adapters.memory import InMemoryMemory


class TestInMemoryMemory:
    """Test cases for InMemoryMemory."""

    @pytest.fixture
    def memory(self):
        """Create an InMemoryMemory instance for testing."""
        return InMemoryMemory()

    @pytest.mark.asyncio
    async def test_basic_storage_and_retrieval(self, memory):
        """Test basic set and get operations."""
        await memory.aset("key1", "value1")
        await memory.aset("key2", {"nested": "data"})

        result1 = await memory.aget("key1")
        result2 = await memory.aget("key2")

        assert result1 == "value1"
        assert result2 == {"nested": "data"}

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory):
        """Test getting a key that doesn't exist returns None."""
        result = await memory.aget("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_value(self, memory):
        """Test overwriting existing values."""
        await memory.aset("key", "original")
        await memory.aset("key", "updated")

        result = await memory.aget("key")
        assert result == "updated"

    @pytest.mark.asyncio
    async def test_access_history_tracking(self, memory):
        """Test that access history is properly tracked."""
        await memory.aset("key1", "value1")
        await memory.aget("key1")
        await memory.aget("nonexistent")

        history = memory.get_access_history()
        assert len(history) == 3

        # Check set operation
        set_op = history[0]
        assert set_op["operation"] == "set"
        assert set_op["key"] == "key1"
        assert set_op["value_type"] == "str"

        # Check successful get operation
        get_op = history[1]
        assert get_op["operation"] == "get"
        assert get_op["key"] == "key1"
        assert get_op["found"] is True

        # Check failed get operation
        failed_get = history[2]
        assert failed_get["operation"] == "get"
        assert failed_get["key"] == "nonexistent"
        assert failed_get["found"] is False

    def test_clear_functionality(self, memory):
        """Test clearing stored data."""
        # Add some data
        asyncio.run(memory.aset("key1", "value1"))
        asyncio.run(memory.aset("key2", "value2"))

        assert memory.size() == 2
        assert "key1" in memory.get_stored_keys()

        # Clear and verify
        memory.clear()
        assert memory.size() == 0
        assert len(memory.get_stored_keys()) == 0

        # History should still be there
        assert len(memory.get_access_history()) > 0

    def test_reset_functionality(self, memory):
        """Test resetting memory completely."""
        # Add some data and history
        asyncio.run(memory.aset("key1", "value1"))
        asyncio.run(memory.aget("key1"))

        assert memory.size() == 1
        assert len(memory.get_access_history()) == 2

        # Reset and verify everything is cleared
        memory.reset()
        assert memory.size() == 0
        assert len(memory.get_access_history()) == 0
        assert len(memory.get_stored_keys()) == 0

    def test_utility_methods(self, memory):
        """Test utility methods."""
        # Initially empty
        assert memory.size() == 0
        assert memory.get_stored_keys() == []
        assert not memory.has_key("any_key")

        # Add some data
        asyncio.run(memory.aset("key1", "value1"))
        asyncio.run(memory.aset("key2", {"complex": "object"}))

        # Test utility methods
        assert memory.size() == 2
        assert set(memory.get_stored_keys()) == {"key1", "key2"}
        assert memory.has_key("key1")
        assert memory.has_key("key2")
        assert not memory.has_key("key3")

    @pytest.mark.asyncio
    async def test_history_immutable(self, memory):
        """Test that access history returns a copy and is not mutable."""
        await memory.aset("test", "value")

        history1 = memory.get_access_history()
        history2 = memory.get_access_history()

        # Should be separate instances
        assert history1 is not history2
        assert history1 == history2

        # Modifying returned history shouldn't affect internal state
        history1.append({"fake": "entry"})
        internal_history = memory.get_access_history()
        assert len(internal_history) == 1
        assert "fake" not in str(internal_history)

    @pytest.mark.asyncio
    async def test_various_data_types(self, memory):
        """Test storing various data types."""
        test_data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": {"deeply": "nested"}},
            "none": None,
            "boolean": True,
        }

        for key, value in test_data.items():
            await memory.aset(key, value)

        for key, expected_value in test_data.items():
            result = await memory.aget(key)
            assert result == expected_value
