"""Tests for FileMemoryAdapter."""

import json
import tempfile
from pathlib import Path

import pytest

from hexdag.stdlib.adapters.memory import FileMemoryAdapter


class TestFileMemoryAdapter:
    """Tests for file-based Memory adapter."""

    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def memory_json(self, temp_dir):
        """Provide a JSON file memory adapter."""
        return FileMemoryAdapter(base_path=str(temp_dir / "json"), format="json")

    @pytest.fixture
    def memory_pickle(self, temp_dir):
        """Provide a pickle file memory adapter."""
        return FileMemoryAdapter(base_path=str(temp_dir / "pickle"), format="pickle")

    @pytest.mark.asyncio
    async def test_set_and_get_json(self, memory_json):
        """Test basic set and get with JSON format."""
        data = {"key": "value", "number": 42}

        await memory_json.aset("test", data)
        retrieved = await memory_json.aget("test")

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_set_and_get_pickle(self, memory_pickle):
        """Test basic set and get with pickle format."""
        # Use a standard Python object that can be pickled
        data = {"key": "value", "nested": {"list": [1, 2, 3]}}

        await memory_pickle.aset("obj", data)
        retrieved = await memory_pickle.aget("obj")

        assert retrieved == data

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory_json):
        """Test getting a key that doesn't exist."""
        value = await memory_json.aget("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_update_existing_key(self, memory_json):
        """Test updating an existing key."""
        await memory_json.aset("key", {"version": 1})
        await memory_json.aset("key", {"version": 2})

        value = await memory_json.aget("key")
        assert value == {"version": 2}

    @pytest.mark.asyncio
    async def test_delete_key(self, memory_json):
        """Test deleting a key."""
        await memory_json.aset("key", "value")

        deleted = await memory_json.adelete("key")
        assert deleted is True

        value = await memory_json.aget("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, memory_json):
        """Test deleting a key that doesn't exist."""
        deleted = await memory_json.adelete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_keys(self, memory_json):
        """Test listing all keys."""
        await memory_json.aset("key1", "value1")
        await memory_json.aset("key2", "value2")
        await memory_json.aset("key3", "value3")

        keys = await memory_json.alist_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    @pytest.mark.asyncio
    async def test_list_keys_with_prefix(self, memory_json):
        """Test listing keys with prefix filter."""
        await memory_json.aset("user_1", "Alice")
        await memory_json.aset("user_2", "Bob")
        await memory_json.aset("config_1", "Setting")

        user_keys = await memory_json.alist_keys(prefix="user_")
        assert len(user_keys) == 2
        assert "user_1" in user_keys
        assert "user_2" in user_keys
        assert "config_1" not in user_keys

    @pytest.mark.asyncio
    async def test_clear(self, memory_json):
        """Test clearing all keys."""
        await memory_json.aset("key1", "value1")
        await memory_json.aset("key2", "value2")

        await memory_json.aclear()

        keys = await memory_json.alist_keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_key_sanitization(self, memory_json):
        """Test that keys with special characters are sanitized."""
        # Keys with colons and slashes should work
        await memory_json.aset("checkpoint:run/123", "value")
        value = await memory_json.aget("checkpoint:run/123")

        assert value == "value"

    @pytest.mark.asyncio
    async def test_json_file_format(self, memory_json, temp_dir):
        """Test that JSON files are properly formatted."""
        data = {"key": "value"}
        await memory_json.aset("test", data)

        # Read the file directly and verify it's valid JSON
        file_path = memory_json._get_file_path("test")
        file_data = json.loads(file_path.read_text())

        assert file_data == data

    @pytest.mark.asyncio
    async def test_text_format(self, temp_dir):
        """Test text format storage."""
        memory = FileMemoryAdapter(base_path=str(temp_dir / "text"), format="text")

        await memory.aset("note", "This is a text note")
        value = await memory.aget("note")

        assert value == "This is a text note"

    @pytest.mark.asyncio
    async def test_custom_extension(self, temp_dir):
        """Test using a custom file extension."""
        memory = FileMemoryAdapter(
            base_path=str(temp_dir / "custom"), format="json", extension="config"
        )

        await memory.aset("app", {"setting": "value"})

        # Check file was created with custom extension
        file_path = temp_dir / "custom" / "app.config"
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_state_storage(self, memory_json):
        """Test storing checkpoint state as JSON string (real use case)."""
        # Simulate Pydantic model_dump_json()
        checkpoint_json = json.dumps({
            "run_id": "test-123",
            "dag_id": "workflow",
            "node_results": {"node1": "result1"},
            "completed_node_ids": ["node1"],
        })

        await memory_json.aset("checkpoint:test-123", checkpoint_json)
        retrieved = await memory_json.aget("checkpoint:test-123")

        # Should get back the same JSON string
        assert retrieved == checkpoint_json

        # Should be parseable
        parsed = json.loads(retrieved)
        assert parsed["run_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_create_dirs_disabled(self, temp_dir):
        """Test with directory creation disabled."""
        nonexistent_path = temp_dir / "does_not_exist"
        memory = FileMemoryAdapter(base_path=str(nonexistent_path), create_dirs=False)

        # Should fail because directory doesn't exist
        with pytest.raises(FileNotFoundError):
            await memory.aset("key", "value")

    @pytest.mark.asyncio
    async def test_empty_list_keys(self, memory_json):
        """Test listing keys when storage is empty."""
        keys = await memory_json.alist_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_pickle_preserves_types(self, memory_pickle):
        """Test that pickle preserves Python types correctly."""
        complex_data = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
        }

        await memory_pickle.aset("data", complex_data)
        retrieved = await memory_pickle.aget("data")

        assert retrieved == complex_data
        assert isinstance(retrieved["tuple"], tuple)
        assert isinstance(retrieved["set"], set)
