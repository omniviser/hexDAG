"""Tests for SQLiteMemoryAdapter."""

import pytest

from hexdag.stdlib.adapters.database import SQLiteAdapter
from hexdag.stdlib.adapters.memory import SQLiteMemoryAdapter


class TestSQLiteMemoryAdapter:
    """Tests for SQLite-backed Memory adapter."""

    @pytest.fixture
    async def database(self):
        """Provide an in-memory SQLite database."""
        db = SQLiteAdapter(db_path=":memory:")
        yield db
        await db.close()

    @pytest.fixture
    def memory(self, database):
        """Provide a SQLite memory adapter."""
        return SQLiteMemoryAdapter(database=database)

    @pytest.mark.asyncio
    async def test_set_and_get(self, memory):
        """Test basic set and get operations."""
        await memory.aset("test_key", "test_value")
        value = await memory.aget("test_key")

        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory):
        """Test getting a key that doesn't exist."""
        value = await memory.aget("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_update_existing_key(self, memory):
        """Test updating an existing key."""
        await memory.aset("key", "value1")
        await memory.aset("key", "value2")

        value = await memory.aget("key")
        assert value == "value2"

    @pytest.mark.asyncio
    async def test_delete_key(self, memory):
        """Test deleting a key."""
        await memory.aset("key", "value")

        deleted = await memory.adelete("key")
        assert deleted is True

        value = await memory.aget("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, memory):
        """Test deleting a key that doesn't exist."""
        deleted = await memory.adelete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_keys(self, memory):
        """Test listing all keys."""
        await memory.aset("key1", "value1")
        await memory.aset("key2", "value2")
        await memory.aset("key3", "value3")

        keys = await memory.alist_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    @pytest.mark.asyncio
    async def test_list_keys_with_prefix(self, memory):
        """Test listing keys with prefix filter."""
        await memory.aset("user:1", "Alice")
        await memory.aset("user:2", "Bob")
        await memory.aset("config:1", "Setting")

        user_keys = await memory.alist_keys(prefix="user:")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys
        assert "config:1" not in user_keys

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """Test clearing all keys."""
        await memory.aset("key1", "value1")
        await memory.aset("key2", "value2")

        await memory.aclear()

        keys = await memory.alist_keys()
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_custom_table_name(self, database):
        """Test using a custom table name."""
        memory = SQLiteMemoryAdapter(database=database, table_name="custom_memory")

        await memory.aset("key", "value")
        value = await memory.aget("key")

        assert value == "value"
        assert memory.table_name == "custom_memory"

    @pytest.mark.asyncio
    async def test_auto_init_disabled(self, database):
        """Test with auto_init disabled."""
        import aiosqlite

        memory = SQLiteMemoryAdapter(database=database, auto_init=False)

        # Should fail because table doesn't exist
        with pytest.raises(aiosqlite.OperationalError):
            await memory.aset("key", "value")

    @pytest.mark.asyncio
    async def test_json_string_storage(self, memory):
        """Test storing JSON strings (like CheckpointState)."""
        json_data = '{"run_id": "123", "data": {"key": "value"}}'

        await memory.aset("checkpoint:123", json_data)
        retrieved = await memory.aget("checkpoint:123")

        assert retrieved == json_data

    @pytest.mark.asyncio
    async def test_multiple_adapters_same_database(self, database):
        """Test multiple memory adapters on same database with different tables."""
        memory1 = SQLiteMemoryAdapter(database=database, table_name="table1")
        memory2 = SQLiteMemoryAdapter(database=database, table_name="table2")

        await memory1.aset("key", "value1")
        await memory2.aset("key", "value2")

        value1 = await memory1.aget("key")
        value2 = await memory2.aget("key")

        assert value1 == "value1"
        assert value2 == "value2"

    @pytest.mark.asyncio
    async def test_empty_list_keys(self, memory):
        """Test listing keys when storage is empty."""
        keys = await memory.alist_keys()
        assert keys == []
