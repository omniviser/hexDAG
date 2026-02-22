"""Unit tests for PGVector adapter read-only mode."""

import pytest

from hexdag.stdlib.adapters.database.pgvector.pgvector_adapter import PgVectorAdapter


class TestPgVectorReadOnly:
    """Test PGVector adapter read-only functionality."""

    @pytest.mark.asyncio
    async def test_pgvector_read_only_initialization(self):
        """Test that adapter can be initialized in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Verify read-only status
        assert await adapter.is_read_only()
        assert adapter.read_only is True

        # Check string representation includes mode
        repr_str = repr(adapter)
        assert "read-only" in repr_str
        assert "localhost" in repr_str
        assert "test_db" in repr_str

    @pytest.mark.asyncio
    async def test_pgvector_read_write_initialization(self):
        """Test that adapter defaults to read-write mode."""
        adapter = PgVectorAdapter(password="test_password", host="localhost", database="test_db")

        # Verify read-write status
        assert not await adapter.is_read_only()
        assert adapter.read_only is False

        # Check string representation includes mode
        repr_str = repr(adapter)
        assert "read-write" in repr_str

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_vector_upsert(self):
        """Test that vector upsert is blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt vector upsert should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.avector_upsert(
                collection="test_collection",
                vectors=[{"id": "1", "vector": [0.1, 0.2, 0.3]}],
            )

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "vector upsert" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_vector_delete(self):
        """Test that vector delete is blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt vector delete should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.avector_delete(collection="test_collection", ids=["1", "2"])

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "vector delete" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_insert_query(self):
        """Test that INSERT queries are blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt INSERT query should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query(
                "INSERT INTO test_table (id, value) VALUES (:id, :value)",
                {"id": 1, "value": "test"},
            )

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "INSERT" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_update_query(self):
        """Test that UPDATE queries are blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt UPDATE query should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query(
                "UPDATE test_table SET value = :value WHERE id = :id",
                {"value": "new_value", "id": 1},
            )

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "UPDATE" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_delete_query(self):
        """Test that DELETE queries are blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt DELETE query should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query("DELETE FROM test_table WHERE id = :id", {"id": 1})

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "DELETE" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_create_query(self):
        """Test that CREATE queries are blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt CREATE query should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query(
                "CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)"
            )

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "CREATE" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_prevents_drop_query(self):
        """Test that DROP queries are blocked in read-only mode."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # Attempt DROP query should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query("DROP TABLE test_table")

        error_msg = str(exc_info.value)
        assert "read-only mode" in error_msg
        assert "DROP" in error_msg

    @pytest.mark.asyncio
    async def test_pgvector_read_only_allows_select_query(self):
        """Test that SELECT queries work in read-only mode (when connected)."""
        adapter = PgVectorAdapter(
            password="test_password",
            host="localhost",
            database="test_db",
            read_only=True,
        )

        # SELECT should not be blocked by read-only check
        # (Will fail with "Not connected" but that's expected)
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.aexecute_query("SELECT * FROM test_table")

        # Should fail due to no connection, not read-only mode
        error_msg = str(exc_info.value)
        assert "Not connected" in error_msg
        assert "read-only" not in error_msg
