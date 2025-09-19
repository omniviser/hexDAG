"""Unit tests for SQLite adapter."""

import asyncio
from pathlib import Path

import pytest

from hexai.adapters.database.sqlite import SQLiteAdapter


class TestSQLiteAdapter:
    """Test SQLite adapter functionality."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up test database."""
        # Clean up any test database files
        test_db = Path("test_hexdag.db")
        if test_db.exists():
            test_db.unlink()

        yield

        # Clean up after test
        if test_db.exists():
            test_db.unlink()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_basic_operations(self):
        """Test basic CRUD operations with SQLite adapter."""
        # Create adapter instance
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Test insert
        doc_id = await adapter.ainsert(
            "users", {"name": "John Doe", "email": "john@example.com", "age": 30}
        )
        assert doc_id is not None

        # Test get
        user = await adapter.aget("users", doc_id)
        assert user is not None
        assert user["name"] == "John Doe"
        assert user["email"] == "john@example.com"

        # Test update
        success = await adapter.aupdate("users", doc_id, {"age": 31, "city": "New York"})
        assert success is True

        # Verify update
        updated_user = await adapter.aget("users", doc_id)
        assert updated_user["age"] == 31
        assert updated_user["city"] == "New York"
        assert updated_user["name"] == "John Doe"  # Original field preserved

        # Test delete
        deleted = await adapter.adelete("users", doc_id)
        assert deleted is True

        # Verify deletion
        deleted_user = await adapter.aget("users", doc_id)
        assert deleted_user is None

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_query_operations(self):
        """Test query operations with SQLite adapter."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Insert multiple documents
        await adapter.ainsert(
            "products", {"id": "1", "name": "Laptop", "category": "Electronics", "price": 999}
        )
        await adapter.ainsert(
            "products", {"id": "2", "name": "Mouse", "category": "Electronics", "price": 29}
        )
        await adapter.ainsert(
            "products", {"id": "3", "name": "Desk", "category": "Furniture", "price": 299}
        )

        # Test query all
        all_products = await adapter.aquery("products")
        assert len(all_products) == 3

        # Test query with filter
        electronics = await adapter.aquery("products", filter={"category": "Electronics"})
        assert len(electronics) == 2

        # Test query with limit
        limited = await adapter.aquery("products", limit=2)
        assert len(limited) == 2

        # Test count
        count = await adapter.acount("products")
        assert count == 3

        # Test list collections
        collections = await adapter.alist_collections()
        assert "products" in collections

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_concurrent_operations(self):
        """Test concurrent operations with SQLite adapter."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Prepare concurrent inserts
        async def insert_doc(i):
            return await adapter.ainsert(
                "concurrent", {"id": f"doc_{i}", "value": i, "data": f"test_{i}"}
            )

        # Insert 10 documents concurrently
        doc_ids = await asyncio.gather(*[insert_doc(i) for i in range(10)])

        assert len(doc_ids) == 10

        # Verify all were inserted
        count = await adapter.acount("concurrent")
        assert count == 10

        # Test concurrent reads
        async def read_doc(doc_id):
            return await adapter.aget("concurrent", doc_id)

        docs = await asyncio.gather(*[read_doc(doc_id) for doc_id in doc_ids])

        assert len(docs) == 10
        assert all(doc is not None for doc in docs)

        adapter.close()

    def test_sqlite_adapter_persistence(self):
        """Test that SQLite adapter persists data between sessions."""
        # First session - write data
        adapter1 = SQLiteAdapter(db_path="test_hexdag.db")

        async def write_data():
            await adapter1.ainsert("persistent", {"id": "test1", "data": "This should persist"})

        asyncio.run(write_data())
        adapter1.close()

        # Second session - read data
        adapter2 = SQLiteAdapter(db_path="test_hexdag.db")

        async def read_data():
            docs = await adapter2.aquery("persistent", filter={"id": "test1"})
            assert len(docs) == 1
            assert docs[0]["data"] == "This should persist"

        asyncio.run(read_data())
        adapter2.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_json_handling(self):
        """Test SQLite adapter handles complex JSON data."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Complex nested data
        complex_data = {
            "id": "complex1",
            "user": {
                "name": "Alice",
                "profile": {"age": 28, "interests": ["coding", "music", "travel"]},
            },
            "metadata": {
                "created": "2024-01-01",
                "tags": ["important", "verified"],
                "settings": {"notifications": True, "theme": "dark"},
            },
        }

        # Insert complex data
        doc_id = await adapter.ainsert("complex", complex_data)

        # Retrieve and verify
        retrieved = await adapter.aget("complex", doc_id)
        assert retrieved["user"]["name"] == "Alice"
        assert retrieved["user"]["profile"]["age"] == 28
        assert "coding" in retrieved["user"]["profile"]["interests"]
        assert retrieved["metadata"]["settings"]["theme"] == "dark"

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_in_memory(self):
        """Test SQLite adapter with in-memory database."""
        # Use :memory: for in-memory database
        adapter = SQLiteAdapter(db_path=":memory:")

        # Test basic operations
        doc_id = await adapter.ainsert("test", {"key": "value"})
        assert doc_id is not None

        doc = await adapter.aget("test", doc_id)
        assert doc["key"] == "value"

        # No persistence for in-memory
        adapter.close()

    def test_sqlite_adapter_decorator_metadata(self):
        """Test that SQLite adapter has proper registry attributes."""
        # Check that the adapter has registry attributes
        assert hasattr(SQLiteAdapter, "_hexdag_type")

        from hexai.core.registry.models import ComponentType

        assert SQLiteAdapter._hexdag_type == ComponentType.ADAPTER
        assert SQLiteAdapter._hexdag_name == "sqlite"
        assert SQLiteAdapter._hexdag_namespace == "database"
        assert "SQLite database adapter" in SQLiteAdapter._hexdag_description

    @pytest.mark.asyncio
    async def test_sqlite_adapter_error_handling(self):
        """Test error handling in SQLite adapter."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Test get with non-existent ID
        result = await adapter.aget("nonexistent", "fake_id")
        assert result is None

        # Test update with non-existent ID
        success = await adapter.aupdate("nonexistent", "fake_id", {"data": "test"})
        assert success is False

        # Test delete with non-existent ID
        success = await adapter.adelete("nonexistent", "fake_id")
        assert success is False

        # Test query on non-existent collection
        results = await adapter.aquery("nonexistent")
        assert results == []

        # Test count on non-existent collection
        count = await adapter.acount("nonexistent")
        assert count == 0

        adapter.close()
