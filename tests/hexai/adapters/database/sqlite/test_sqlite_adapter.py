"""Unit tests for SQLite adapter."""

import sqlite3
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
    async def test_sqlite_adapter_query_execution(self):
        """Test SQL query execution with SQLite adapter."""
        # Create adapter instance
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create a test table
        await adapter.aexecute_query("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER
            )
        """)

        # Insert data
        await adapter.aexecute_query(
            "INSERT INTO users (name, email, age) VALUES (:name, :email, :age)",
            {"name": "John Doe", "email": "john@example.com", "age": 30},
        )

        # Query data
        results = await adapter.aexecute_query(
            "SELECT * FROM users WHERE email = :email", {"email": "john@example.com"}
        )
        assert len(results) == 1
        assert results[0]["name"] == "John Doe"
        assert results[0]["age"] == 30

        # Update data
        await adapter.aexecute_query(
            "UPDATE users SET age = :age WHERE email = :email",
            {"age": 31, "email": "john@example.com"},
        )

        # Verify update
        results = await adapter.aexecute_query(
            "SELECT age FROM users WHERE email = :email", {"email": "john@example.com"}
        )
        assert results[0]["age"] == 31

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_schema_introspection(self):
        """Test schema introspection capabilities."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create tables with relationships
        await adapter.aexecute_query("""
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)

        await adapter.aexecute_query("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dept_id INTEGER,
                FOREIGN KEY (dept_id) REFERENCES departments(id)
            )
        """)

        # Get table schemas
        schemas = await adapter.aget_table_schemas()
        assert "departments" in schemas
        assert "employees" in schemas

        # Check department schema
        dept_schema = schemas["departments"]
        assert dept_schema["table_name"] == "departments"
        assert "id" in dept_schema["columns"]
        assert "name" in dept_schema["columns"]
        assert "id" in dept_schema["primary_keys"]

        # Check employee schema with foreign key
        emp_schema = schemas["employees"]
        assert emp_schema["table_name"] == "employees"
        assert len(emp_schema["foreign_keys"]) > 0
        fk = emp_schema["foreign_keys"][0]
        assert fk["from_column"] == "dept_id"
        assert fk["to_table"] == "departments"
        assert fk["to_column"] == "id"

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_relationships(self):
        """Test foreign key relationship detection."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create tables with FK relationship
        await adapter.aexecute_query("""
            CREATE TABLE categories (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)

        await adapter.aexecute_query("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category_id INTEGER,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )
        """)

        # Get relationships
        relationships = await adapter.aget_relationships()
        assert len(relationships) > 0

        # Find the product->category relationship
        found = False
        for rel in relationships:
            if rel["from_table"] == "products" and rel["to_table"] == "categories":
                assert rel["from_column"] == "category_id"
                assert rel["to_column"] == "id"
                assert rel["relationship_type"] == "many_to_one"
                found = True
                break
        assert found

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_indexes(self):
        """Test index information retrieval."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create table
        await adapter.aexecute_query("""
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                name TEXT,
                code TEXT UNIQUE
            )
        """)

        # Create an index
        await adapter.aexecute_query("CREATE INDEX idx_items_name ON items(name)")

        # Get indexes
        indexes = await adapter.aget_indexes()

        # Find our created index
        found = False
        for idx in indexes:
            if idx["index_name"] == "idx_items_name":
                assert idx["table_name"] == "items"
                assert "name" in idx["columns"]
                assert idx["index_type"] == "btree"
                assert idx["is_unique"] is False
                found = True
                break
        assert found

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_statistics(self):
        """Test table statistics retrieval."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create table and insert data
        await adapter.aexecute_query("""
            CREATE TABLE records (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)

        for i in range(10):
            await adapter.aexecute_query(
                "INSERT INTO records (data) VALUES (:data)", {"data": f"record_{i}"}
            )

        # Get statistics
        stats = await adapter.aget_table_statistics()
        assert "records" in stats

        record_stats = stats["records"]
        assert record_stats["row_count"] == 10
        assert "size_bytes" in record_stats
        assert record_stats["last_updated"] is None  # SQLite doesn't track this

        adapter.close()

    def test_sqlite_adapter_decorator_metadata(self):
        """Test that SQLite adapter has correct decorator metadata."""
        from hexai.core.registry.models import ComponentType

        assert SQLiteAdapter._hexdag_type == ComponentType.ADAPTER
        assert SQLiteAdapter._hexdag_name == "sqlite"
        assert SQLiteAdapter._hexdag_namespace == "user"
        assert "SQLite database adapter" in SQLiteAdapter._hexdag_description

    @pytest.mark.asyncio
    async def test_sqlite_adapter_error_handling(self):
        """Test error handling in SQLite adapter."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Test query on non-existent table
        with pytest.raises(sqlite3.OperationalError):
            await adapter.aexecute_query("SELECT * FROM non_existent_table")

        # Test invalid SQL
        with pytest.raises(sqlite3.OperationalError):
            await adapter.aexecute_query("INVALID SQL STATEMENT")

        adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_parameterized_queries(self):
        """Test parameterized query support for SQL injection prevention."""
        adapter = SQLiteAdapter(db_path="test_hexdag.db")

        # Create table
        await adapter.aexecute_query("""
            CREATE TABLE test_data (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)

        # Insert with parameters (safe from injection)
        await adapter.aexecute_query(
            "INSERT INTO test_data (value) VALUES (:value)",
            {"value": "'; DROP TABLE test_data; --"},  # Attempted injection
        )

        # Verify table still exists and data was inserted as string
        results = await adapter.aexecute_query("SELECT * FROM test_data")
        assert len(results) == 1
        assert results[0]["value"] == "'; DROP TABLE test_data; --"

        adapter.close()

    def test_sqlite_adapter_repr(self):
        """Test string representation of adapter."""
        adapter = SQLiteAdapter(db_path="test.db")
        assert repr(adapter) == "SQLiteAdapter(db_path='test.db')"
        adapter.close()
