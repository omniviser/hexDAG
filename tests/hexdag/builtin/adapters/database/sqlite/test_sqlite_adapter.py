"""Unit tests for SQLite adapter."""

import pathlib
import sqlite3
import tempfile

import pytest

from hexdag.builtin.adapters.database.sqlite.sqlite_adapter import SQLiteAdapter


class TestSQLiteAdapter:
    """Test SQLite adapter functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary database path that gets cleaned up.

        Yields
        ------
        str
            Path to a temporary database file
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Clean up after test
        if pathlib.Path(db_path).exists():
            pathlib.Path(db_path).unlink()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_query_execution(self, temp_db_path):
        """Test SQL query execution with SQLite adapter."""
        # Create adapter instance
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_schema_introspection(self, temp_db_path):
        """Test schema introspection capabilities."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_relationships(self, temp_db_path):
        """Test foreign key relationship detection."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_indexes(self, temp_db_path):
        """Test index information retrieval."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_statistics(self, temp_db_path):
        """Test table statistics retrieval."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    def test_sqlite_adapter_decorator_metadata(self):
        """Test that SQLite adapter has correct decorator metadata."""
        from hexdag.core.registry.models import ComponentType

        assert SQLiteAdapter._hexdag_type == ComponentType.ADAPTER
        assert SQLiteAdapter._hexdag_name == "sqlite"
        assert SQLiteAdapter._hexdag_namespace == "user"
        assert "SQLite database adapter" in SQLiteAdapter._hexdag_description

    @pytest.mark.asyncio
    async def test_sqlite_adapter_error_handling(self, temp_db_path):
        """Test error handling in SQLite adapter."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

        # Test query on non-existent table
        with pytest.raises(sqlite3.OperationalError):
            await adapter.aexecute_query("SELECT * FROM non_existent_table")

        # Test invalid SQL
        with pytest.raises(sqlite3.OperationalError):
            await adapter.aexecute_query("INVALID SQL STATEMENT")

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_parameterized_queries(self, temp_db_path):
        """Test parameterized query support for SQL injection prevention."""
        adapter = SQLiteAdapter(db_path=temp_db_path)

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

        await adapter.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_repr(self, temp_db_path):
        """Test string representation of adapter."""
        adapter = SQLiteAdapter(db_path=temp_db_path)
        assert "SQLiteAdapter" in repr(adapter)
        assert temp_db_path in repr(adapter)
        assert "mode='read-write'" in repr(adapter)
        await adapter.close()

        # Test with read-only mode
        adapter_ro = SQLiteAdapter(db_path=temp_db_path, read_only=True)
        assert "SQLiteAdapter" in repr(adapter_ro)
        assert temp_db_path in repr(adapter_ro)
        assert "mode='read-only'" in repr(adapter_ro)
        await adapter_ro.close()

    @pytest.mark.asyncio
    async def test_sqlite_adapter_read_only_mode(self):
        """Test read-only mode prevents modifications."""
        import tempfile

        # Create a database with data
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # First, create database and add data in read-write mode
            adapter_rw = SQLiteAdapter(db_path=db_path, read_only=False)

            # Create table and insert data
            await adapter_rw.aexecute_query("""
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)

            await adapter_rw.aexecute_query(
                "INSERT INTO test_data (value) VALUES (:value)", {"value": "test_value"}
            )

            await adapter_rw.close()

            # Now open in read-only mode
            adapter_ro = SQLiteAdapter(db_path=db_path, read_only=True)

            # SELECT should work
            results = await adapter_ro.aexecute_query("SELECT * FROM test_data")
            assert len(results) == 1
            assert results[0]["value"] == "test_value"

            # INSERT should fail
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await adapter_ro.aexecute_query(
                    "INSERT INTO test_data (value) VALUES (:value)", {"value": "should_fail"}
                )
            # SQLite returns "attempt to write a readonly database"
            assert "readonly" in str(exc_info.value).lower()

            # UPDATE should fail
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await adapter_ro.aexecute_query(
                    "UPDATE test_data SET value = :value WHERE id = 1", {"value": "should_fail"}
                )
            assert "readonly" in str(exc_info.value).lower()

            # DELETE should fail
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await adapter_ro.aexecute_query("DELETE FROM test_data WHERE id = 1")
            assert "readonly" in str(exc_info.value).lower()

            # CREATE TABLE should fail
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await adapter_ro.aexecute_query("CREATE TABLE new_table (id INTEGER)")
            assert "readonly" in str(exc_info.value).lower()

            await adapter_ro.close()

        finally:
            # Clean up
            pathlib.Path(db_path).unlink()
