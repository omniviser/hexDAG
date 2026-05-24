"""Tests for MySQL adapter plugin."""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from hexdag_plugins.mysql_adapter.mysql_adapter import MySQLAdapter


class TestMySQLAdapter:
    """Test external MySQL plugin functionality."""

    def test_mysql_plugin_registration_decorator(self):
        """Test that MySQL adapter is properly decorated for registry."""
        # Check that the adapter has registry attributes
        assert hasattr(MySQLAdapter, "_hexdag_type")
        assert hasattr(MySQLAdapter, "_hexdag_name")
        assert hasattr(MySQLAdapter, "_hexdag_namespace")
        assert hasattr(MySQLAdapter, "_hexdag_description")

        assert MySQLAdapter._hexdag_type == "adapter"
        assert MySQLAdapter._hexdag_name == "mysql"
        assert MySQLAdapter._hexdag_namespace == "plugin"
        assert "MySQL database adapter" in MySQLAdapter._hexdag_description

    @pytest.mark.asyncio
    async def test_mysql_adapter_with_mock_connection(self):
        """Test MySQL adapter operations with mocked connection."""
        # Mock pymysql to avoid needing actual MySQL server
        with patch("hexdag_plugins.mysql_adapter.mysql_adapter.pymysql") as mock_pymysql:
            # Setup mock connection
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.__enter__.return_value = mock_connection
            mock_connection.__exit__.return_value = None
            mock_pymysql.connect.return_value = mock_connection

            # Mock cursor responses
            mock_cursor.fetchone.return_value = None  # For initial checks
            mock_cursor.fetchall.return_value = []
            mock_cursor.rowcount = 1

            # Create adapter - this will try to create database and tables
            adapter = MySQLAdapter(
                host="localhost", user="test", password="test", database="test_hexdag"
            )

            # Test insert
            doc_id = await adapter.ainsert(
                "test_collection", {"name": "Test Document", "value": 42}
            )
            assert doc_id is not None

            # Test get (mock returns a document)
            mock_cursor.fetchone.return_value = {
                "document": '{"name": "Test Document", "value": 42}'
            }
            doc = await adapter.aget("test_collection", doc_id)
            assert doc is not None
            assert doc["name"] == "Test Document"

            # Test query
            mock_cursor.fetchall.return_value = [
                {"id": "1", "document": '{"name": "Doc1"}'},
                {"id": "2", "document": '{"name": "Doc2"}'},
            ]
            docs = await adapter.aquery("test_collection")
            assert len(docs) == 2

            # Test update
            result = await adapter.aupdate("test_collection", doc_id, {"status": "updated"})
            assert result is True

            # Test delete
            result = await adapter.adelete("test_collection", doc_id)
            assert result is True

            # Test count
            mock_cursor.fetchone.return_value = {"count": 5}
            count = await adapter.acount("test_collection")
            assert count == 5

            # Test list collections
            mock_cursor.fetchall.return_value = [{"collection": "col1"}, {"collection": "col2"}]
            collections = await adapter.alist_collections()
            assert len(collections) == 2

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("MYSQL_TEST_HOST"), reason="MySQL test server not configured"
    )
    async def test_mysql_adapter_with_real_server(self):
        """Test MySQL adapter with real MySQL server (if available).

        Set environment variables to run this test:
        - MYSQL_TEST_HOST: MySQL server host
        - MYSQL_TEST_USER: MySQL user
        - MYSQL_TEST_PASSWORD: MySQL password
        - MYSQL_TEST_DATABASE: Test database name
        """
        adapter = MySQLAdapter(
            host=os.environ.get("MYSQL_TEST_HOST", "localhost"),
            user=os.environ.get("MYSQL_TEST_USER", "root"),
            password=os.environ.get("MYSQL_TEST_PASSWORD", ""),
            database=os.environ.get("MYSQL_TEST_DATABASE", "test_hexdag"),
        )

        # Clean up test collection
        await adapter.adrop_collection("test_mysql")

        # Test full CRUD cycle
        # Insert
        doc_id = await adapter.ainsert(
            "test_mysql",
            {"title": "MySQL Test", "type": "integration", "tags": ["mysql", "test", "database"]},
        )
        assert doc_id is not None

        # Get
        doc = await adapter.aget("test_mysql", doc_id)
        assert doc is not None
        assert doc["title"] == "MySQL Test"
        assert doc["_id"] == doc_id

        # Update
        updated = await adapter.aupdate("test_mysql", doc_id, {"status": "verified", "score": 100})
        assert updated is True

        # Verify update
        doc = await adapter.aget("test_mysql", doc_id)
        assert doc["status"] == "verified"
        assert doc["score"] == 100
        assert doc["title"] == "MySQL Test"  # Original field preserved

        # Query
        docs = await adapter.aquery("test_mysql", filter={"type": "integration"})
        assert len(docs) >= 1
        assert any(d["_id"] == doc_id for d in docs)

        # Count
        count = await adapter.acount("test_mysql")
        assert count >= 1

        # List collections
        collections = await adapter.alist_collections()
        assert "test_mysql" in collections

        # Delete
        deleted = await adapter.adelete("test_mysql", doc_id)
        assert deleted is True

        # Verify deletion
        doc = await adapter.aget("test_mysql", doc_id)
        assert doc is None

        # Clean up
        await adapter.adrop_collection("test_mysql")

    @pytest.mark.asyncio
    async def test_mysql_concurrent_operations(self):
        """Test concurrent operations with MySQL adapter."""
        with patch("hexdag_plugins.mysql_adapter.mysql_adapter.pymysql") as mock_pymysql:
            # Setup mock
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.__enter__.return_value = mock_connection
            mock_connection.__exit__.return_value = None
            mock_pymysql.connect.return_value = mock_connection

            mock_cursor.fetchone.return_value = None
            mock_cursor.fetchall.return_value = []
            mock_cursor.rowcount = 1

            adapter = MySQLAdapter()

            # Concurrent inserts
            async def insert_doc(i):
                return await adapter.ainsert("concurrent", {"id": f"doc_{i}", "index": i})

            # Insert 10 documents concurrently
            doc_ids = await asyncio.gather(*[insert_doc(i) for i in range(10)])

            assert len(doc_ids) == 10
            assert all(doc_id is not None for doc_id in doc_ids)

    def test_mysql_connection_parameters(self):
        """Test MySQL adapter initialization with various parameters."""
        with patch("hexdag_plugins.mysql_adapter.mysql_adapter.pymysql") as mock_pymysql:
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.__enter__.return_value = mock_connection
            mock_connection.__exit__.return_value = None
            mock_pymysql.connect.return_value = mock_connection

            # Test with custom parameters
            adapter = MySQLAdapter(
                host="db.example.com",
                port=3307,
                user="custom_user",
                password="secure_pass",
                database="custom_db",
                charset="utf8",
                connect_timeout=30,
            )

            # Verify connection parameters
            assert adapter.connection_params["host"] == "db.example.com"
            assert adapter.connection_params["port"] == 3307
            assert adapter.connection_params["user"] == "custom_user"
            assert adapter.connection_params["database"] == "custom_db"
            assert adapter.connection_params["connect_timeout"] == 30

    @pytest.mark.asyncio
    async def test_mysql_json_handling(self):
        """Test MySQL adapter with complex JSON data."""
        with patch("hexdag_plugins.mysql_adapter.mysql_adapter.pymysql") as mock_pymysql:
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.__enter__.return_value = mock_connection
            mock_connection.__exit__.return_value = None
            mock_pymysql.connect.return_value = mock_connection

            mock_cursor.fetchone.return_value = None
            mock_cursor.rowcount = 1

            adapter = MySQLAdapter()

            # Complex nested data
            complex_data = {
                "id": "complex1",
                "metadata": {
                    "author": "Test Author",
                    "tags": ["tag1", "tag2", "tag3"],
                    "settings": {"enabled": True, "threshold": 0.85, "options": ["opt1", "opt2"]},
                },
                "data": {
                    "values": [1, 2, 3, 4, 5],
                    "nested": {"level1": {"level2": {"level3": "deep value"}}},
                },
            }

            # Test insert with complex data
            doc_id = await adapter.ainsert("complex", complex_data)
            assert doc_id is not None

            # Mock retrieval
            import json

            mock_cursor.fetchone.return_value = {"document": json.dumps(complex_data)}

            doc = await adapter.aget("complex", doc_id)
            assert doc is not None
            assert doc["metadata"]["settings"]["threshold"] == 0.85
            assert doc["data"]["nested"]["level1"]["level2"]["level3"] == "deep value"
