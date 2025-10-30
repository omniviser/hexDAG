"""Integration test for MySQL external plugin discovery and registration.

Setup Instructions:
-------------------
Before running these tests, ensure the MySQL plugin is installed:

    uv pip install -e hexdag_plugins/mysql_adapter/

This will install the MySQL adapter as an editable package, allowing
the tests to discover it as an external plugin.

Note: These tests use mocked MySQL connections, so no actual MySQL
server is required.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from hexdag.core.config.models import ManifestEntry
from hexdag.core.registry import registry as global_registry


class TestMySQLExternalPlugin:
    """Test MySQL adapter as an external plugin."""

    @pytest.fixture(autouse=True, scope="class")
    def ensure_mysql_plugin_installed(self):
        """Ensure MySQL plugin is installed before running tests."""
        try:
            # Try importing to check if installed
            import hexdag_plugins.mysql_adapter  # noqa: F401
        except ImportError:
            pytest.skip(
                "MySQL plugin not installed. Run: uv pip install -e hexdag_plugins/mysql_adapter/"
            )

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Ensure registry is clean before and after each test."""
        if global_registry.ready:
            global_registry._reset_for_testing()
        yield
        if global_registry.ready:
            global_registry._reset_for_testing()

    def test_mysql_external_plugin_discovery(self):
        """Test that MySQL adapter can be discovered as an external plugin."""
        # Add MySQL plugin path to sys.path for import
        if "hexdag_plugins/mysql_adapter" not in sys.path:
            sys.path.insert(0, "hexdag_plugins/mysql_adapter")

        manifest = [
            ManifestEntry(namespace="core", module="hexdag.core.ports"),
            ManifestEntry(namespace="plugin", module="hexdag.builtin.adapters.database.sqlite"),
            ManifestEntry(namespace="plugin", module="mysql_adapter"),
        ]

        global_registry.bootstrap(manifest, dev_mode=False)

        components = global_registry.list_components()
        adapter_names = [c.name for c in components if c.component_type.value == "adapter"]

        # Both adapters should be registered
        assert "sqlite" in adapter_names, "SQLite should be registered as core adapter"
        assert "mysql" in adapter_names, "MySQL should be registered as plugin adapter"

        # Verify namespaces
        sqlite_info = global_registry.get_info("sqlite", namespace="plugin")
        assert sqlite_info.namespace == "plugin"

        mysql_info = global_registry.get_info("mysql", namespace="plugin")
        assert mysql_info.namespace == "plugin"

    @pytest.mark.asyncio
    async def test_mysql_adapter_with_mock_connection(self):
        """Test MySQL adapter operations with mocked connection."""
        if "hexdag_plugins/mysql_adapter" not in sys.path:
            sys.path.insert(0, "hexdag_plugins/mysql_adapter")

        manifest = [
            ManifestEntry(namespace="core", module="hexdag.core.ports"),
            ManifestEntry(namespace="plugin", module="mysql_adapter"),
        ]

        global_registry.bootstrap(manifest, dev_mode=False)

        # Mock pymysql to avoid needing actual MySQL server
        with patch("mysql_adapter.pymysql") as mock_pymysql:
            # Setup mock connection
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.__enter__.return_value = mock_connection
            mock_connection.__exit__.return_value = None
            mock_pymysql.connect.return_value = mock_connection

            # Mock cursor responses
            mock_cursor.fetchone.return_value = None
            mock_cursor.fetchall.return_value = []
            mock_cursor.rowcount = 1

            # Get MySQL adapter from registry
            mysql = global_registry.get(
                "mysql",
                namespace="plugin",
                init_params={
                    "host": "localhost",
                    "user": "test",
                    "password": "test",
                    "database": "test_db",
                },
            )

            # Test basic operations
            doc_id = await mysql.ainsert("test_collection", {"name": "Test"})
            assert doc_id is not None

            # Mock a document retrieval
            mock_cursor.fetchone.return_value = {
                "document": '{"name": "Test", "_id": "' + doc_id + '"}'
            }
            doc = await mysql.aget("test_collection", doc_id)
            assert doc is not None
            assert doc["name"] == "Test"
