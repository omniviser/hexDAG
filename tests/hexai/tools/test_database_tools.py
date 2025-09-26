"""Tests for database tools."""

from typing import Any

import pytest

from hexai.core.config.models import ManifestEntry
from hexai.core.registry.models import ComponentType
from hexai.core.registry.registry import ComponentRegistry
from hexai.tools.database_tools import (
    database_execute,
    database_query,
    database_query_sync,
    describe_table,
    list_tables,
)


class MockDatabasePort:
    """Mock database port for testing."""

    async def aexecute_query(self, sql: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Mock aexecute_query method - matches the DatabasePort interface."""
        if "information_schema.tables" in sql:
            return [{"table_name": "users"}, {"table_name": "products"}]
        elif "information_schema.columns" in sql:
            return [
                {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                {"column_name": "name", "data_type": "varchar", "is_nullable": "YES"},
            ]
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]

    # Keep old methods for backward compatibility
    async def query(self, sql: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Mock query method - delegates to aexecute_query."""
        return await self.aexecute_query(sql, params)

    async def execute(self, sql: str, params: dict | None = None) -> None:
        """Mock execute method."""
        # Execute doesn't return anything, just run the query
        await self.aexecute_query(sql, params)

    def query_sync(self, sql: str, params: dict | None = None) -> list[dict[str, Any]]:
        """Mock synchronous query."""
        return [{"id": 1, "data": "test"}]


class MockLLMPort:
    """Mock LLM port for testing."""

    async def aresponse(self, messages: list) -> list[float]:
        """Mock aresponse method for embeddings - matches the LLM interface."""
        # Return mock embedding vector regardless of input
        return [0.1, 0.2, 0.3]  # Mock embedding vector

    async def generate(self, prompt: str) -> list[float]:
        """Mock generate method for backward compatibility."""
        return [0.1, 0.2, 0.3]  # Mock embedding vector


@pytest.fixture
def mock_database_port():
    """Create a mock database port."""
    return MockDatabasePort()


@pytest.fixture
def mock_llm_port():
    """Create a mock LLM port."""
    return MockLLMPort()


@pytest.fixture
def registry():
    """Create a registry with database tools registered."""
    reg = ComponentRegistry()
    reg.bootstrap(
        manifest=[ManifestEntry(namespace="core", module="hexai.tools.database_tools")],
        dev_mode=True,
    )
    return reg


class TestDatabaseToolsRegistration:
    """Test database tools registration."""

    def test_tools_registered(self, registry):
        """Test all database tools are registered."""
        tools = registry.list_components(component_type=ComponentType.TOOL, namespace="core")
        tool_names = [t.name for t in tools]

        # Check primary names are registered
        assert "sql_query" in tool_names  # Primary name of database_query
        assert "db_execute" in tool_names
        assert "db_tables" in tool_names
        assert "describe_table" in tool_names
        assert "db_query_sync" in tool_names

    def test_port_requirements(self, registry):
        """Test tools have correct port requirements."""
        test_cases = [
            ("sql_query", ["database"]),
            ("db_execute", ["database"]),
            ("db_tables", ["database"]),
            ("describe_table", ["database"]),
            ("db_query_sync", ["database"]),
        ]

        for tool_name, expected_ports in test_cases:
            metadata = registry.get_metadata(tool_name, namespace="core")
            assert metadata.port_requirements == expected_ports

    def test_aliases_resolve_to_same_tool(self, registry):
        """Test that aliases resolve to the same tool."""
        # Get tools by different aliases - sql_query is the primary name
        registry.get("sql_query", namespace="core")

        # db_query_sync and database_query_sync are aliases for the same function
        sync1 = registry.get("db_query_sync", namespace="core")
        sync2 = registry.get("database_query_sync", namespace="core")

        # These should be the same function
        assert sync1 == sync2


class TestDatabaseToolsExecution:
    """Test database tools execution."""

    @pytest.mark.asyncio
    async def test_database_query(self, mock_database_port):
        """Test database query execution."""
        result = await database_query(
            sql="SELECT * FROM users",
            database_port=mock_database_port,
        )

        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_database_execute(self, mock_database_port):
        """Test database execute."""
        # Should not raise
        await database_execute(
            sql="INSERT INTO users (name) VALUES (:name)",
            params={"name": "Charlie"},
            database_port=mock_database_port,
        )

    @pytest.mark.asyncio
    async def test_list_tables(self, mock_database_port):
        """Test listing tables."""
        tables = await list_tables(database_port=mock_database_port)
        assert "users" in tables
        assert "products" in tables

    @pytest.mark.asyncio
    async def test_describe_table(self, mock_database_port):
        """Test describing table schema."""
        schema = await describe_table(
            table="users",
            database_port=mock_database_port,
        )

        assert len(schema) == 2
        assert schema[0]["column_name"] == "id"
        assert schema[0]["data_type"] == "integer"

    def test_database_query_sync(self, mock_database_port):
        """Test synchronous database query."""
        result = database_query_sync(
            sql="SELECT * FROM data",
            database_port=mock_database_port,
        )

        # Now uses asyncio.run(aexecute_query) internally,
        # so it returns the same as aexecute_query
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[1]["name"] == "Bob"


class TestDatabaseToolsValidation:
    """Test database tools validation."""

    @pytest.mark.asyncio
    async def test_missing_database_port(self):
        """Test error when database port is missing."""
        with pytest.raises(ValueError, match="Database port is required"):
            await database_query(sql="SELECT *", database_port=None)
