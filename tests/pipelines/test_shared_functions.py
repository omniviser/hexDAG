"""Tests for shared pipeline functions."""

from unittest.mock import Mock

import pytest

from pipelines.shared_functions import (
    DatabaseSchemaInput,
    DatabaseSchemaOutput,
    UserQueryInput,
    UserQueryOutput,
    extract_user_query,
    load_database_schema,
)


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self):
        """Initialize the mock event manager."""
        self.traces = []
        self.memory = {}

    def add_trace(self, node_name: str, message: str):
        """Add a trace message."""
        self.traces.append(f"{node_name}: {message}")

    def set_memory(self, key: str, value):
        """Set a memory value."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Get a memory value."""
        return self.memory.get(key, default)


class TestSharedFunctions:
    """Tests for shared pipeline functions."""

    @pytest.mark.asyncio
    async def test_extract_user_query_with_user_query(self):
        """Test extract_user_query with user_query field."""
        event_manager = MockEventManager()
        input_data = UserQueryInput(user_query="Find all customers", question=None)

        result = await extract_user_query(input_data, event_manager=event_manager)

        assert isinstance(result, UserQueryOutput)
        assert result.user_query == "Find all customers"
        assert result.original_input == input_data.model_dump()
        assert result.metadata["node_name"] == "extract_user_query"
        assert any("Extracting user query" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_extract_user_query_with_question_field(self):
        """Test extract_user_query with question field."""
        event_manager = MockEventManager()
        input_data = UserQueryInput(user_query="", question="What are the sales?")

        result = await extract_user_query(input_data, event_manager=event_manager)

        assert isinstance(result, UserQueryOutput)
        assert result.user_query == "What are the sales?"
        assert result.original_input == input_data.model_dump()

    @pytest.mark.asyncio
    async def test_extract_user_query_empty_query(self):
        """Test extract_user_query with empty query."""
        event_manager = MockEventManager()
        input_data = UserQueryInput(user_query="", question=None)

        result = await extract_user_query(input_data, event_manager=event_manager)

        assert isinstance(result, UserQueryOutput)
        assert result.user_query == ""
        assert any("No user query found" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_extract_user_query_with_metadata(self):
        """Test extract_user_query with metadata."""
        event_manager = MockEventManager()
        input_data = UserQueryInput(
            user_query="Test query", question=None, metadata={"source": "test", "priority": "high"}
        )

        result = await extract_user_query(input_data, event_manager=event_manager)

        assert isinstance(result, UserQueryOutput)
        assert result.user_query == "Test query"
        assert result.original_input["metadata"] == {"source": "test", "priority": "high"}

    @pytest.mark.asyncio
    async def test_load_database_schema_with_port(self):
        """Test load_database_schema with database port."""
        event_manager = MockEventManager()

        # Mock database port with all methods - using dict format like MockDatabaseAdapter
        mock_db_port = Mock()
        mock_db_port.get_table_schemas.return_value = {
            "users": {
                "table_name": "users",
                "columns": {"id": "INTEGER", "name": "VARCHAR(255)"},
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
            "orders": {
                "table_name": "orders",
                "columns": {"id": "INTEGER", "user_id": "INTEGER"},
                "primary_keys": ["id"],
                "foreign_keys": [
                    {"column": "user_id", "references_table": "users", "references_column": "id"}
                ],
            },
        }
        mock_db_port.get_relationships.return_value = [
            {
                "from_table": "orders",
                "from_column": "user_id",
                "to_table": "users",
                "to_column": "id",
            }
        ]
        mock_db_port.get_indexes.return_value = [{"table": "users", "column": "id"}]
        mock_db_port.get_table_statistics.return_value = {"users": {"row_count": 100}}

        input_data = DatabaseSchemaInput(user_query="Show me the database schema", question=None)
        ports = {"database": mock_db_port, "event_manager": event_manager}

        result = await load_database_schema(input_data, **ports)

        # Check result type
        assert isinstance(result, DatabaseSchemaOutput)

        # Check table schemas format
        assert len(result.table_schemas) == 2
        assert result.table_count == 2
        assert result.relationship_count == 1

        # Check table schemas content
        users_schema = next(s for s in result.table_schemas if s.table_name == "users")
        assert users_schema.columns == ["id", "name"]
        assert users_schema.primary_keys == ["id"]

        orders_schema = next(s for s in result.table_schemas if s.table_name == "orders")
        assert orders_schema.columns == ["id", "user_id"]
        assert orders_schema.primary_keys == ["id"]

        # Check relationships
        assert len(result.relationships) == 1
        relationship = result.relationships[0]
        assert relationship.from_table == "orders"
        assert relationship.from_column == "user_id"
        assert relationship.to_table == "users"
        assert relationship.to_column == "id"

        # Check foreign keys strings
        assert len(result.foreign_keys) == 1
        assert "orders.user_id -> users.id" in result.foreign_keys[0]

        # Check additional data
        assert result.tables == mock_db_port.get_table_schemas.return_value
        assert result.indexes == [{"table": "users", "column": "id"}]
        assert result.statistics == {"users": {"row_count": 100}}
        assert "2 tables" in result.summary
        assert result.error is None
        assert result.metadata["node_name"] == "load_database_schema"

    @pytest.mark.asyncio
    async def test_load_database_schema_without_port(self):
        """Test load_database_schema without database port (fallback)."""
        event_manager = MockEventManager()
        input_data = DatabaseSchemaInput(user_query="Show me the database schema", question=None)
        ports = {"event_manager": event_manager}

        result = await load_database_schema(input_data, **ports)

        # Check result type
        assert isinstance(result, DatabaseSchemaOutput)

        # Should return MockDatabaseAdapter data (4 tables)
        assert len(result.table_schemas) == 4  # MockDatabaseAdapter has 4 tables
        assert result.table_count == 4
        assert "4 tables" in result.summary
        assert result.error is None

    @pytest.mark.asyncio
    async def test_load_database_schema_with_error(self):
        """Test load_database_schema with database error."""
        event_manager = MockEventManager()

        # Mock database port that raises an exception
        mock_db_port = Mock()
        mock_db_port.get_table_schemas.side_effect = Exception("Database connection failed")

        input_data = DatabaseSchemaInput(user_query="Show me the database schema", question=None)
        ports = {"database": mock_db_port, "event_manager": event_manager}

        result = await load_database_schema(input_data, **ports)

        # Check result type
        assert isinstance(result, DatabaseSchemaOutput)

        # Should return error state
        assert result.error is not None
        assert "Database connection failed" in result.error
        assert result.table_count == 0
        assert result.relationship_count == 0
        assert result.summary == "Failed to load database schema"
        assert result.metadata["error"] == result.error

    @pytest.mark.asyncio
    async def test_load_database_schema_with_list_format(self):
        """Test load_database_schema with list format schemas."""
        event_manager = MockEventManager()

        # Mock database port that returns list format
        mock_db_port = Mock()
        mock_db_port.get_table_schemas.return_value = [
            {
                "table_name": "users",
                "name": "users",
                "columns": ["id", "name"],
                "column_types": {"id": "INTEGER", "name": "VARCHAR(255)"},
                "primary_keys": ["id"],
                "foreign_keys": [],
            }
        ]
        mock_db_port.get_relationships.return_value = []
        mock_db_port.get_indexes.return_value = {}
        mock_db_port.get_table_statistics.return_value = {}

        input_data = DatabaseSchemaInput(user_query="Show me the database schema", question=None)
        ports = {"database": mock_db_port, "event_manager": event_manager}

        result = await load_database_schema(input_data, **ports)

        # Check result type
        assert isinstance(result, DatabaseSchemaOutput)

        # Check table schemas
        assert len(result.table_schemas) == 1
        assert result.table_schemas[0].table_name == "users"
        assert result.table_schemas[0].columns == ["id", "name"]

    @pytest.mark.asyncio
    async def test_functions_are_reusable(self):
        """Test that functions can be used by different pipelines."""
        event_manager = MockEventManager()

        # Test that the same function works for different input types
        input_data1 = UserQueryInput(user_query="Test query 1", question=None)
        input_data2 = UserQueryInput(user_query="Test query 2", question=None)

        # Should work for different queries
        result1 = await extract_user_query(input_data1, event_manager=event_manager)
        result2 = await extract_user_query(input_data2, event_manager=event_manager)

        # Results should be properly typed
        assert isinstance(result1, UserQueryOutput)
        assert isinstance(result2, UserQueryOutput)
        assert result1.user_query == "Test query 1"
        assert result2.user_query == "Test query 2"

    @pytest.mark.asyncio
    async def test_event_manager_integration(self):
        """Test that functions properly integrate with event manager."""
        event_manager = MockEventManager()
        input_data = UserQueryInput(user_query="Test query with events", question=None)

        await extract_user_query(input_data, event_manager=event_manager)

        # Check that traces were added
        assert len(event_manager.traces) > 0
        assert any("extract_user_query" in trace for trace in event_manager.traces)
        assert any("Extracting user query" in trace for trace in event_manager.traces)

    @pytest.mark.asyncio
    async def test_database_schema_event_manager_integration(self):
        """Test that load_database_schema properly integrates with event manager."""
        event_manager = MockEventManager()

        # Mock database port
        mock_db_port = Mock()
        mock_db_port.get_table_schemas.return_value = {
            "users": {
                "table_name": "users",
                "columns": {"id": "INTEGER"},
                "primary_keys": ["id"],
                "foreign_keys": [],
            }
        }
        mock_db_port.get_relationships.return_value = []
        mock_db_port.get_indexes.return_value = {}
        mock_db_port.get_table_statistics.return_value = {}

        input_data = DatabaseSchemaInput(user_query="Test schema loading", question=None)
        ports = {"database": mock_db_port, "event_manager": event_manager}

        await load_database_schema(input_data, **ports)

        # Check that traces were added
        assert len(event_manager.traces) > 0
        assert any("load_database_schema" in trace for trace in event_manager.traces)

        # Check that memory was set
        assert "table_schemas" in event_manager.memory
        assert "relationships" in event_manager.memory
