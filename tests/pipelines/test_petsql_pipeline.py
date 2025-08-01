"""Tests for PetSQL pipeline functionality."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipelines.petsql.functions import (
    FilterSchemaInput,
    SelectExamplesInput,
    SqlConverterInput,
    SqlValidatorInput,
    convert_sqlite_to_postgres,
    filter_relevant_schema,
    select_examples,
    sql_validator,
)
from pipelines.petsql.pipeline import PetSQLPipeline


class MockEventManager:
    """Mock event manager for testing."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or "test-session-123"
        self.traces = []
        self.memory = {}

    async def add_trace(self, node_name: str, message: str):
        """Mock add_trace that stores traces."""
        self.traces.append(f"[{node_name}] {message}")

    async def set_memory(self, key: str, value):
        """Mock set_memory that stores in memory dict."""
        self.memory[key] = value

    def get_memory(self, key: str, default=None):
        """Mock get_memory that retrieves from memory dict."""
        return self.memory.get(key, default)


class TestSelectExamples:
    """Test select_examples function."""

    @pytest.mark.asyncio
    async def test_select_examples_basic(self):
        """Test select_examples with basic input."""
        event_manager = MockEventManager()
        input_data = SelectExamplesInput(
            user_query="Show me all customers",
            database_schema={"customers": {"columns": ["id", "name"]}},
        )

        ports = {"event_manager": event_manager}
        result = await select_examples(input_data, **ports)

        assert hasattr(result, "examples")
        assert hasattr(result, "metadata")
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_select_examples_with_llm(self):
        """Test select_examples with LLM integration."""
        event_manager = MockEventManager()
        mock_llm = AsyncMock()
        mock_llm.aresponse.return_value = json.dumps({"examples": ["example1", "example2"]})

        input_data = SelectExamplesInput(
            user_query="Show me all customers",
            database_schema={"customers": {"columns": ["id", "name"]}},
        )

        ports = {"event_manager": event_manager, "llm": mock_llm}
        result = await select_examples(input_data, **ports)

        assert hasattr(result, "examples")
        assert hasattr(result, "metadata")


class TestFilterRelevantSchema:
    """Test filter_relevant_schema function."""

    @pytest.mark.asyncio
    async def test_filter_relevant_schema_basic(self):
        """Test filter_relevant_schema with basic input."""
        event_manager = MockEventManager()
        mock_database = MagicMock()
        mock_database.get_schema.return_value = {
            "customers": {"columns": ["id", "name"]},
        }

        input_data = FilterSchemaInput(
            user_query="Show me customers",
            examples=[{"sql": "SELECT * FROM customers;"}],
        )

        ports = {"event_manager": event_manager, "database": mock_database}
        result = await filter_relevant_schema(input_data, **ports)

        assert hasattr(result, "filtered_schema")
        assert hasattr(result, "metadata")
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_filter_relevant_schema_with_ontology_context(self):
        """Test schema filtering with ontology context."""
        event_manager = MockEventManager()
        mock_database = MagicMock()
        mock_database.get_schema.return_value = {
            "customers": {"columns": ["id", "name"]},
            "orders": {"columns": ["id", "customer_id"]},
        }
        mock_database.filter_schema_by_context.return_value = ["customers"]

        input_data = FilterSchemaInput(
            user_query="Customer information",
            ontology_context=[{"db_table": "customers"}],
        )

        ports = {"event_manager": event_manager, "database": mock_database}
        result = await filter_relevant_schema(input_data, **ports)

        assert hasattr(result, "filtered_schema")
        mock_database.filter_schema_by_context.assert_called_once()


class TestSqlValidator:
    """Test sql_validator function."""

    @pytest.mark.asyncio
    async def test_sql_validator_valid_sql(self):
        """Test SQL validation with valid SQL."""
        event_manager = MockEventManager()
        input_data = SqlValidatorInput(
            sql_query="SELECT * FROM customers WHERE name = 'John';",
            database_schema={"customers": {"columns": ["id", "name"]}},
        )

        ports = {"event_manager": event_manager}
        result = await sql_validator(input_data, **ports)

        assert hasattr(result, "validated_sql")
        assert hasattr(result, "table_names")
        assert hasattr(result, "metadata")
        assert result.table_names == ["customers"]
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_sql_validator_invalid_sql(self):
        """Test SQL validation with invalid SQL."""
        event_manager = MockEventManager()
        input_data = SqlValidatorInput(
            sql_query="INVALID SQL QUERY;",
            database_schema={"customers": {"columns": ["id", "name"]}},
        )

        ports = {"event_manager": event_manager}
        result = await sql_validator(input_data, **ports)

        assert hasattr(result, "validated_sql")
        assert hasattr(result, "table_names")
        assert hasattr(result, "metadata")
        # Invalid SQL should still return the original query
        assert result.validated_sql == "INVALID SQL QUERY;"


class TestConvertSqliteToPostgres:
    """Test convert_sqlite_to_postgres function."""

    @pytest.mark.asyncio
    async def test_convert_sqlite_to_postgres_basic(self):
        """Test basic SQLite to PostgreSQL conversion."""
        event_manager = MockEventManager()
        input_data = SqlConverterInput(
            sql_query="SELECT * FROM customers LIMIT 10;",
            table_names=["customers"],
        )

        ports = {"event_manager": event_manager}
        result = await convert_sqlite_to_postgres(input_data, **ports)

        assert hasattr(result, "converted_sql")
        assert hasattr(result, "metadata")
        assert len(event_manager.traces) > 0

    @pytest.mark.asyncio
    async def test_convert_sqlite_to_postgres_complex(self):
        """Test complex SQLite to PostgreSQL conversion."""
        event_manager = MockEventManager()
        input_data = SqlConverterInput(
            sql_query='SELECT * FROM customers WHERE name LIKE "John%";',
            table_names=["customers"],
        )

        ports = {"event_manager": event_manager}
        result = await convert_sqlite_to_postgres(input_data, **ports)

        assert hasattr(result, "converted_sql")
        # Should convert double quotes to single quotes for PostgreSQL
        assert "'" in result.converted_sql
        assert len(event_manager.traces) > 0


class TestPetSQLPipeline:
    """Test PetSQLPipeline class."""

    def test_pipeline_creation(self):
        """Test pipeline creation and basic properties."""
        pipeline = PetSQLPipeline()

        assert pipeline.name == "petsql_pipeline"
        assert pipeline.description is not None
        assert hasattr(pipeline, "builder")

    def test_pipeline_function_registration(self):
        """Test that pipeline functions are registered."""
        pipeline = PetSQLPipeline()

        # Check that functions are registered
        assert "select_examples" in pipeline.builder.registered_functions
        assert "filter_relevant_schema" in pipeline.builder.registered_functions
        assert "sql_validator" in pipeline.builder.registered_functions
        assert "convert_sqlite_to_postgres" in pipeline.builder.registered_functions

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = PetSQLPipeline()

        validation_result = pipeline.validate()
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0


class TestPetSQLIntegration:
    """Integration tests for PetSQL pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_function_flow(self):
        """Test the flow between pipeline functions."""
        event_manager = MockEventManager()

        # Test select_examples -> filter_relevant_schema flow
        examples_input = SelectExamplesInput(
            user_query="Show me all customers",
            database_schema={"customers": {"columns": ["id", "name"]}},
        )

        examples_result = await select_examples(examples_input, event_manager=event_manager)

        # Use examples result as input for filter_relevant_schema
        mock_database = MagicMock()
        mock_database.get_schema.return_value = {"customers": {"columns": ["id", "name"]}}

        filter_input = FilterSchemaInput(
            user_query=examples_input.user_query,
            examples=examples_result.examples,
        )

        filter_result = await filter_relevant_schema(
            filter_input, event_manager=event_manager, database=mock_database
        )

        assert hasattr(filter_result, "filtered_schema")
        assert len(event_manager.traces) > 1  # Should have traces from both functions
