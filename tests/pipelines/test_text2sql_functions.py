"""Tests for text2sql functions and tool router."""

from datetime import datetime
from unittest.mock import Mock

from hexai.adapters.mock import MockDatabaseAdapter
import pytest

from pipelines.text2sql.functions import Text2SQLToolRouter


class TestText2SQLToolRouterDetailed:
    """Detailed tests for Text2SQLToolRouter class."""

    def test_initialization_with_none(self):
        """Test router initialization when database_port is None."""
        router = Text2SQLToolRouter(None)
        assert router.database_port is not None
        assert isinstance(router.database_port, MockDatabaseAdapter)

    def test_initialization_with_custom_port(self):
        """Test router initialization with custom database port."""
        custom_port = Mock()
        router = Text2SQLToolRouter(custom_port)
        assert router.database_port is custom_port

    @pytest.mark.asyncio
    async def test_aroute_with_none_input(self):
        """Test aroute method with None input data."""
        router = Text2SQLToolRouter()

        # get_schema should work with None input
        result = await router.aroute("get_schema", None)
        assert "tables" in result

    @pytest.mark.asyncio
    async def test_aroute_with_empty_dict_input(self):
        """Test aroute method with empty dict input."""
        router = Text2SQLToolRouter()

        result = await router.aroute("get_schema", {})
        assert "tables" in result

    @pytest.mark.asyncio
    async def test_aroute_invalid_parameters_type_error(self):
        """Test aroute with parameters that cause TypeError."""
        router = Text2SQLToolRouter()

        # Try to call a function with wrong parameter type - should raise TypeError
        with pytest.raises(TypeError):
            await router.aroute("get_column_enum_values", {"table": 123})

    @pytest.mark.asyncio
    async def test_get_schema_comprehensive(self):
        """Comprehensive test of _get_schema method."""
        router = Text2SQLToolRouter()

        result = await router._get_schema()

        # Check overall structure
        assert isinstance(result, dict)
        assert "tables" in result
        assert isinstance(result["tables"], dict)

        # Check specific table structures
        for table_name, table_info in result["tables"].items():
            assert isinstance(table_name, str)
            assert isinstance(table_info, dict)
            assert "columns" in table_info
            assert "primary_key" in table_info
            assert "foreign_keys" in table_info
            assert "description" in table_info

            # Verify types
            assert isinstance(table_info["columns"], list)
            assert isinstance(table_info["primary_key"], str)
            assert isinstance(table_info["foreign_keys"], dict)
            assert isinstance(table_info["description"], str)

    @pytest.mark.asyncio
    async def test_get_column_enum_values_edge_cases(self):
        """Test edge cases for _get_column_enum_values method."""
        router = Text2SQLToolRouter()

        # Test with empty table/column names
        result = await router._get_column_enum_values("", "")
        assert result == []

        # Test with special characters
        result = await router._get_column_enum_values("table@#$", "column!@#")
        assert result == []

        # Test with None-like strings
        result = await router._get_column_enum_values("null", "none")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_join_path_edge_cases(self):
        """Test edge cases for _get_join_path method."""
        router = Text2SQLToolRouter()

        # Test with same table name for both parameters
        result = await router._get_join_path("customers", "customers")
        assert result["relationship"] == "none"

        # Test with empty table names
        result = await router._get_join_path("", "")
        assert result["relationship"] == "none"

        # Test with special characters
        result = await router._get_join_path("table@#$", "table!@#")
        assert result["relationship"] == "none"

    @pytest.mark.asyncio
    async def test_resolve_date_range_edge_cases(self):
        """Test edge cases for _resolve_date_range method."""
        router = Text2SQLToolRouter()

        # Test with empty string
        result = await router._resolve_date_range("")
        assert "start_date" in result
        assert "end_date" in result
        assert result["phrase_interpreted"] == ""

        # Test with whitespace only
        result = await router._resolve_date_range("   ")
        assert "start_date" in result
        assert "end_date" in result

        # Test with mixed case
        result = await router._resolve_date_range("LAST 3 MONTHS")
        assert "start_date" in result
        assert "end_date" in result

        # Test case sensitivity
        result = await router._resolve_date_range("LaSt WeEk")
        assert "start_date" in result
        assert "end_date" in result

    @pytest.mark.asyncio
    async def test_resolve_date_range_date_formats(self):
        """Test that resolve_date_range returns proper date formats."""
        router = Text2SQLToolRouter()

        result = await router._resolve_date_range("last week")

        # Verify date format (YYYY-MM-DD)
        start_date = result["start_date"]
        end_date = result["end_date"]

        # Should be valid date strings
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")

        # SQL condition should contain the start date
        assert start_date in result["sql_condition"]

    @pytest.mark.asyncio
    async def test_preview_data_edge_cases(self):
        """Test edge cases for _preview_data method."""
        router = Text2SQLToolRouter()

        # Test with limit of 0
        result = await router._preview_data("customers", 0)
        assert result["rows_shown"] == 0
        assert len(result["sample_data"]) == 0

        # Test with very large limit
        result = await router._preview_data("customers", 1000)
        assert result["rows_shown"] == 1000
        assert len(result["sample_data"]) == 1000

        # Test with negative limit (should be clamped to 0)
        result = await router._preview_data("customers", -1)
        assert result["rows_shown"] == 0

    @pytest.mark.asyncio
    async def test_preview_data_with_database_error(self):
        """Test preview_data when database port raises an error."""
        # Create a mock database port that raises an exception
        mock_port = Mock()
        mock_port.get_table_schemas.side_effect = Exception("Database error")

        router = Text2SQLToolRouter(mock_port)

        # Should handle the exception gracefully
        with pytest.raises(Exception, match="Database error"):
            await router._preview_data("customers", 5)

    def test_generate_sample_data_all_table_types(self):
        """Test _generate_sample_data for all supported table types."""
        router = Text2SQLToolRouter()

        test_cases = [
            ("customers", {"columns": {"id": "int", "name": "str"}}),
            ("orders", {"columns": {"id": "int", "customer_id": "int"}}),
            ("products", {"columns": {"id": "int", "name": "str"}}),
            ("order_items", {"columns": {"id": "int", "order_id": "int"}}),
        ]

        for table_name, schema in test_cases:
            sample_data = router._generate_sample_data(table_name, schema, 3)
            assert len(sample_data) == 3
            assert isinstance(sample_data, list)
            for row in sample_data:
                assert isinstance(row, dict)

    def test_generate_sample_data_consistency(self):
        """Test that _generate_sample_data returns consistent data."""
        router = Text2SQLToolRouter()
        schema = {"columns": {"id": "int", "name": "str"}}

        # Generate data multiple times
        data1 = router._generate_sample_data("customers", schema, 5)
        data2 = router._generate_sample_data("customers", schema, 5)

        # Should be consistent (same structure, same content)
        assert len(data1) == len(data2)
        for i in range(len(data1)):
            assert data1[i] == data2[i]

    def test_generate_sample_data_different_limits(self):
        """Test _generate_sample_data with different limit values."""
        router = Text2SQLToolRouter()
        schema = {"columns": {"id": "int", "name": "str"}}

        limits = [0, 1, 5, 10, 100]
        for limit in limits:
            sample_data = router._generate_sample_data("generic", schema, limit)
            assert len(sample_data) == limit

    @pytest.mark.asyncio
    async def test_all_tools_return_proper_types(self):
        """Test that all tools return proper data types."""
        router = Text2SQLToolRouter()

        # Test get_schema returns dict
        result = await router.aroute("get_schema")
        assert isinstance(result, dict)

        # Test get_column_enum_values returns list
        result = await router.aroute(
            "get_column_enum_values", {"table": "orders", "column": "status"}
        )
        assert isinstance(result, list)

        # Test get_join_path returns dict
        result = await router.aroute("get_join_path", {"table1": "orders", "table2": "customers"})
        assert isinstance(result, dict)

        # Test resolve_date_range returns dict
        result = await router.aroute("resolve_date_range", {"phrase": "last week"})
        assert isinstance(result, dict)

        # Test preview_data returns dict
        result = await router.aroute("preview_data", {"table": "customers", "limit": 3})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_tool_execution_with_mock_database_methods(self):
        """Test tool execution with mocked database methods."""
        mock_port = Mock()

        # Mock the database methods
        mock_port.get_table_schemas.return_value = {
            "test_table": {
                "columns": {"id": "int", "name": "str"},
                "primary_keys": ["id"],
                "foreign_keys": [],
            }
        }
        mock_port.get_table_statistics.return_value = {"test_table": {"row_count": 42}}
        mock_port.get_relationships.return_value = [
            {
                "from_table": "orders",
                "to_table": "customers",
                "from_column": "customer_id",
                "to_column": "id",
                "relationship_type": "many_to_one",
            }
        ]

        router = Text2SQLToolRouter(mock_port)

        # Test get_schema
        result = await router.aroute("get_schema")
        assert "test_table" in result["tables"]

        # Test preview_data
        result = await router.aroute("preview_data", {"table": "test_table", "limit": 5})
        assert result["total_available"] == 42

        # Test get_join_path
        result = await router.aroute("get_join_path", {"table1": "orders", "table2": "customers"})
        assert result["relationship"] == "many_to_one"

    def test_tools_dictionary_immutability(self):
        """Test that the tools dictionary is properly configured."""
        router = Text2SQLToolRouter()

        # Should not be able to modify the tools dictionary from outside
        original_tools = router.tools.copy()

        # Try to modify (this shouldn't affect the router)
        async def malicious_tool():
            return "hacked"

        router.tools["malicious_tool"] = malicious_tool

        # Create a new router and verify it has the original tools
        new_router = Text2SQLToolRouter()
        assert "malicious_tool" not in new_router.tools
        assert len(new_router.tools) == len(original_tools)
