"""Tests for the text2sql pipeline."""

from unittest.mock import Mock

from hexai.adapters.mock import MockDatabaseAdapter, MockLLM
import pytest

from pipelines.text2sql import Text2SQLPipeline
from pipelines.text2sql.functions import Text2SQLToolRouter


class TestText2SQLPipeline:
    """Tests for Text2SQLPipeline."""

    def test_pipeline_attributes(self):
        """Test pipeline attributes."""
        pipeline = Text2SQLPipeline()

        assert pipeline.name == "text2sql_pipeline"
        assert (
            pipeline.description
            == "Intelligent Text-to-SQL generation with agent-based reasoning and tool usage"
        )

    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = Text2SQLPipeline()

        # Should have valid YAML path and config
        assert pipeline._yaml_path is not None
        assert pipeline._config is not None

        # Validate pipeline
        validation = pipeline.validate()
        assert validation["valid"] is True
        assert validation["errors"] == []

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        pipeline = Text2SQLPipeline()
        config = pipeline.get_config()

        assert config is not None
        assert "nodes" in config
        assert len(config["nodes"]) == 1  # Single agent node

        # Check that the agent node is present
        agent_node = config["nodes"][0]
        assert agent_node["type"] == "agent"
        assert agent_node["id"] == "text2sql_agent"
        assert "params" in agent_node

        # Check key configuration parameters
        params = agent_node["params"]
        assert "initial_prompt_template" in params
        assert "continuation_prompt_template" in params
        assert "max_steps" in params
        assert "input_schema" in params
        assert "output_schema" in params

        # Verify input/output schemas
        assert "question" in params["input_schema"]
        assert "database_name" in params["input_schema"]
        assert "sql_query" in params["output_schema"]
        assert "explanation" in params["output_schema"]
        assert "confidence" in params["output_schema"]


class TestText2SQLToolRouter:
    """Tests for Text2SQLToolRouter."""

    def test_tool_router_initialization(self):
        """Test tool router initialization."""
        # Test with default MockDatabaseAdapter
        router = Text2SQLToolRouter()
        assert router.database_port is not None
        assert isinstance(router.database_port, MockDatabaseAdapter)

        # Test with custom database port
        custom_port = MockDatabaseAdapter()
        router = Text2SQLToolRouter(custom_port)
        assert router.database_port is custom_port

    def test_tool_registration(self):
        """Test that all tools are properly registered."""
        router = Text2SQLToolRouter()

        expected_tools = [
            "get_schema",
            "get_column_enum_values",
            "get_join_path",
            "resolve_date_range",
            "preview_data",
        ]
        for tool in expected_tools:
            assert tool in router.tools

    @pytest.mark.asyncio
    async def test_get_schema_tool(self):
        """Test the get_schema tool."""
        router = Text2SQLToolRouter()

        result = await router.aroute("get_schema")

        assert "tables" in result
        assert isinstance(result["tables"], dict)
        # MockDatabaseAdapter has 4 tables
        assert len(result["tables"]) == 4
        assert "customers" in result["tables"]
        assert "orders" in result["tables"]

    @pytest.mark.asyncio
    async def test_get_schema_tool_detailed_structure(self):
        """Test the detailed structure of get_schema tool response."""
        router = Text2SQLToolRouter()

        result = await router.aroute("get_schema")

        # Verify each table has required fields
        for _table_name, table_info in result["tables"].items():
            assert "columns" in table_info
            assert "primary_key" in table_info
            assert "foreign_keys" in table_info
            assert "description" in table_info
            assert isinstance(table_info["columns"], list)
            assert isinstance(table_info["foreign_keys"], dict)

    @pytest.mark.asyncio
    async def test_get_column_enum_values_tool(self):
        """Test the get_column_enum_values tool."""
        router = Text2SQLToolRouter()

        # Test with known enum values
        result = await router.aroute(
            "get_column_enum_values", {"table": "orders", "column": "status"}
        )
        assert isinstance(result, list)
        assert "pending" in result
        assert "cancelled" in result

        # Test with unknown column
        result = await router.aroute(
            "get_column_enum_values", {"table": "unknown", "column": "unknown"}
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_column_enum_values_all_known_combinations(self):
        """Test all known enum value combinations."""
        router = Text2SQLToolRouter()

        test_cases = [
            (
                "orders",
                "status",
                ["pending", "processing", "shipped", "delivered", "cancelled", "returned"],
            ),
            ("customers", "status", ["active", "inactive", "suspended"]),
            ("customers", "segment", ["enterprise", "small_business", "individual"]),
            ("products", "category", ["electronics", "clothing", "books", "home", "sports"]),
        ]

        for table, column, expected_values in test_cases:
            result = await router.aroute(
                "get_column_enum_values", {"table": table, "column": column}
            )
            assert result == expected_values

    @pytest.mark.asyncio
    async def test_get_join_path_tool(self):
        """Test the get_join_path tool."""
        router = Text2SQLToolRouter()

        # Test with tables that have a relationship
        result = await router.aroute("get_join_path", {"table1": "orders", "table2": "customers"})
        assert "join_condition" in result
        assert "relationship" in result
        # Should find the relationship from MockDatabaseAdapter
        assert (
            "orders.customer_id = customers.id" in result["join_condition"]
            or "customers.id = orders.customer_id" in result["join_condition"]
        )

    @pytest.mark.asyncio
    async def test_get_join_path_no_relationship(self):
        """Test get_join_path with tables that have no relationship."""
        router = Text2SQLToolRouter()

        result = await router.aroute("get_join_path", {"table1": "table1", "table2": "table2"})
        assert result["relationship"] == "none"
        assert "No direct relationship found" in result["join_condition"]
        assert "No known relationship" in result["description"]

    @pytest.mark.asyncio
    async def test_resolve_date_range_tool(self):
        """Test the resolve_date_range tool."""
        router = Text2SQLToolRouter()

        result = await router.aroute("resolve_date_range", {"phrase": "last 3 months"})
        assert "start_date" in result
        assert "end_date" in result
        assert "phrase_interpreted" in result
        assert result["phrase_interpreted"] == "last 3 months"

    @pytest.mark.asyncio
    async def test_resolve_date_range_all_phrases(self):
        """Test resolve_date_range with all supported phrases."""
        router = Text2SQLToolRouter()

        test_phrases = [
            "last 3 months",
            "past 3 months",
            "last month",
            "past month",
            "last week",
            "past week",
            "last year",
            "past year",
            "today",
            "yesterday",
            "unknown phrase",  # Should default to last 30 days
        ]

        for phrase in test_phrases:
            result = await router.aroute("resolve_date_range", {"phrase": phrase})
            assert "start_date" in result
            assert "end_date" in result
            assert "phrase_interpreted" in result
            assert "sql_condition" in result
            assert result["phrase_interpreted"] == phrase

    @pytest.mark.asyncio
    async def test_preview_data_tool(self):
        """Test the preview_data tool."""
        router = Text2SQLToolRouter()

        # Test with valid table
        result = await router.aroute("preview_data", {"table": "customers", "limit": 3})
        assert "table" in result
        assert "sample_data" in result
        assert "rows_shown" in result
        assert result["table"] == "customers"
        assert len(result["sample_data"]) == 3

        # Test with invalid table
        result = await router.aroute("preview_data", {"table": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_preview_data_all_tables(self):
        """Test preview_data with all known table types."""
        router = Text2SQLToolRouter()

        tables_to_test = ["customers", "orders", "products", "order_items", "unknown_table"]

        for table in tables_to_test:
            result = await router.aroute("preview_data", {"table": table, "limit": 2})

            if table == "unknown_table":
                # Should return error for unknown table
                assert "error" in result
                assert "available_tables" in result
            else:
                # Should return sample data
                assert "table" in result
                assert "sample_data" in result
                assert "rows_shown" in result
                assert "total_available" in result
                assert "columns" in result
                assert len(result["sample_data"]) == 2

    @pytest.mark.asyncio
    async def test_preview_data_different_limits(self):
        """Test preview_data with different limit values."""
        router = Text2SQLToolRouter()

        limits = [1, 3, 5, 10]
        for limit in limits:
            result = await router.aroute("preview_data", {"table": "customers", "limit": limit})
            assert len(result["sample_data"]) == limit

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test behavior when tool is not found."""
        router = Text2SQLToolRouter()

        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not found"):
            await router.aroute("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_tool_parameter_error(self):
        """Test behavior when tool parameters are invalid."""
        router = Text2SQLToolRouter()

        # Try to call get_column_enum_values without required parameters
        with pytest.raises(TypeError):
            await router.aroute("get_column_enum_values", {"table": "orders"})  # Missing 'column'

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test behavior when tool execution encounters an error."""
        # Create a router with a mocked database port that raises an exception
        mock_port = Mock()
        mock_port.get_table_schemas.side_effect = Exception("Database connection error")

        router = Text2SQLToolRouter(mock_port)

        with pytest.raises(Exception, match="Database connection error"):
            await router.aroute("get_schema")

    @pytest.mark.asyncio
    async def test_generate_sample_data_generic_table(self):
        """Test _generate_sample_data with a generic table."""
        router = Text2SQLToolRouter()

        # Test with a table that doesn't have specific mock data
        generic_schema = {"columns": {"id": "int", "name": "str", "value": "float"}}

        sample_data = router._generate_sample_data("generic_table", generic_schema, 3)
        assert len(sample_data) == 3

        for row in sample_data:
            assert "id" in row
            assert "name" in row
            assert "value" in row

    def test_tool_router_tools_property(self):
        """Test that tool router properly exposes all tools."""
        router = Text2SQLToolRouter()

        # Verify tools dictionary is properly populated
        expected_tools = [
            "get_schema",
            "get_column_enum_values",
            "get_join_path",
            "resolve_date_range",
            "preview_data",
        ]
        assert all(tool in router.tools for tool in expected_tools)

        # Verify all tools are callable
        for _tool_name, tool_func in router.tools.items():
            assert callable(tool_func)


def get_test_questions():
    """Get test questions for pipeline testing."""
    return [
        {
            "question": "Show me customers who canceled orders in the last 3 months",
            "description": "Simple join with date filtering and status condition",
        },
        {
            "question": "Find all active customers who have placed orders recently",
            "description": "Customer status filtering with recent order activity",
        },
        {
            "question": "What are the top 5 most expensive products in electronics category?",
            "description": "Category filtering with ordering and limiting",
        },
    ]


def analyze_pipeline_result(result):
    """Analyze pipeline execution results."""
    if result["status"] != "success":
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
            "trace_count": len(result.get("trace", [])),
        }

    # Extract agent results
    agent_result = result["results"]["text2sql_agent"]
    trace = result.get("trace", [])

    # Count tool usage - for composite agent, look for tool execution traces
    tools_used = []

    # Look for tool calls in trace (old agent format)
    for step in trace:
        if "Tool '" in step and "completed" in step:
            tool_name = step.split("Tool '")[1].split("'")[0]
            tools_used.append(tool_name)

    # If no tools found in old format, look for composite agent format
    # Look for tool_executor nodes in the trace
    if not tools_used:
        for step in trace:
            if "continuation_tool_executor" in step and "completed" in step:
                # Mock agent just calls get_schema for now, so we know at least one tool was used
                tools_used.append("get_schema")

    # Handle both dict and object results
    if isinstance(agent_result, dict):
        thought = agent_result.get("thought", "")
        confidence = agent_result.get("confidence", "unknown")
    else:
        thought = getattr(agent_result, "thought", "")
        confidence = getattr(agent_result, "confidence", "unknown")

    return {
        "success": True,
        "final_thought": thought[:100] + "..." if len(thought) > 100 else thought,
        "confidence": confidence,
        "tools_used": tools_used,
        "tool_count": len(tools_used),
        "step_count": len(
            [step for step in trace if "Step" in step and "Starting reasoning" in step]
        ),
        "trace_count": len(trace),
    }


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_mock_llm(self):
        """Test full pipeline execution with MockLLM."""
        pipeline = Text2SQLPipeline()

        # Set up mock responses for agent reasoning (max 8 steps)
        mock_responses = [
            "I need to understand the database structure first.",
            "INVOKE_TOOL: get_schema()",
            "Great! I can see the available tables. Let me check the status values.",
            'INVOKE_TOOL: get_column_enum_values(table="orders", column="status")',
            "Perfect! Now let me resolve the date range.",
            'INVOKE_TOOL: resolve_date_range(phrase="last 3 months")',
            """Based on my analysis, I can generate the SQL query.
            INVOKE_TOOL: tool_end(sql_query="SELECT * FROM customers WHERE status = 'cancelled'",
            explanation="Query finds customers with cancelled orders", confidence="high")""",
        ]

        # Set up ports
        ports = {
            "llm": MockLLM(mock_responses),
            "tool_router": Text2SQLToolRouter(),
        }

        # Execute pipeline
        input_data = {
            "question": "Show me customers who canceled orders in the last 3 months",
            "database_name": "ecommerce_db",
        }

        result = await pipeline.execute(input_data, ports)

        # Verify execution
        if result["status"] != "success":
            error_msg = result.get("error", "Unknown error")
            results = result.get("results", {})
            pytest.fail(f"Pipeline failed with error: {error_msg}. Results: {results}")
        assert result["status"] == "success"
        assert "text2sql_agent" in result["results"]

        # Analyze results
        # Test without expecting high quality output - the mock agent only runs one step
        analysis = analyze_pipeline_result(result)
        assert analysis["success"]

        # Verify the agent returned something
        agent_result = result["results"]["text2sql_agent"]
        # Agent now returns custom output schema based on
        # pipeline.yaml output_schema
        assert hasattr(agent_result, "sql_query")
        assert hasattr(agent_result, "explanation")
        assert hasattr(agent_result, "confidence")

        # Verify the generated SQL
        assert "SELECT" in agent_result.sql_query
        assert "customers" in agent_result.sql_query
        assert agent_result.confidence in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_multiple_questions(self):
        """Test pipeline with multiple different questions."""
        pipeline = Text2SQLPipeline()
        questions = get_test_questions()

        for question_data in questions[:2]:  # Test first 2 questions
            mock_responses = [
                "Let me analyze this question step by step.",
                "INVOKE_TOOL: get_schema {}",
                "I can see the table structure. Let me get column information.",
                'INVOKE_TOOL: get_column_enum_values {"table": "customers", "column": "status"}',
                """I can now generate the appropriate SQL:

{
  "sql_query": "SELECT * FROM customers WHERE status = 'active'",
  "explanation": "Query based on the question requirements",
  "confidence": "medium"
}

INVOKE_TOOL: tool_end(sql_query="SELECT * FROM customers WHERE status = 'active'",
explanation="Query based on the question requirements", confidence="medium")""",
            ]

            ports = {
                "llm": MockLLM(mock_responses),
                "tool_router": Text2SQLToolRouter(),
            }

            input_data = {
                "question": question_data["question"],
                "database_name": "test_db",
            }

            result = await pipeline.execute(input_data, ports)
            assert result["status"] == "success"

            analysis = analyze_pipeline_result(result)
            assert analysis["success"] is True

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_tool_errors(self):
        """Test pipeline execution when tools encounter errors."""
        pipeline = Text2SQLPipeline()

        # Mock responses that try to use non-existent tool
        mock_responses = [
            "I'll use a tool to help.",
            "INVOKE_TOOL: nonexistent_tool()",
            """The tool had an error, but I'll continue anyway.
            INVOKE_TOOL: tool_end(sql_query="SELECT * FROM customers",
            explanation="Simple query despite tool error", confidence="low")""",
        ]

        ports = {
            "llm": MockLLM(mock_responses),
            "tool_router": Text2SQLToolRouter(),
        }

        input_data = {
            "question": "Show me all customers",
            "database_name": "test_db",
        }

        result = await pipeline.execute(input_data, ports)

        # Pipeline should still succeed even with tool errors
        assert result["status"] == "success"
        analysis = analyze_pipeline_result(result)
        assert analysis["success"] is True

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_database_port(self):
        """Test pipeline execution with a custom database port."""
        pipeline = Text2SQLPipeline()

        # Create a custom mock database port
        custom_db_port = Mock()
        custom_db_port.get_table_schemas.return_value = {
            "custom_table": {
                "columns": {"id": "int", "name": "str"},
                "primary_keys": ["id"],
                "foreign_keys": [],
            }
        }
        custom_db_port.get_table_statistics.return_value = {"custom_table": {"row_count": 100}}
        custom_db_port.get_relationships.return_value = []

        mock_responses = [
            """INVOKE_TOOL: get_schema()
I can see the custom table structure.
INVOKE_TOOL: tool_end(sql_query="SELECT * FROM custom_table",
explanation="Query using custom database schema", confidence="high")""",
        ]

        ports = {
            "llm": MockLLM(mock_responses),
            "tool_router": Text2SQLToolRouter(custom_db_port),
        }

        input_data = {
            "question": "Show me data from custom table",
            "database_name": "custom_db",
        }

        result = await pipeline.execute(input_data, ports)
        assert result["status"] == "success"
        analysis = analyze_pipeline_result(result)
        assert analysis["success"] is True
