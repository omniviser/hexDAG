"""Clean tests for the FunctionNode implementation."""

from unittest.mock import AsyncMock

from hexai.core.application.nodes.function_node import FunctionNode
from hexai.core.domain.dag import NodeSpec
from pydantic import BaseModel
import pytest


class _UserInput(BaseModel):
    name: str
    age: int = 25


class _UserOutput(BaseModel):
    greeting: str
    user_age: int


class TestFunctionNode:
    """Test the FunctionNode factory."""

    @pytest.fixture
    def factory(self):
        """Create function node factory."""
        return FunctionNode()

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports."""
        return {"event_manager": AsyncMock(), "database": AsyncMock(), "cache": AsyncMock()}

    def test_create_node_basic(self, factory):
        """Test creating a basic function node."""

        def add_one(x: int) -> int:
            return x + 1

        node = factory("add_one", add_one)

        assert isinstance(node, NodeSpec)
        assert node.name == "add_one"
        assert node.in_type is None
        assert node.out_type is None
        assert node.deps == frozenset()

    def test_create_node_with_schemas(self, factory):
        """Test creating a node with input/output schemas."""

        def process_user(user_data: dict) -> dict:
            return {"greeting": f"Hello, {user_data['name']}!"}

        node = factory(
            "process_user", process_user, input_schema=_UserInput, output_schema=_UserOutput
        )

        assert node.name == "process_user"
        assert node.in_type is not None
        assert node.out_type is not None

    def test_create_node_with_dependencies(self, factory):
        """Test creating a node with dependencies."""

        def dependent_func(x: int) -> int:
            return x * 2

        node = factory("dependent", dependent_func, deps=["input_node", "config_node"])

        assert node.deps == frozenset(["input_node", "config_node"])

    def test_create_node_with_params(self, factory):
        """Test creating a node with extra parameters."""

        def configurable_func(x: int) -> int:
            return x + 1

        node = factory("configurable", configurable_func, multiplier=2, debug=True)

        assert node.params["multiplier"] == 2
        assert node.params["debug"] is True

    @pytest.mark.asyncio
    async def test_sync_function_execution(self, factory):
        """Test executing a sync function."""

        def multiply_by_two(x: int) -> int:
            return x * 2

        node = factory("multiply", multiply_by_two)
        result = await node.fn(5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_async_function_execution(self, factory):
        """Test executing an async function."""

        async def async_add_one(x: int) -> int:
            return x + 1

        node = factory("async_add", async_add_one)
        result = await node.fn(5)

        assert result == 6

    @pytest.mark.asyncio
    async def test_function_with_kwargs_gets_all_ports(self, factory, mock_ports):
        """Test that functions with **kwargs receive all ports."""

        def func_with_kwargs(data: str, **kwargs) -> dict:
            return {
                "data": data,
                "port_count": len(kwargs),
                "has_event_manager": "event_manager" in kwargs,
                "has_database": "database" in kwargs,
            }

        node = factory("kwargs_func", func_with_kwargs)
        result = await node.fn("test", **mock_ports)

        assert result["data"] == "test"
        assert result["port_count"] == 3  # event_manager, database, cache
        assert result["has_event_manager"] is True
        assert result["has_database"] is True

    @pytest.mark.asyncio
    async def test_function_with_specific_params_gets_matching_ports(self, factory, mock_ports):
        """Test that functions with specific parameters get only matching ports."""

        def func_with_params(data: str, event_manager=None, unknown_port=None) -> dict:
            return {
                "data": data,
                "has_event_manager": event_manager is not None,
                "has_unknown_port": unknown_port is not None,
            }

        node = factory("params_func", func_with_params)

        # Add an unknown port that should be filtered out
        all_ports = {**mock_ports, "unknown_port": AsyncMock(), "extra_port": AsyncMock()}
        result = await node.fn("test", **all_ports)

        assert result["data"] == "test"
        assert result["has_event_manager"] is True
        assert result["has_unknown_port"] is True  # This matches a parameter
        # extra_port should be filtered out (not accessible to function)

    @pytest.mark.asyncio
    async def test_function_without_ports_gets_no_ports(self, factory, mock_ports):
        """Test that functions without port parameters get no ports."""

        def func_no_ports(data: str) -> str:
            return data.upper()

        node = factory("no_ports", func_no_ports)
        result = await node.fn("test", **mock_ports)

        assert result == "TEST"

    @pytest.mark.asyncio
    async def test_input_validation_success(self, factory):
        """Test function execution with dict input (validation handled by orchestrator)."""

        def greet_user(user: dict) -> dict:
            return {"message": f"Hello, {user['name']}!"}

        node = factory("greet", greet_user, input_schema=_UserInput)

        result = await node.fn({"name": "Alice", "age": 30})
        assert result["message"] == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_input_validation_failure(self, factory):
        """Test function execution with missing data (orchestrator should handle validation)."""

        def process_user(user: dict) -> dict:
            # Function will fail if 'name' key is missing - this is expected
            return {"name": user["name"]}

        node = factory("process", process_user, input_schema=_UserInput)

        with pytest.raises(KeyError):  # Function fails when accessing missing key
            await node.fn({"age": 30})  # Missing required 'name'

    @pytest.mark.asyncio
    async def test_output_validation_success(self, factory):
        """Test node returns raw output."""

        def create_user_output(data: dict) -> dict:
            return {"greeting": "Hello!", "user_age": 25}

        node = factory("create_output", create_user_output, output_schema=_UserOutput)
        result = await node.fn({"input": "test"})

        # Node returns raw dict - orchestrator handles validation
        assert isinstance(result, dict)
        assert result["greeting"] == "Hello!"
        assert result["user_age"] == 25

    @pytest.mark.asyncio
    async def test_output_validation_failure(self, factory):
        """Test output validation graceful fallback."""

        def create_invalid_output(data: dict) -> dict:
            return {"user_age": 25}  # Missing required 'greeting'

        node = factory("invalid_output", create_invalid_output, output_schema=_UserOutput)

        # Should return raw result when validation fails (no exception raised)
        result = await node.fn({"input": "test"})
        assert result == {"user_age": 25}  # Raw result returned

    @pytest.mark.asyncio
    async def test_full_validation_workflow(self, factory):
        """Test complete workflow with dict input and raw output."""

        def process_user_data(user: dict) -> dict:
            return {"greeting": f"Welcome, {user['name']}!", "user_age": user["age"]}

        node = factory(
            "process_user", process_user_data, input_schema=_UserInput, output_schema=_UserOutput
        )

        result = await node.fn({"name": "Bob", "age": 35})

        # Node returns raw dict - orchestrator handles validation
        assert isinstance(result, dict)
        assert result["greeting"] == "Welcome, Bob!"
        assert result["user_age"] == 35

    def test_function_metadata_preservation(self, factory):
        """Test that function metadata is preserved."""

        def documented_function(x: int) -> int:
            """This is a well-documented function."""
            return x + 1

        node = factory("documented", documented_function)

        assert "documented" in node.fn.__name__
        assert node.fn.__doc__ is not None

    @pytest.mark.asyncio
    async def test_port_combinations(self, factory):
        """Test various combinations of port handling."""

        # Function that uses specific ports
        def func_with_event_manager(data: str, event_manager=None) -> str:
            return f"{data}_with_em" if event_manager else f"{data}_no_em"

        node = factory("em_func", func_with_event_manager)

        # Test with event_manager
        result1 = await node.fn("test", event_manager=AsyncMock())
        assert result1 == "test_with_em"

        # Test without event_manager
        result2 = await node.fn("test")
        assert result2 == "test_no_em"

        # Test with extra ports (should be filtered out)
        result3 = await node.fn("test", event_manager=AsyncMock(), extra_port=AsyncMock())
        assert result3 == "test_with_em"

    @pytest.mark.asyncio
    async def test_complex_port_scenarios(self, factory):
        """Test complex port handling scenarios."""

        # Function with multiple specific ports
        def multi_port_func(data: str, event_manager=None, database=None, cache=None) -> dict:
            return {
                "data": data,
                "event_manager": event_manager is not None,
                "database": database is not None,
                "cache": cache is not None,
            }

        node = factory("multi_port", multi_port_func)

        # Test with subset of ports
        result = await node.fn(
            "test", event_manager=AsyncMock(), cache=AsyncMock(), unknown_service=AsyncMock()
        )  # Should be filtered out

        assert result["event_manager"] is True
        assert result["database"] is False  # Not provided
        assert result["cache"] is True

    @pytest.mark.asyncio
    async def test_edge_cases(self, factory):
        """Test edge cases and unusual scenarios."""

        # Function with no parameters at all
        def no_params() -> str:
            return "no_params_result"

        # This should work but not be practical in real use
        with pytest.raises(Exception):
            node = factory("no_params", no_params)
            # This would fail because wrapped_fn expects input_data
            # But we test that the node creation works

        # Function with only **kwargs
        def only_kwargs(**kwargs) -> dict:
            return {"kwargs_count": len(kwargs)}

        # This would also be unusual but test node creation
        with pytest.raises(Exception):
            factory("only_kwargs", only_kwargs)
            # Expected - functions need input_data parameter

    def test_input_mapping_basic(self, factory):
        """Test basic input mapping functionality."""

        def process_data(content: str, language: str, status: str) -> str:
            return f"{content} ({language}) - {status}"

        node = factory(
            "processor",
            process_data,
            deps=["source1", "source2"],
            input_mapping={
                "content": "source1.text",
                "language": "source1.metadata.lang",
                "status": "source2.status",
            },
        )

        assert "input_mapping" in node.params
        assert node.params["input_mapping"]["content"] == "source1.text"
        assert node.params["input_mapping"]["language"] == "source1.metadata.lang"
        assert node.params["input_mapping"]["status"] == "source2.status"

    def test_input_mapping_nested_paths(self, factory):
        """Test input mapping with nested field paths."""

        def process_nested(data: dict) -> str:
            return f"{data['user']['name']} - {data['score']}"

        node = factory(
            "nested_processor",
            process_nested,
            deps=["user_data", "validation"],
            input_mapping={
                "user": "user_data.profile",
                "score": "validation.results.score",
            },
        )

        assert node.params["input_mapping"]["user"] == "user_data.profile"
        assert node.params["input_mapping"]["score"] == "validation.results.score"

    def test_input_mapping_with_schemas(self, factory):
        """Test input mapping combined with input/output schemas."""

        def process_user(user_data: _UserInput) -> _UserOutput:
            return _UserOutput(greeting=f"Hello, {user_data.name}!", user_age=user_data.age)

        node = factory(
            "user_processor",
            process_user,
            input_schema=_UserInput,
            output_schema=_UserOutput,
            deps=["user_source", "age_source"],
            input_mapping={
                "name": "user_source.name",
                "age": "age_source.age",
            },
        )

        assert node.in_type is not None
        assert node.out_type is not None
        assert "input_mapping" in node.params
        assert node.params["input_mapping"]["name"] == "user_source.name"
        assert node.params["input_mapping"]["age"] == "age_source.age"

    def test_input_mapping_empty(self, factory):
        """Test input mapping with empty mapping dictionary."""

        def simple_func(data: str) -> str:
            return data.upper()

        node = factory("simple", simple_func, input_mapping={})

        assert "input_mapping" in node.params
        assert node.params["input_mapping"] == {}

    def test_input_mapping_none(self, factory):
        """Test input mapping with None (no mapping)."""

        def simple_func(data: str) -> str:
            return data.upper()

        node = factory("simple", simple_func, input_mapping=None)

        # Should not have input_mapping in params when None
        assert "input_mapping" not in node.params

    def test_convenience_methods(self, factory):
        """Test convenience methods for creating input mappings."""

        # Test passthrough mapping
        passthrough = FunctionNode.create_passthrough_mapping(["text", "status", "score"])
        expected_passthrough = {"text": "text", "status": "status", "score": "score"}
        assert passthrough == expected_passthrough

        # Test rename mapping
        rename = FunctionNode.create_rename_mapping({"content": "text", "validation": "status"})
        expected_rename = {"content": "text", "validation": "status"}
        assert rename == expected_rename

        # Test prefixed mapping
        prefixed = FunctionNode.create_prefixed_mapping(["text", "score"], "processor", "proc_")
        expected_prefixed = {"proc_text": "processor.text", "proc_score": "processor.score"}
        assert prefixed == expected_prefixed

    def test_with_input_mapping_enhancement(self, factory):
        """Test enhancing existing nodes with input mapping."""

        # Create basic node
        def basic_func(data: str) -> str:
            return data.upper()

        basic_node = factory(
            "basic",
            basic_func,
            input_schema=str,
            output_schema=str,
        )

        # Enhance with input mapping
        enhanced_node = factory.with_input_mapping(basic_node, {"data": "source.text"})

        # Verify input mapping was added
        assert "input_mapping" in enhanced_node.params
        assert enhanced_node.params["input_mapping"]["data"] == "source.text"

        # Verify other properties are preserved
        assert enhanced_node.name == "basic"
        assert enhanced_node.fn == basic_node.fn
        assert enhanced_node.in_type == basic_node.in_type
        assert enhanced_node.out_type == basic_node.out_type

    def test_with_input_mapping_overwrite(self, factory):
        """Test that with_input_mapping overwrites existing mapping."""

        # Create node with existing mapping
        def func(data: str) -> str:
            return data.upper()

        original_node = factory("test", func, input_mapping={"data": "old.source"})

        # Overwrite with new mapping
        updated_node = factory.with_input_mapping(
            original_node, {"data": "new.source", "extra": "other.field"}
        )

        # Verify new mapping
        assert updated_node.params["input_mapping"]["data"] == "new.source"
        assert updated_node.params["input_mapping"]["extra"] == "other.field"

    def test_input_mapping_with_ports(self, factory):
        """Test input mapping with functions that use ports."""

        def port_func(content: str, event_manager=None) -> str:
            return f"{content}_processed"

        node = factory(
            "port_processor", port_func, deps=["source"], input_mapping={"content": "source.text"}
        )

        # Verify both input mapping and port handling work together
        assert "input_mapping" in node.params
        assert node.params["input_mapping"]["content"] == "source.text"

    def test_input_mapping_complex_scenario(self, factory):
        """Test complex input mapping scenario with multiple dependencies."""

        def complex_processor(
            user_name: str, user_age: int, validation_status: str, score: float, metadata: dict
        ) -> dict:
            return {
                "result": f"{user_name} ({user_age}) - {validation_status}",
                "score": score,
                "metadata": metadata,
            }

        node = factory(
            "complex_processor",
            complex_processor,
            deps=["user_data", "validation", "scoring", "config"],
            input_mapping={
                "user_name": "user_data.name",
                "user_age": "user_data.age",
                "validation_status": "validation.status",
                "score": "scoring.score",
                "metadata": "config.metadata",
            },
        )

        # Verify all mappings are correct
        expected_mappings = {
            "user_name": "user_data.name",
            "user_age": "user_data.age",
            "validation_status": "validation.status",
            "score": "scoring.score",
            "metadata": "config.metadata",
        }

        assert node.params["input_mapping"] == expected_mappings
        assert node.deps == frozenset(["user_data", "validation", "scoring", "config"])
