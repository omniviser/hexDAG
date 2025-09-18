"""Comprehensive tests for UnifiedToolRouter."""

import asyncio
from typing import Any

import pytest

from hexai.adapters.unified_tool_router import UnifiedToolRouter
from hexai.core.application.nodes.tool_utils import ToolDefinition
from hexai.core.registry.decorators import tool
from hexai.core.registry.registry import ComponentRegistry


class TestUnifiedToolRouter:
    """Test UnifiedToolRouter functionality."""

    @pytest.fixture
    def router(self):
        """Create a UnifiedToolRouter instance."""
        return UnifiedToolRouter()

    @pytest.fixture
    def mock_registry(self):
        """Create a mock ComponentRegistry."""
        registry = ComponentRegistry()
        # Bootstrap in dev mode to allow registration

        registry.bootstrap(
            manifest=[],  # Empty manifest for testing
            dev_mode=True,
        )
        return registry

    def test_initialization(self):
        """Test router initialization."""
        router = UnifiedToolRouter()
        assert router.component_registry is None
        assert router.port_registry is None
        assert len(router.tools) > 0  # Built-in tools should be registered

    def test_initialization_with_registry(self, mock_registry):
        """Test router initialization with ComponentRegistry."""
        router = UnifiedToolRouter(component_registry=mock_registry)
        assert router.component_registry is mock_registry
        assert router.port_registry is None

    def test_builtin_tools_registered(self, router):
        """Test that built-in tools are registered."""
        tools = router.get_available_tools()
        assert "tool_end" in tools
        assert "end" in tools  # Alias
        assert "change_phase" in tools
        assert "phase" in tools  # Alias

    def test_direct_function_registration(self, router):
        """Test direct function registration."""

        def add_numbers(x: int, y: int) -> int:
            """Add two numbers together."""
            return x + y

        router.register_function(add_numbers)

        assert "add_numbers" in router.get_available_tools()
        tool_def = router.tool_definitions["add_numbers"]
        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "add_numbers"
        assert len(tool_def.parameters) == 2

    def test_direct_function_registration_with_name(self, router):
        """Test direct function registration with custom name."""

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        router.register_function(multiply, name="mult")

        assert "mult" in router.get_available_tools()
        assert "multiply" not in router.get_available_tools()

    def test_decorator_registration(self, router):
        """Test tool decorator registration."""

        @router.tool
        def divide(x: float, y: float) -> float:
            """Divide x by y."""
            if y == 0:
                raise ValueError("Cannot divide by zero")
            return x / y

        assert "divide" in router.get_available_tools()

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, router):
        """Test synchronous tool execution."""

        def square(x: int) -> int:
            """Square a number."""
            return x * x

        router.register_function(square)
        result = await router.acall_tool("square", {"x": 5})
        assert result == 25

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, router):
        """Test asynchronous tool execution."""

        @router.tool
        async def fetch_data(key: str) -> dict:
            """Fetch data asynchronously."""
            await asyncio.sleep(0.01)  # Simulate async operation
            return {"key": key, "value": f"data_{key}"}

        result = await router.acall_tool("fetch_data", {"key": "test"})
        assert result == {"key": "test", "value": "data_test"}

    @pytest.mark.asyncio
    async def test_kwargs_tool(self, router):
        """Test tool that accepts **kwargs."""

        def flexible_tool(**kwargs: Any) -> dict:
            """Tool that accepts any parameters."""
            return {"received": kwargs}

        router.register_function(flexible_tool)
        result = await router.acall_tool("flexible_tool", {"a": 1, "b": "test", "c": [1, 2, 3]})
        assert result == {"received": {"a": 1, "b": "test", "c": [1, 2, 3]}}

    @pytest.mark.asyncio
    async def test_tool_not_found_error(self, router):
        """Test error when tool is not found."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await router.acall_tool("nonexistent", {})

    def test_get_tool_schema(self, router):
        """Test getting tool schema."""

        def process_text(text: str, max_length: int = 100) -> str:
            """Process text with optional length limit."""
            return text[:max_length]

        router.register_function(process_text)
        schema = router.get_tool_schema("process_text")

        assert schema["name"] == "process_text"
        assert "description" in schema
        assert len(schema["parameters"]) == 2

        # Check parameter details
        params = {p["name"]: p for p in schema["parameters"]}
        assert params["text"]["required"] is True
        assert params["max_length"]["required"] is False
        assert params["max_length"]["default"] == 100

    def test_get_all_tool_schemas(self, router):
        """Test getting all tool schemas."""

        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        router.register_function(tool_a)
        router.register_function(tool_b)

        schemas = router.get_all_tool_schemas()

        # Should include built-in tools plus registered tools
        assert "tool_a" in schemas
        assert "tool_b" in schemas
        assert "tool_end" in schemas

    def test_get_tool_definitions(self, router):
        """Test getting ToolDefinitions."""

        def analyze(data: dict) -> dict:
            """Analyze data."""
            return {"analyzed": data}

        router.register_function(analyze)
        definitions = router.get_tool_definitions()

        # Find our tool
        analyze_def = next(d for d in definitions if d.name == "analyze")
        assert analyze_def.simplified_description == "Analyze data."
        assert len(analyze_def.parameters) == 1
        assert len(analyze_def.examples) > 0

    def test_reset_clears_history(self, router):
        """Test that reset clears call history."""

        def dummy_tool() -> str:
            return "result"

        router.register_function(dummy_tool)

        # Make a call to populate history
        asyncio.run(router.acall_tool("dummy_tool", {}))
        assert len(router.call_history) > 0

        # Reset and check
        router.reset()
        assert len(router.call_history) == 0
        assert len(router._registry_instances) == 0

    @pytest.mark.asyncio
    async def test_registry_tool_integration(self, mock_registry):
        """Test integration with ComponentRegistry tools."""

        # Register a tool in the registry
        @tool(name="registry_tool", namespace="test")
        def registry_tool(value: int) -> int:
            """Tool from registry."""
            return value * 2

        mock_registry.register(
            name="registry_tool",
            component=registry_tool,
            component_type="tool",
            namespace="test",
            privileged=True,
        )

        # Create router with registry
        router = UnifiedToolRouter(component_registry=mock_registry)

        # Tool should be available
        tools = router.get_available_tools()
        assert "registry_tool" in tools

        # Should be executable
        result = await router.acall_tool("registry_tool", {"value": 10})
        assert result == 20

    @pytest.mark.asyncio
    async def test_class_tool_with_execute_method(self, router):
        """Test class-based tool with execute method."""

        class DataProcessor:
            """Process data with state."""

            def __init__(self):
                self.count = 0

            def execute(self, data: str) -> dict:
                """Execute processing."""
                self.count += 1
                return {"processed": data, "count": self.count}

        # Register instance
        processor = DataProcessor()
        router.tools["processor"] = processor
        router.tool_definitions["processor"] = router._generate_tool_definition(
            processor.execute, "processor"
        )

        # Execute
        result1 = await router.acall_tool("processor", {"data": "test1"})
        result2 = await router.acall_tool("processor", {"data": "test2"})

        assert result1 == {"processed": "test1", "count": 1}
        assert result2 == {"processed": "test2", "count": 2}

    @pytest.mark.asyncio
    async def test_class_tool_async_execute(self, router):
        """Test class-based tool with async execute method."""

        class AsyncProcessor:
            """Async processor."""

            async def execute(self, value: int) -> int:
                """Execute async processing."""
                await asyncio.sleep(0.01)
                return value + 1

        processor = AsyncProcessor()
        router.tools["async_proc"] = processor
        router.tool_definitions["async_proc"] = router._generate_tool_definition(
            processor.execute, "async_proc"
        )

        result = await router.acall_tool("async_proc", {"value": 5})
        assert result == 6

    def test_tool_definition_generation_with_types(self, router):
        """Test ToolDefinition generation with various type hints."""

        def complex_tool(
            text: str, count: int = 5, enabled: bool = True, multiplier: float = 1.5
        ) -> dict:
            """Complex tool with multiple parameter types."""
            return {"text": text, "count": count, "enabled": enabled, "multiplier": multiplier}

        router.register_function(complex_tool)
        tool_def = router.tool_definitions["complex_tool"]

        # Check parameters
        params = {p.name: p for p in tool_def.parameters}

        assert params["text"].param_type == "str"
        assert params["text"].required is True

        assert params["count"].param_type == "int"
        assert params["count"].required is False
        assert params["count"].default == 5

        assert params["enabled"].param_type == "bool"
        assert params["enabled"].required is False
        assert params["enabled"].default is True

        assert params["multiplier"].param_type == "float"
        assert params["multiplier"].required is False
        assert params["multiplier"].default == 1.5

    def test_tool_examples_generation(self, router):
        """Test that examples are properly generated."""

        def example_tool(name: str, age: int) -> str:
            """Tool for testing examples."""
            return f"{name} is {age}"

        router.register_function(example_tool)
        tool_def = router.tool_definitions["example_tool"]

        assert len(tool_def.examples) > 0
        example = tool_def.examples[0]
        assert "example_tool" in example
        assert "name=" in example
        assert "age=" in example

    @pytest.mark.asyncio
    async def test_exception_propagation(self, router):
        """Test that exceptions are properly propagated."""

        def failing_tool(value: int) -> int:
            """Tool that always fails."""
            raise RuntimeError("Tool failed!")

        router.register_function(failing_tool)

        with pytest.raises(RuntimeError, match="Tool failed!"):
            await router.acall_tool("failing_tool", {"value": 1})

    def test_parameter_filtering(self, router):
        """Test that only expected parameters are passed to tools."""

        def strict_tool(a: int, b: int) -> int:
            """Tool with specific parameters."""
            return a + b

        router.register_function(strict_tool)

        # Pass extra parameters that should be filtered
        result = asyncio.run(
            router.acall_tool("strict_tool", {"a": 1, "b": 2, "c": 3, "extra": "ignored"})
        )

        assert result == 3  # Should work despite extra params


class TestUnifiedToolRouterWithPorts:
    """Test UnifiedToolRouter with port injection."""

    @pytest.fixture
    def mock_port_registry(self):
        """Create a mock port registry."""

        class MockPortRegistry:
            def get_adapter(self, port_type: str):
                if port_type == "database":
                    return MockDatabasePort()
                elif port_type == "llm":
                    return MockLLMPort()
                raise ValueError(f"Unknown port type: {port_type}")

        return MockPortRegistry()

    @pytest.fixture
    def registry_with_ports(self, mock_port_registry):
        """Create a registry with port-dependent tools."""

        registry = ComponentRegistry()
        registry.bootstrap(
            manifest=[],  # Empty manifest for testing
            dev_mode=True,
        )

        # Register a tool that requires ports
        from hexai.core.registry.models import ToolMetadata

        class DatabaseTool:
            def __init__(self, database):
                self.database = database

            def execute(self, query: str) -> list:
                return self.database.query(query)

        # Manual registration with metadata
        registry.register(
            name="db_tool",
            component=DatabaseTool,
            component_type="tool",
            namespace="test",
            privileged=True,
        )

        # Update with tool metadata
        metadata = registry.get_metadata("db_tool", namespace="test")
        metadata.tool_metadata = ToolMetadata(required_ports={"database": "database"})

        return registry

    @pytest.mark.asyncio
    async def test_tool_with_port_injection(self, registry_with_ports, mock_port_registry):
        """Test tool that requires port injection."""
        router = UnifiedToolRouter(
            component_registry=registry_with_ports, port_registry=mock_port_registry
        )

        # Should be able to call tool that needs database port
        result = await router.acall_tool("db_tool", {"query": "SELECT * FROM users"})
        assert result == [{"id": 1, "name": "test"}]


class MockDatabasePort:
    """Mock database port for testing."""

    def query(self, sql: str) -> list:
        """Mock query execution."""
        return [{"id": 1, "name": "test"}]


class MockLLMPort:
    """Mock LLM port for testing."""

    async def generate(self, prompt: str) -> str:
        """Mock text generation."""
        return f"Generated: {prompt}"
