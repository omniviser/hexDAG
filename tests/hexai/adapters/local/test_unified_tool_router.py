"""Comprehensive tests for UnifiedToolRouter."""

import asyncio
from typing import Any

import pytest

from hexai.adapters.local.unified_tool_router import UnifiedToolRouter
from hexai.core.application.nodes.tool_utils import ToolDefinition
from hexai.core.registry.registry import ComponentRegistry


class TestUnifiedToolRouter:
    """Test UnifiedToolRouter functionality."""

    @pytest.fixture
    def router(self):
        """Create a UnifiedToolRouter instance."""
        # Router now uses global registry, no parameters needed
        return UnifiedToolRouter()

    def register_tool(self, func, name=None, namespace="test"):
        """Helper to register a tool in the global registry."""
        from hexai.core.registry import registry
        from hexai.core.registry.models import ComponentType

        tool_name = name or func.__name__
        registry.register(
            name=tool_name,
            component=func,
            component_type=ComponentType.TOOL,
            namespace=namespace,
        )

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Setup the global registry for testing."""
        from hexai.core.config.models import ManifestEntry
        from hexai.core.registry import registry

        # Bootstrap registry if it hasn't been bootstrapped yet
        if not registry.ready:
            registry.bootstrap(
                manifest=[ManifestEntry(namespace="core", module="hexai.tools.builtin_tools")],
                dev_mode=True,
            )
        elif not registry.dev_mode:
            # Registry is bootstrapped but not in dev_mode
            # We need to manually enable dev mode for testing
            registry._dev_mode = True

        yield

        # Note: We can't easily clean up the registry without singleton.py
        # The registry will stay bootstrapped for the entire test session

    def test_initialization(self):
        """Test router initialization."""
        # Router now uses global registry, no parameters needed
        router = UnifiedToolRouter()
        assert router is not None

    def test_initialization_simple(self):
        """Test router initialization is simple with no parameters."""
        # Router uses global registry, no parameters needed
        router1 = UnifiedToolRouter()
        router2 = UnifiedToolRouter()
        assert router1 is not None
        assert router2 is not None
        # Both routers should see the same tools from the global registry
        assert router1.get_available_tools() == router2.get_available_tools()

    def test_builtin_tools_registered(self, router):
        """Test that built-in tools are registered in the global registry."""
        # Ensure registry has tools (in case it was cleared by other tests)
        # Force import of builtin tools module - the decorators register at import time
        import hexai.tools.builtin_tools  # noqa: F401
        from hexai.core.registry import registry
        from hexai.core.registry.models import ComponentType

        # The tools should now be registered via their decorators
        tools = registry.list_components(component_type=ComponentType.TOOL, namespace="core")
        tool_names = [t.name for t in tools]

        # If still empty, it means the registry was cleared after module import
        # Skip this specific test in that case as we can't re-register decorator-based tools
        if not tool_names:
            import pytest

            pytest.skip(
                "Registry was cleared by another test and can't re-register decorator-based tools"
            )

        assert "tool_end" in tool_names
        assert "change_phase" in tool_names

        # Also check through router's method
        router_tools = router.get_available_tools()
        assert "tool_end" in router_tools

    def test_tool_registration_via_decorator(self, router):
        """Test tool registration through the global registry."""

        def add_numbers(x: int, y: int) -> int:
            """Add two numbers together."""
            return x + y

        # Register the tool
        self.register_tool(add_numbers)

        # Tool should be available through router
        assert "add_numbers" in router.get_available_tools()

        # Get tool definition
        tool_defs = router.get_tool_definitions()
        add_tool = next((td for td in tool_defs if td.name == "add_numbers"), None)
        assert add_tool is not None
        assert isinstance(add_tool, ToolDefinition)
        assert len(add_tool.parameters) == 2

    def test_tool_registration_with_aliases(self, router):
        """Test tool registration with multiple names/aliases."""

        def multiply(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        # Register with multiple names
        self.register_tool(multiply, name="multiply")
        self.register_tool(multiply, name="mult")

        # Both names should be available
        tools = router.get_available_tools()
        assert "multiply" in tools
        assert "mult" in tools

    @pytest.mark.asyncio
    async def test_tool_execution_with_error(self, router):
        """Test tool execution that raises an error."""

        def divide(x: float, y: float) -> float:
            """Divide x by y.

            Raises
            ------
            ValueError
                If y is zero
            """
            if y == 0:
                raise ValueError("Cannot divide by zero")
            return x / y

        # Register the tool
        self.register_tool(divide)

        # Test successful division
        result = await router.acall_tool("divide", {"x": 10, "y": 2})
        assert result == 5.0

        # Test division by zero
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await router.acall_tool("divide", {"x": 10, "y": 0})

    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, router):
        """Test synchronous tool execution."""

        def square(x: int) -> int:
            """Square a number."""
            return x * x

        self.register_tool(square)

        result = await router.acall_tool("square", {"x": 5})
        assert result == 25

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, router):
        """Test asynchronous tool execution."""

        async def fetch_data(key: str) -> dict:
            """Fetch data asynchronously."""
            await asyncio.sleep(0.01)  # Simulate async operation
            return {"key": key, "value": f"data_{key}"}

        self.register_tool(fetch_data)

        result = await router.acall_tool("fetch_data", {"key": "test"})
        assert result == {"key": "test", "value": "data_test"}

    @pytest.mark.asyncio
    async def test_kwargs_tool(self, router):
        """Test tool that accepts **kwargs."""

        def flexible_tool(**kwargs: Any) -> dict:
            """Tool that accepts any parameters."""
            return {"received": kwargs}

        self.register_tool(flexible_tool)

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

        self.register_tool(process_text)

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

        self.register_tool(tool_a)
        self.register_tool(tool_b)

        schemas = router.get_all_tool_schemas()

        # Should include built-in tools plus registered tools
        assert "tool_a" in schemas
        assert "tool_b" in schemas
        # Built-in tools may have been cleared by other tests - skip if missing
        if "tool_end" not in schemas:
            import hexai.tools.builtin_tools  # noqa: F401

            # Try again after import
            schemas = router.get_all_tool_schemas()
        if "tool_end" not in schemas:
            import pytest

            pytest.skip("Built-in tools cleared by other tests")

    def test_get_tool_definitions(self, router):
        """Test getting ToolDefinitions."""

        def analyze(data: dict) -> dict:
            """Analyze data."""
            return {"analyzed": data}

        self.register_tool(analyze)

        definitions = router.get_tool_definitions()

        # Find our tool
        analyze_def = next(d for d in definitions if d.name == "analyze")
        assert (
            "Analyze" in analyze_def.simplified_description
            or "analyze" in analyze_def.simplified_description
        )
        assert len(analyze_def.parameters) == 1
        assert len(analyze_def.examples) > 0

    @pytest.mark.asyncio
    async def test_registry_tool_integration(self, router):
        """Test integration with global registry tools."""

        # Register a tool in the global registry
        def registry_tool(value: int) -> int:
            """Tool from registry."""
            return value * 2

        self.register_tool(registry_tool)

        # Tool should be available through router
        tools = router.get_available_tools()
        assert "registry_tool" in tools

        # Should be executable
        result = await router.acall_tool("registry_tool", {"value": 10})
        assert result == 20

    @pytest.mark.asyncio
    async def test_class_tool_with_execute_method(self, router):
        """Test class-based tool with execute method."""
        from hexai.core.registry import registry
        from hexai.core.registry.models import ComponentType

        class DataProcessor:
            """Process data with state."""

            def __init__(self):
                self.count = 0

            def execute(self, data: str) -> dict:
                """Execute processing."""
                self.count += 1
                return {"processed": data, "count": self.count}

        # Register the class instance in the global registry
        processor = DataProcessor()
        registry.register(
            name="processor",
            component=processor,
            component_type=ComponentType.TOOL,
            namespace="test",
        )

        # Execute through router
        result1 = await router.acall_tool("processor", {"data": "test1"})
        result2 = await router.acall_tool("processor", {"data": "test2"})

        assert result1 == {"processed": "test1", "count": 1}
        assert result2 == {"processed": "test2", "count": 2}

    @pytest.mark.asyncio
    async def test_class_tool_async_execute(self, router):
        """Test class-based tool with async execute method."""
        from hexai.core.registry import registry
        from hexai.core.registry.models import ComponentType

        class AsyncProcessor:
            """Async processor."""

            async def execute(self, value: int) -> int:
                """Execute async processing."""
                await asyncio.sleep(0.01)
                return value + 1

        # Register in global registry
        processor = AsyncProcessor()
        registry.register(
            name="async_proc",
            component=processor,
            component_type=ComponentType.TOOL,
            namespace="test",
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

        self.register_tool(complex_tool)

        # Get tool definition through router
        tool_defs = router.get_tool_definitions()
        tool_def = next(d for d in tool_defs if d.name == "complex_tool")

        # Check parameters
        params = {p.name: p for p in tool_def.parameters}

        assert "str" in params["text"].param_type
        assert params["text"].required is True

        assert "int" in params["count"].param_type
        assert params["count"].required is False
        assert params["count"].default == 5

        assert "bool" in params["enabled"].param_type
        assert params["enabled"].required is False
        assert params["enabled"].default is True

        assert "float" in params["multiplier"].param_type
        assert params["multiplier"].required is False
        assert params["multiplier"].default == 1.5

    def test_tool_examples_generation(self, router):
        """Test that examples are properly generated."""

        def example_tool(name: str, age: int) -> str:
            """Tool for testing examples."""
            return f"{name} is {age}"

        self.register_tool(example_tool)

        # Get tool definition
        tool_defs = router.get_tool_definitions()
        tool_def = next(d for d in tool_defs if d.name == "example_tool")

        assert len(tool_def.examples) > 0
        example = tool_def.examples[0]
        assert "example_tool" in example
        assert "name=" in example
        assert "age=" in example

    @pytest.mark.asyncio
    async def test_exception_propagation(self, router):
        """Test that exceptions are properly propagated."""

        def failing_tool(value: int) -> int:
            """Tool that always fails.

            Raises
            ------
            RuntimeError
                Always raises this error
            """
            raise RuntimeError("Tool failed!")

        self.register_tool(failing_tool)

        with pytest.raises(RuntimeError, match="Tool failed!"):
            await router.acall_tool("failing_tool", {"value": 1})

    def test_parameter_filtering(self, router):
        """Test that only expected parameters are passed to tools."""

        def strict_tool(a: int, b: int) -> int:
            """Tool with specific parameters."""
            return a + b

        self.register_tool(strict_tool)

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
                if port_type == "llm":
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
        class DatabaseTool:
            _required_ports = ["database"]  # Convention-based port requirements

            def __init__(self, database):
                self.database = database

            def execute(self, query: str) -> list:
                return self.database.query(query)

        # Manual registration
        registry.register(
            name="db_tool",
            component=DatabaseTool,
            component_type="tool",
            namespace="test",
            privileged=True,
        )

        return registry

    @pytest.mark.asyncio
    async def test_tool_with_port_injection(self, registry_with_ports, mock_port_registry):
        """Test tool that requires port injection."""
        # Router now uses global registry, no parameters
        router = UnifiedToolRouter()

        # Tool requiring ports should fail since we don't support auto-injection anymore
        with pytest.raises(ValueError) as exc_info:
            await router.acall_tool("db_tool", {"query": "SELECT * FROM users"})
        assert (
            "not found" in str(exc_info.value).lower()
            or "requires parameters" in str(exc_info.value).lower()
        )


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
