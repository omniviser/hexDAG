"""Tests for HexDAGLib backwards-compatibility shim.

HexDAGLib is now an alias for Service. These tests verify the alias
works correctly and that Service-based libs function properly.
"""

from __future__ import annotations

import pytest

from hexdag.kernel.service import Service, get_service_tool_schemas, tool
from hexdag.stdlib.lib_base import HexDAGLib, get_lib_tool_schemas

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class EmptyLib(Service):
    """Lib with no tools."""


class CalculatorLib(Service):
    """Lib with two async tools and one sync method (excluded)."""

    @tool
    async def aadd(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    async def amultiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y

    def helper(self) -> str:
        """Sync method — should NOT be exposed as tool."""
        return "not a tool"

    async def _private_method(self) -> None:
        """Private — should NOT be exposed."""


class LifecycleLib(Service):
    """Lib that tracks lifecycle calls."""

    def __init__(self) -> None:
        self.setup_called = False
        self.teardown_called = False

    async def asetup(self) -> None:
        self.setup_called = True

    async def ateardown(self) -> None:
        self.teardown_called = True

    @tool
    async def ado_work(self) -> str:
        """Do some work."""
        return "done"


class CustomToolsLib(Service):
    """Lib that overrides get_tools()."""

    @tool
    async def avisible(self) -> str:
        """This is the only tool."""
        return "visible"

    async def ahidden(self) -> str:
        """Not decorated, so not exposed."""
        return "hidden"


# ---------------------------------------------------------------------------
# Tests: backwards-compatibility alias
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Verify that HexDAGLib is an alias for Service."""

    def test_hexdaglib_is_service(self) -> None:
        assert HexDAGLib is Service

    def test_get_lib_tool_schemas_is_get_service_tool_schemas(self) -> None:
        assert get_lib_tool_schemas is get_service_tool_schemas


# ---------------------------------------------------------------------------
# Tests: tool discovery
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    """Verify that get_tools() discovers the right methods."""

    def test_empty_lib_has_no_tools(self) -> None:
        lib = EmptyLib()
        assert lib.get_tools() == {}

    def test_calculator_exposes_async_a_methods(self) -> None:
        lib = CalculatorLib()
        tools = lib.get_tools()
        assert "aadd" in tools
        assert "amultiply" in tools

    def test_calculator_excludes_sync_method(self) -> None:
        lib = CalculatorLib()
        tools = lib.get_tools()
        assert "helper" not in tools

    def test_calculator_excludes_private_method(self) -> None:
        lib = CalculatorLib()
        tools = lib.get_tools()
        assert "_private_method" not in tools

    def test_lifecycle_methods_excluded(self) -> None:
        lib = LifecycleLib()
        tools = lib.get_tools()
        assert "asetup" not in tools
        assert "ateardown" not in tools

    def test_lifecycle_lib_exposes_work_method(self) -> None:
        lib = LifecycleLib()
        tools = lib.get_tools()
        assert "ado_work" in tools

    def test_undecorated_not_exposed(self) -> None:
        lib = CustomToolsLib()
        tools = lib.get_tools()
        assert "avisible" in tools
        assert "ahidden" not in tools


# ---------------------------------------------------------------------------
# Tests: lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio()
    async def test_setup_called(self) -> None:
        lib = LifecycleLib()
        await lib.asetup()
        assert lib.setup_called is True

    @pytest.mark.asyncio()
    async def test_teardown_called(self) -> None:
        lib = LifecycleLib()
        await lib.ateardown()
        assert lib.teardown_called is True

    @pytest.mark.asyncio()
    async def test_base_lifecycle_is_noop(self) -> None:
        lib = Service()
        await lib.asetup()
        await lib.ateardown()


# ---------------------------------------------------------------------------
# Tests: tool invocation
# ---------------------------------------------------------------------------


class TestToolInvocation:
    @pytest.mark.asyncio()
    async def test_invoke_tool_via_get_tools(self) -> None:
        lib = CalculatorLib()
        tools = lib.get_tools()
        result = await tools["aadd"](1, 2)
        assert result == 3

    @pytest.mark.asyncio()
    async def test_invoke_multiply(self) -> None:
        lib = CalculatorLib()
        tools = lib.get_tools()
        result = await tools["amultiply"](3.0, 4.0)
        assert result == 12.0


# ---------------------------------------------------------------------------
# Tests: tool schema generation
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_schema_for_calculator(self) -> None:
        lib = CalculatorLib()
        schemas = get_lib_tool_schemas(lib)
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"aadd", "amultiply"}

    def test_schema_has_correct_structure(self) -> None:
        lib = CalculatorLib()
        schemas = get_lib_tool_schemas(lib)
        add_schema = next(s for s in schemas if s["function"]["name"] == "aadd")
        assert add_schema["type"] == "function"
        func = add_schema["function"]
        assert "description" in func
        assert func["parameters"]["type"] == "object"
        assert "a" in func["parameters"]["properties"]
        assert "b" in func["parameters"]["properties"]
        assert func["parameters"]["required"] == ["a", "b"]

    def test_schema_type_inference(self) -> None:
        lib = CalculatorLib()
        schemas = get_lib_tool_schemas(lib)
        add_schema = next(s for s in schemas if s["function"]["name"] == "aadd")
        assert add_schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
        mul_schema = next(s for s in schemas if s["function"]["name"] == "amultiply")
        assert mul_schema["function"]["parameters"]["properties"]["x"]["type"] == "number"

    def test_empty_lib_schemas(self) -> None:
        lib = EmptyLib()
        schemas = get_lib_tool_schemas(lib)
        assert schemas == []


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_includes_tool_names(self) -> None:
        lib = CalculatorLib()
        r = repr(lib)
        assert "CalculatorLib" in r
        assert "aadd" in r
        assert "amultiply" in r

    def test_repr_empty_lib(self) -> None:
        lib = EmptyLib()
        assert repr(lib) == "EmptyLib(tools=[], steps=[])"
