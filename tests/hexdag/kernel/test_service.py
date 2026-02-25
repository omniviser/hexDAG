"""Tests for hexdag.kernel.service — Service base class + decorators."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from hexdag.kernel.service import (
    Service,
    get_service_step_schemas,
    get_service_tool_schemas,
    step,
    tool,
)

# ---------------------------------------------------------------------------
# Fixtures: example services
# ---------------------------------------------------------------------------


class OrderService(Service):
    """Example service for testing."""

    def __init__(self, prefix: str = "order") -> None:
        self.prefix = prefix

    @tool
    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID."""
        return {"id": f"{self.prefix}:{order_id}"}

    @step
    async def save_order(self, order_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Persist an order to the store."""
        return {"saved": True, "order_id": order_id}

    @tool
    @step
    async def validate_order(self, order_id: str) -> dict[str, Any]:
        """Validate order data."""
        return {"valid": True, "order_id": order_id}

    async def _private_helper(self) -> None:
        """Should not be exposed."""

    def sync_method(self) -> str:
        """Also not exposed (not decorated)."""
        return "sync"


class EmptyService(Service):
    """Service with no decorated methods."""


class ToolOnlyService(Service):
    """Service with only @tool methods."""

    @tool
    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search records."""
        return []


class StepOnlyService(Service):
    """Service with only @step methods."""

    @step
    async def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Transform input data."""
        return data


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestDecorators:
    """Test @tool and @step decorators."""

    def test_tool_decorator_sets_marker(self) -> None:
        """@tool sets _hexdag_tool attribute."""

        @tool
        async def my_tool() -> None:
            pass

        assert getattr(my_tool, "_hexdag_tool", False) is True

    def test_step_decorator_sets_marker(self) -> None:
        """@step sets _hexdag_step attribute."""

        @step
        async def my_step() -> None:
            pass

        assert getattr(my_step, "_hexdag_step", False) is True

    def test_stacked_decorators(self) -> None:
        """Both decorators can be stacked."""

        @tool
        @step
        async def both() -> None:
            pass

        assert getattr(both, "_hexdag_tool", False) is True
        assert getattr(both, "_hexdag_step", False) is True

    def test_decorators_preserve_function_metadata(self) -> None:
        """Decorators don't alter function name or doc."""

        @tool
        async def search(query: str) -> list:
            """Search things."""
            return []

        assert search.__name__ == "search"
        assert search.__doc__ == "Search things."

    def test_decorators_preserve_signature(self) -> None:
        """Decorators preserve the original function signature."""

        @tool
        async def find(name: str, limit: int = 5) -> list:
            """Find items."""
            return []

        sig = inspect.signature(find)
        assert "name" in sig.parameters
        assert "limit" in sig.parameters
        assert sig.parameters["limit"].default == 5


# ---------------------------------------------------------------------------
# Service.get_tools / get_steps
# ---------------------------------------------------------------------------


class TestGetTools:
    """Test Service.get_tools()."""

    def test_returns_tool_methods(self) -> None:
        svc = OrderService()
        tools = svc.get_tools()
        assert "get_order" in tools
        assert "validate_order" in tools  # has @tool
        assert "save_order" not in tools  # only @step

    def test_excludes_private_methods(self) -> None:
        svc = OrderService()
        tools = svc.get_tools()
        assert "_private_helper" not in tools

    def test_excludes_undecorated_methods(self) -> None:
        svc = OrderService()
        tools = svc.get_tools()
        assert "sync_method" not in tools
        assert "asetup" not in tools
        assert "ateardown" not in tools

    def test_empty_service_returns_empty(self) -> None:
        svc = EmptyService()
        assert svc.get_tools() == {}


class TestGetSteps:
    """Test Service.get_steps()."""

    def test_returns_step_methods(self) -> None:
        svc = OrderService()
        steps = svc.get_steps()
        assert "save_order" in steps
        assert "validate_order" in steps  # has @step
        assert "get_order" not in steps  # only @tool

    def test_empty_service_returns_empty(self) -> None:
        svc = EmptyService()
        assert svc.get_steps() == {}


class TestDualDecorated:
    """Test methods with both @tool and @step."""

    def test_appears_in_both(self) -> None:
        svc = OrderService()
        assert "validate_order" in svc.get_tools()
        assert "validate_order" in svc.get_steps()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test asetup / ateardown lifecycle."""

    @pytest.mark.asyncio
    async def test_default_lifecycle_is_noop(self) -> None:
        """Base asetup/ateardown do nothing and don't raise."""
        svc = Service()
        await svc.asetup()
        await svc.ateardown()

    @pytest.mark.asyncio
    async def test_lifecycle_can_be_overridden(self) -> None:
        """Subclasses can override lifecycle methods."""
        events: list[str] = []

        class MyService(Service):
            async def asetup(self) -> None:
                events.append("setup")

            async def ateardown(self) -> None:
                events.append("teardown")

        svc = MyService()
        await svc.asetup()
        await svc.ateardown()
        assert events == ["setup", "teardown"]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    """Test that decorated methods actually execute correctly."""

    @pytest.mark.asyncio
    async def test_tool_executes(self) -> None:
        svc = OrderService(prefix="test")
        result = await svc.get_order(order_id="123")
        assert result == {"id": "test:123"}

    @pytest.mark.asyncio
    async def test_step_executes(self) -> None:
        svc = OrderService()
        result = await svc.save_order(order_id="456", data={"item": "widget"})
        assert result == {"saved": True, "order_id": "456"}

    @pytest.mark.asyncio
    async def test_dual_decorated_executes(self) -> None:
        svc = OrderService()
        result = await svc.validate_order(order_id="789")
        assert result == {"valid": True, "order_id": "789"}


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


class TestToolSchemas:
    """Test get_service_tool_schemas()."""

    def test_generates_schemas_for_tools(self) -> None:
        svc = OrderService()
        schemas = get_service_tool_schemas(svc)
        names = {s["function"]["name"] for s in schemas}
        assert "get_order" in names
        assert "validate_order" in names
        assert "save_order" not in names  # step-only

    def test_schema_structure(self) -> None:
        svc = ToolOnlyService()
        schemas = get_service_tool_schemas(svc)
        assert len(schemas) == 1

        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search records."

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert "query" in params["required"]
        assert "limit" not in params["required"]  # has default

    def test_type_inference(self) -> None:
        svc = ToolOnlyService()
        schemas = get_service_tool_schemas(svc)
        params = schemas[0]["function"]["parameters"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"

    def test_empty_service(self) -> None:
        svc = EmptyService()
        assert get_service_tool_schemas(svc) == []


class TestStepSchemas:
    """Test get_service_step_schemas()."""

    def test_generates_schemas_for_steps(self) -> None:
        svc = OrderService()
        schemas = get_service_step_schemas(svc)
        names = {s["function"]["name"] for s in schemas}
        assert "save_order" in names
        assert "validate_order" in names
        assert "get_order" not in names  # tool-only

    def test_empty_service(self) -> None:
        svc = EmptyService()
        assert get_service_step_schemas(svc) == []


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    """Test Service.__repr__."""

    def test_repr_shows_tools_and_steps(self) -> None:
        svc = OrderService()
        r = repr(svc)
        assert "OrderService" in r
        assert "tools=" in r
        assert "steps=" in r
        assert "get_order" in r
        assert "save_order" in r

    def test_empty_repr(self) -> None:
        svc = EmptyService()
        assert repr(svc) == "EmptyService(tools=[], steps=[])"
