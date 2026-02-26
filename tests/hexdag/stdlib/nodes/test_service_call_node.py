"""Tests for ServiceCallNode factory."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.context import clear_execution_context, set_services
from hexdag.kernel.service import Service, step, tool
from hexdag.stdlib.nodes.service_call_node import ServiceCallNode, ServiceCallOutput

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class MockOrderService(Service):
    """Service with both @step and @tool methods for testing."""

    @step
    async def save_order(self, order_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Persist an order."""
        return {"saved": True, "order_id": order_id, "data": data}

    @step
    def sync_step(self, value: int) -> dict[str, int]:
        """Synchronous step for testing sync dispatch."""
        return {"doubled": value * 2}

    @tool
    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Tool-only method — not a step."""
        return {"order_id": order_id}

    @step
    async def failing_step(self) -> None:
        """Step that always raises."""
        msg = "Intentional failure"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Factory creation tests
# ---------------------------------------------------------------------------


class TestServiceCallNodeCreation:
    def test_creates_node_spec(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        assert node.name == "save"

    def test_passes_dependencies(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order", deps=["prev"])
        assert "prev" in node.deps

    def test_output_model_is_service_call_output(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        assert node.out_model is ServiceCallOutput

    def test_framework_params_extracted(self) -> None:
        factory = ServiceCallNode()
        node = factory(
            name="save",
            service="orders",
            method="save_order",
            timeout=30,
            max_retries=3,
            when="status == 'active'",
        )
        assert node.timeout == 30
        assert node.max_retries == 3
        assert node.when == "status == 'active'"

    def test_system_kind_marker(self) -> None:
        assert ServiceCallNode._hexdag_system_kind is True


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestServiceCallNodeExecution:
    @pytest.fixture(autouse=True)
    def _setup_context(self) -> None:  # noqa: PT004
        svc = MockOrderService()
        set_services({"orders": svc})
        yield  # type: ignore[misc]
        clear_execution_context()

    @pytest.mark.asyncio
    async def test_calls_step_method(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        result = await node.fn({"order_id": "123", "data": {"item": "widget"}})
        assert result["error"] is None
        assert result["result"]["saved"] is True
        assert result["result"]["order_id"] == "123"
        assert result["service"] == "orders"
        assert result["method"] == "save_order"

    @pytest.mark.asyncio
    async def test_calls_sync_step(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="double", service="orders", method="sync_step")
        result = await node.fn({"value": 5})
        assert result["error"] is None
        assert result["result"]["doubled"] == 10

    @pytest.mark.asyncio
    async def test_error_on_missing_service(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="nonexistent", method="save_order")
        result = await node.fn({})
        assert result["error"] is not None
        assert "nonexistent" in result["error"]
        assert "orders" in result["error"]  # lists available services

    @pytest.mark.asyncio
    async def test_error_on_tool_only_method(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="get", service="orders", method="get_order")
        result = await node.fn({"order_id": "123"})
        assert result["error"] is not None
        assert "get_order" in result["error"]
        assert "not a @step" in result["error"]

    @pytest.mark.asyncio
    async def test_error_on_nonexistent_method(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="nope", service="orders", method="does_not_exist")
        result = await node.fn({})
        assert result["error"] is not None
        assert "does_not_exist" in result["error"]

    @pytest.mark.asyncio
    async def test_error_on_no_services(self) -> None:
        clear_execution_context()  # remove services
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        result = await node.fn({})
        assert result["error"] is not None
        assert "No services" in result["error"]

    @pytest.mark.asyncio
    async def test_step_exception_returned_as_error(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="fail", service="orders", method="failing_step")
        result = await node.fn({})
        assert result["error"] is not None
        assert "Intentional failure" in result["error"]
        assert result["result"] is None


# ---------------------------------------------------------------------------
# Alias / discovery tests
# ---------------------------------------------------------------------------


class TestServiceCallNodeDiscovery:
    def test_yaml_alias_registered(self) -> None:
        from hexdag.stdlib.nodes.base_node_factory import BaseNodeFactory

        assert "service_call_node" in BaseNodeFactory._registry

    def test_system_alias_in_discovery(self) -> None:
        from hexdag.stdlib.nodes._discovery import discover_node_factories

        # Clear the lru_cache so discovery picks up the new node
        discover_node_factories.cache_clear()
        aliases = discover_node_factories()
        assert "service_call_node" in aliases
        assert "core:service_call_node" in aliases
        assert "core:service_call" in aliases
        assert "system:service_call_node" in aliases
        assert "system:service_call" in aliases
