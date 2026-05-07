"""Tests for ServiceCallNode factory."""

from __future__ import annotations

from typing import Any

import pytest

from hexdag.kernel.context import clear_execution_context, set_services
from hexdag.kernel.service import Service, step, tool
from hexdag.stdlib.nodes.service_call_node import ServiceCallNode

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
    async def record_counter(
        self,
        load_id: str,
        counter_amount: float,
        carrier_email: str,
    ) -> dict[str, Any]:
        """Step with params that match upstream output field names."""
        return {
            "load_id": load_id,
            "counter_amount": counter_amount,
            "carrier_email": carrier_email,
        }

    @step
    def sync_step(self, value: int) -> dict[str, int]:
        """Synchronous step for testing sync dispatch."""
        return {"doubled": value * 2}

    @step
    async def accepts_kwargs(self, order_id: str, **kwargs: Any) -> dict[str, Any]:
        """Step that accepts **kwargs — should receive all input."""
        return {"order_id": order_id, "extra_keys": list(kwargs.keys())}

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

    def test_no_output_model(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        assert node.out_model is None

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
        assert result["saved"] is True
        assert result["order_id"] == "123"

    @pytest.mark.asyncio
    async def test_calls_sync_step(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="double", service="orders", method="sync_step")
        result = await node.fn({"value": 5})
        assert result["doubled"] == 10

    @pytest.mark.asyncio
    async def test_error_on_missing_service(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="save", service="nonexistent", method="save_order")
        with pytest.raises(KeyError, match="nonexistent"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_error_on_tool_only_method(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="get", service="orders", method="get_order")
        with pytest.raises(AttributeError, match="get_order"):
            await node.fn({"order_id": "123"})

    @pytest.mark.asyncio
    async def test_error_on_nonexistent_method(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="nope", service="orders", method="does_not_exist")
        with pytest.raises(AttributeError, match="does_not_exist"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_error_on_no_services(self) -> None:
        clear_execution_context()  # remove services
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        with pytest.raises(RuntimeError, match="No services"):
            await node.fn({})

    @pytest.mark.asyncio
    async def test_step_exception_propagates(self) -> None:
        factory = ServiceCallNode()
        node = factory(name="fail", service="orders", method="failing_step")
        with pytest.raises(RuntimeError, match="Intentional failure"):
            await node.fn({})


# ---------------------------------------------------------------------------
# Input filtering tests
# ---------------------------------------------------------------------------


class TestServiceCallNodeInputFiltering:
    """Verify that input is filtered to match the @step method signature.

    When ``_apply_input_mapping`` merges ALL upstream node results into
    ``input_data``, extra kwargs would cause ``TypeError`` on methods with
    strict signatures. The node must filter to accepted params only.
    """

    @pytest.fixture(autouse=True)
    def _setup_context(self) -> None:  # noqa: PT004
        svc = MockOrderService()
        set_services({"orders": svc})
        yield  # type: ignore[misc]
        clear_execution_context()

    @pytest.mark.asyncio
    async def test_filters_extra_kwargs_for_strict_signature(self) -> None:
        """Extra keys in input_data must be silently dropped."""
        factory = ServiceCallNode()
        node = factory(name="save", service="orders", method="save_order")
        # Pass extra keys that save_order(order_id, data) does not accept
        result = await node.fn({
            "order_id": "X1",
            "data": {"item": "bolt"},
            "upstream_noise": 42,
            "constants": {"key": "val"},
        })
        assert result["saved"] is True
        assert result["order_id"] == "X1"

    @pytest.mark.asyncio
    async def test_passes_all_to_kwargs_method(self) -> None:
        """Methods with **kwargs must receive the full input dict."""
        factory = ServiceCallNode()
        node = factory(name="kw", service="orders", method="accepts_kwargs")
        result = await node.fn({
            "order_id": "X2",
            "extra_field": "hello",
            "another": 99,
        })
        assert result["order_id"] == "X2"
        assert "extra_field" in result["extra_keys"]
        assert "another" in result["extra_keys"]

    @pytest.mark.asyncio
    async def test_sync_step_also_filtered(self) -> None:
        """Sync steps must also be filtered."""
        factory = ServiceCallNode()
        node = factory(name="double", service="orders", method="sync_step")
        result = await node.fn({"value": 7, "noise": "ignored"})
        assert result["doubled"] == 14


# ---------------------------------------------------------------------------
# Auto-inference tests
# ---------------------------------------------------------------------------


class TestServiceCallNodeAutoInference:
    """Verify that missing params are auto-inferred from upstream node outputs.

    When the additive input system provides ``{node_name: result_dict, ...}``,
    the node should search inside those result dicts for keys matching
    the @step method's parameter names.
    """

    @pytest.fixture(autouse=True)
    def _setup_context(self) -> None:  # noqa: PT004
        svc = MockOrderService()
        set_services({"orders": svc})
        yield  # type: ignore[misc]
        clear_execution_context()

    @pytest.mark.asyncio
    async def test_auto_infers_from_upstream_dicts(self) -> None:
        """Params found inside upstream result dicts are auto-mapped."""
        factory = ServiceCallNode()
        node = factory(name="rec", service="orders", method="record_counter")
        # Simulate additive input: upstream node results as nested dicts
        result = await node.fn({
            "resolved_match": {"load_id": "L42", "origin": "Dallas"},
            "compute_counter": {"counter_amount": 500.0, "round": 2},
            "input": {"carrier_email": "test@carrier.com", "extra": "ignored"},
        })
        assert result["load_id"] == "L42"
        assert result["counter_amount"] == 500.0
        assert result["carrier_email"] == "test@carrier.com"

    @pytest.mark.asyncio
    async def test_ambiguous_param_not_inferred(self) -> None:
        """If a param name exists in multiple upstream sources, skip it."""
        factory = ServiceCallNode()
        node = factory(name="rec", service="orders", method="record_counter")
        # load_id exists in BOTH upstream sources — should not be auto-mapped
        with pytest.raises(TypeError):
            await node.fn({
                "source_a": {"load_id": "L1", "counter_amount": 500.0},
                "source_b": {"load_id": "L2"},
                "input": {"carrier_email": "test@carrier.com"},
            })

    @pytest.mark.asyncio
    async def test_top_level_takes_priority(self) -> None:
        """Params already at top level are not overridden by auto-inference."""
        factory = ServiceCallNode()
        node = factory(name="rec", service="orders", method="record_counter")
        result = await node.fn({
            "load_id": "TOP_LEVEL",  # Already at top level
            "upstream": {"load_id": "NESTED", "counter_amount": 300.0},
            "input": {"carrier_email": "test@carrier.com"},
        })
        assert result["load_id"] == "TOP_LEVEL"
        assert result["counter_amount"] == 300.0


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
