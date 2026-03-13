"""Tests for middleware composition and prepare_ports."""

from __future__ import annotations

from typing import Any

from hexdag.kernel.ports.llm import (
    SupportsGeneration,
    SupportsStructuredOutput,
)
from hexdag.kernel.ports.tool_router import ToolRouter
from hexdag.stdlib.adapters.mock import MockLLM
from hexdag.stdlib.middleware.observable import ObservableLLM
from hexdag.stdlib.middleware.structured_output import StructuredOutputFallback


class TestMiddlewareComposition:
    """Test that middleware stacking preserves isinstance checks."""

    def test_mock_llm_is_structured_output(self) -> None:
        """MockLLM natively supports SupportsStructuredOutput."""
        mock = MockLLM()
        assert isinstance(mock, SupportsStructuredOutput)

    def test_mock_llm_skips_fallback(self) -> None:
        """prepare_ports should NOT add StructuredOutputFallback for MockLLM."""
        from hexdag.kernel.orchestration.port_wrappers import prepare_ports

        ports = {"llm": MockLLM()}
        prepared = prepare_ports(ports)
        llm = prepared["llm"]

        # Should be ObservableLLM wrapping MockLLM directly (no fallback)
        assert isinstance(llm, ObservableLLM)
        assert isinstance(llm._inner, MockLLM)

    def test_non_structured_adapter_gets_fallback(self) -> None:
        """Adapter without SupportsStructuredOutput gets fallback middleware."""
        from hexdag.kernel.orchestration.port_wrappers import prepare_ports

        class PlainAdapter:
            async def aresponse(self, messages: Any) -> str:
                return "plain"

        ports = {"llm": PlainAdapter()}
        prepared = prepare_ports(ports)
        llm = prepared["llm"]

        # Should be ObservableLLM → StructuredOutputFallback → PlainAdapter
        assert isinstance(llm, ObservableLLM)
        assert isinstance(llm._inner, StructuredOutputFallback)

    def test_full_stack_isinstance_checks(self) -> None:
        """Full middleware stack passes all isinstance checks."""
        from hexdag.kernel.orchestration.port_wrappers import prepare_ports

        class PlainAdapter:
            async def aresponse(self, messages: Any) -> str:
                return "plain"

        ports = {"llm": PlainAdapter()}
        prepared = prepare_ports(ports)
        llm = prepared["llm"]

        assert isinstance(llm, SupportsGeneration)
        assert isinstance(llm, SupportsStructuredOutput)

    def test_tool_router_wrapped(self) -> None:
        """ToolRouter gets wrapped with ObservableToolRouter."""
        from hexdag.kernel.orchestration.port_wrappers import prepare_ports

        ports = {"tool_router": ToolRouter()}
        prepared = prepare_ports(ports)
        tr = prepared["tool_router"]

        # Should NOT be HealthCheckable (to avoid dict health check bug)
        from hexdag.kernel.protocols import HealthCheckable

        assert not isinstance(tr, HealthCheckable)

    def test_non_llm_port_passthrough(self) -> None:
        """Non-LLM, non-ToolRouter ports pass through unchanged."""
        from hexdag.kernel.orchestration.port_wrappers import prepare_ports

        sentinel = object()
        ports = {"memory": sentinel}
        prepared = prepare_ports(ports)
        assert prepared["memory"] is sentinel
