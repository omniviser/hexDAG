"""Tests for Phase C2: declarative middleware stacking from YAML config."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from hexdag.kernel.orchestration.port_wrappers import prepare_ports
from hexdag.stdlib.adapters.mock import MockLLM
from hexdag.stdlib.middleware.observable import ObservableLLM

# ---------------------------------------------------------------------------
# Dummy middleware for testing (must accept inner as sole constructor arg)
# ---------------------------------------------------------------------------


class _UppercaseMiddleware:
    """Test middleware that wraps an adapter."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _LoggingMiddleware:
    """Another test middleware."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# Map of fake module paths → actual classes for testing
_TEST_MW_CLASSES: dict[str, type] = {
    "test.UppercaseMiddleware": _UppercaseMiddleware,
    "test.LoggingMiddleware": _LoggingMiddleware,
}


def _mock_resolve(module_path: str) -> type:
    """Resolve test middleware paths to actual classes."""
    if module_path in _TEST_MW_CLASSES:
        return _TEST_MW_CLASSES[module_path]
    from hexdag.kernel.resolver import resolve

    return resolve(module_path)


class TestUserMiddlewareStacking:
    """Test that user-declared middleware is applied before auto-middleware."""

    def test_single_user_middleware(self) -> None:
        """Single user middleware wraps adapter, auto-middleware wraps that."""
        ports = {"llm": MockLLM()}
        mw_config = {"llm": ["test.UppercaseMiddleware"]}

        with patch(
            "hexdag.kernel.orchestration.port_wrappers._resolve_middleware_class",
            side_effect=_mock_resolve,
        ):
            prepared = prepare_ports(ports, middleware_config=mw_config)

        llm = prepared["llm"]

        # Outermost: ObservableLLM (auto)
        assert isinstance(llm, ObservableLLM)
        # Inner: _UppercaseMiddleware (user)
        assert isinstance(llm._inner, _UppercaseMiddleware)
        # Innermost: MockLLM (adapter)
        assert isinstance(llm._inner._inner, MockLLM)

    def test_multiple_user_middleware(self) -> None:
        """Multiple user middleware applied inner-to-outer."""
        ports = {"llm": MockLLM()}
        mw_config = {"llm": ["test.UppercaseMiddleware", "test.LoggingMiddleware"]}

        with patch(
            "hexdag.kernel.orchestration.port_wrappers._resolve_middleware_class",
            side_effect=_mock_resolve,
        ):
            prepared = prepare_ports(ports, middleware_config=mw_config)

        llm = prepared["llm"]

        # Stack: ObservableLLM → _LoggingMiddleware → _UppercaseMiddleware → MockLLM
        assert isinstance(llm, ObservableLLM)
        assert isinstance(llm._inner, _LoggingMiddleware)
        assert isinstance(llm._inner._inner, _UppercaseMiddleware)
        assert isinstance(llm._inner._inner._inner, MockLLM)

    def test_no_middleware_config_unchanged(self) -> None:
        """Without middleware_config, behavior is unchanged (backward compat)."""
        ports = {"llm": MockLLM()}

        prepared = prepare_ports(ports)
        llm = prepared["llm"]

        assert isinstance(llm, ObservableLLM)
        assert isinstance(llm._inner, MockLLM)

    def test_empty_middleware_config(self) -> None:
        """Empty middleware config behaves like no config."""
        ports = {"llm": MockLLM()}
        prepared = prepare_ports(ports, middleware_config={})
        llm = prepared["llm"]

        assert isinstance(llm, ObservableLLM)
        assert isinstance(llm._inner, MockLLM)

    def test_middleware_on_non_llm_port(self) -> None:
        """User middleware on non-LLM port (no auto-middleware)."""
        sentinel = object()
        ports = {"custom": sentinel}
        mw_config = {"custom": ["test.UppercaseMiddleware"]}

        with patch(
            "hexdag.kernel.orchestration.port_wrappers._resolve_middleware_class",
            side_effect=_mock_resolve,
        ):
            prepared = prepare_ports(ports, middleware_config=mw_config)

        custom = prepared["custom"]

        # User middleware applied, no auto-middleware (not LLM or ToolRouter)
        assert isinstance(custom, _UppercaseMiddleware)
        assert custom._inner is sentinel

    def test_middleware_only_for_declared_ports(self) -> None:
        """Middleware config only affects ports that have declarations."""
        ports = {"llm": MockLLM(), "memory": object()}
        mw_config = {"llm": ["test.UppercaseMiddleware"]}

        with patch(
            "hexdag.kernel.orchestration.port_wrappers._resolve_middleware_class",
            side_effect=_mock_resolve,
        ):
            prepared = prepare_ports(ports, middleware_config=mw_config)

        # LLM gets user middleware
        assert isinstance(prepared["llm"]._inner, _UppercaseMiddleware)
        # Memory passes through unchanged
        assert not isinstance(prepared["memory"], _UppercaseMiddleware)


class TestMiddlewareDefinitionPlugin:
    """Test the kind: Middleware compiler plugin."""

    def test_register_and_lookup(self) -> None:
        """Register a middleware stack and look it up by name."""
        from hexdag.compiler.plugins.middleware_definition import (
            _middleware_registry,
            clear_middleware_registry,
            get_middleware_stack,
        )

        clear_middleware_registry()

        _middleware_registry["test-stack"] = [
            "hexdag.stdlib.middleware.structured_output.StructuredOutputFallback",
        ]

        result = get_middleware_stack("test-stack")
        assert result == [
            "hexdag.stdlib.middleware.structured_output.StructuredOutputFallback",
        ]

        clear_middleware_registry()

    def test_lookup_nonexistent(self) -> None:
        """Looking up nonexistent name returns None."""
        from hexdag.compiler.plugins.middleware_definition import (
            clear_middleware_registry,
            get_middleware_stack,
        )

        clear_middleware_registry()
        assert get_middleware_stack("nonexistent") is None

    def test_plugin_can_handle(self) -> None:
        """Plugin handles kind: Middleware."""
        from hexdag.compiler.plugins.middleware_definition import MiddlewareDefinitionPlugin

        plugin = MiddlewareDefinitionPlugin()
        assert plugin.can_handle({"kind": "Middleware"})
        assert not plugin.can_handle({"kind": "Pipeline"})
        assert not plugin.can_handle({"kind": "Macro"})

    def test_plugin_build_registers_stack(self) -> None:
        """Plugin build registers the middleware stack."""
        from hexdag.compiler.plugins.middleware_definition import (
            MiddlewareDefinitionPlugin,
            clear_middleware_registry,
            get_middleware_stack,
        )
        from hexdag.kernel.domain.dag import DirectedGraph

        clear_middleware_registry()

        plugin = MiddlewareDefinitionPlugin()
        doc = {
            "kind": "Middleware",
            "metadata": {"name": "prod-llm"},
            "spec": {
                "stack": [
                    "hexdag.stdlib.middleware.structured_output.StructuredOutputFallback",
                ]
            },
        }

        result = plugin.build(doc, None, DirectedGraph())  # type: ignore[arg-type]
        assert result is None

        stack = get_middleware_stack("prod-llm")
        assert stack == [
            "hexdag.stdlib.middleware.structured_output.StructuredOutputFallback",
        ]

        clear_middleware_registry()

    def test_plugin_missing_name_raises(self) -> None:
        """Plugin raises if metadata.name is missing."""
        from hexdag.compiler.plugins.middleware_definition import MiddlewareDefinitionPlugin
        from hexdag.kernel.domain.dag import DirectedGraph
        from hexdag.kernel.exceptions import YamlPipelineBuilderError

        plugin = MiddlewareDefinitionPlugin()
        doc = {
            "kind": "Middleware",
            "metadata": {},
            "spec": {"stack": ["some.Middleware"]},
        }

        with pytest.raises(YamlPipelineBuilderError, match="missing 'metadata.name'"):
            plugin.build(doc, None, DirectedGraph())  # type: ignore[arg-type]

    def test_plugin_missing_stack_raises(self) -> None:
        """Plugin raises if spec.stack is missing."""
        from hexdag.compiler.plugins.middleware_definition import MiddlewareDefinitionPlugin
        from hexdag.kernel.domain.dag import DirectedGraph
        from hexdag.kernel.exceptions import YamlPipelineBuilderError

        plugin = MiddlewareDefinitionPlugin()
        doc = {
            "kind": "Middleware",
            "metadata": {"name": "empty"},
            "spec": {},
        }

        with pytest.raises(YamlPipelineBuilderError, match="no stack"):
            plugin.build(doc, None, DirectedGraph())  # type: ignore[arg-type]


class TestExtractMiddlewareConfig:
    """Test OrchestratorFactory._extract_middleware_config."""

    def _factory(self) -> Any:
        from hexdag.kernel.orchestration.orchestrator_factory import OrchestratorFactory

        return OrchestratorFactory()

    def test_inline_list(self) -> None:
        """Extract inline middleware list from port spec."""
        factory = self._factory()
        port_specs = {
            "llm": {
                "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                "middleware": ["some.Middleware", "other.Middleware"],
            }
        }
        result = factory._extract_middleware_config(port_specs)
        assert result == {"llm": ["some.Middleware", "other.Middleware"]}

    def test_single_string(self) -> None:
        """Single middleware string is wrapped in a list."""
        factory = self._factory()
        port_specs = {
            "llm": {
                "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                "middleware": "some.Middleware",
            }
        }
        result = factory._extract_middleware_config(port_specs)
        assert result == {"llm": ["some.Middleware"]}

    def test_named_stack_reference(self) -> None:
        """Named middleware reference resolves to registered stack."""
        from hexdag.compiler.plugins.middleware_definition import (
            _middleware_registry,
            clear_middleware_registry,
        )

        clear_middleware_registry()
        _middleware_registry["prod-llm"] = ["a.Middleware", "b.Middleware"]

        factory = self._factory()
        port_specs = {
            "llm": {
                "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                "middleware": "prod-llm",
            }
        }
        result = factory._extract_middleware_config(port_specs)
        assert result == {"llm": ["a.Middleware", "b.Middleware"]}

        clear_middleware_registry()

    def test_no_middleware(self) -> None:
        """Port without middleware key returns empty dict."""
        factory = self._factory()
        port_specs = {"llm": {"adapter": "hexdag.stdlib.adapters.mock.MockLLM"}}
        result = factory._extract_middleware_config(port_specs)
        assert result == {}

    def test_mixed_ports(self) -> None:
        """Only ports with middleware declared are included."""
        factory = self._factory()
        port_specs = {
            "llm": {
                "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                "middleware": ["some.Middleware"],
            },
            "memory": {"adapter": "hexdag.stdlib.adapters.mock.MockMemory"},
        }
        result = factory._extract_middleware_config(port_specs)
        assert "llm" in result
        assert "memory" not in result
