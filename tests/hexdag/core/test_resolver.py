"""Tests for the resolver module.

This module tests the simple path-based component resolver that
replaced the registry system.
"""

from __future__ import annotations

import pytest

from hexdag.core.resolver import (
    ResolveError,
    _runtime_components,
    clear_aliases,
    get_registered_aliases,
    get_runtime,
    register_alias,
    register_runtime,
    resolve,
    resolve_function,
    unregister_alias,
)


class TestResolveError:
    """Tests for ResolveError exception."""

    def test_error_message_format(self) -> None:
        """Test that error message includes kind and reason."""
        error = ResolveError("my.module.MyClass", "Module not found")
        assert "my.module.MyClass" in str(error)
        assert "Module not found" in str(error)

    def test_error_attributes(self) -> None:
        """Test that error has kind and reason attributes."""
        error = ResolveError("my.module.MyClass", "Module not found")
        assert error.kind == "my.module.MyClass"
        assert error.reason == "Module not found"


class TestRuntimeComponents:
    """Tests for runtime component registration."""

    def setup_method(self) -> None:
        """Clear runtime components before each test."""
        _runtime_components.clear()

    def teardown_method(self) -> None:
        """Clear runtime components after each test."""
        _runtime_components.clear()

    def test_register_and_get_runtime_component(self) -> None:
        """Test registering and retrieving a runtime component."""

        class MyMacro:
            pass

        register_runtime("my_macro", MyMacro)
        result = get_runtime("my_macro")
        assert result is MyMacro

    def test_get_nonexistent_runtime_component(self) -> None:
        """Test that getting a nonexistent component returns None."""
        result = get_runtime("nonexistent")
        assert result is None

    def test_resolve_runtime_component(self) -> None:
        """Test that resolve() finds runtime components."""

        class MyRuntimeMacro:
            pass

        register_runtime("my_runtime_macro", MyRuntimeMacro)
        result = resolve("my_runtime_macro")
        assert result is MyRuntimeMacro


class TestResolve:
    """Tests for the resolve() function."""

    def setup_method(self) -> None:
        """Clear runtime components before each test."""
        _runtime_components.clear()

    def teardown_method(self) -> None:
        """Clear runtime components after each test."""
        _runtime_components.clear()

    def test_resolve_builtin_node_full_path(self) -> None:
        """Test resolving a builtin node by full module path."""
        from hexdag.builtin.nodes import FunctionNode

        result = resolve("hexdag.builtin.nodes.FunctionNode")
        assert result is FunctionNode

    def test_resolve_builtin_adapter_full_path(self) -> None:
        """Test resolving a builtin adapter by full module path."""
        from hexdag.builtin.adapters.mock import MockLLM

        result = resolve("hexdag.builtin.adapters.mock.MockLLM")
        assert result is MockLLM

    def test_resolve_legacy_short_name(self) -> None:
        """Test resolving using legacy short name (backwards compatibility)."""
        from hexdag.builtin.nodes import FunctionNode

        result = resolve("function_node")
        assert result is FunctionNode

    def test_resolve_legacy_namespace_format(self) -> None:
        """Test resolving using legacy namespace:name format."""
        from hexdag.builtin.nodes import FunctionNode

        result = resolve("core:function_node")
        assert result is FunctionNode

    def test_resolve_all_legacy_short_names(self) -> None:
        """Test that all legacy short names can be resolved."""
        legacy_names = [
            "llm_node",
            "function_node",
            "agent_node",
            "tool_call_node",
        ]
        for name in legacy_names:
            result = resolve(name)
            assert result is not None
            assert isinstance(result, type)

    def test_resolve_nonexistent_module_raises_error(self) -> None:
        """Test that resolving a nonexistent module raises ResolveError."""
        with pytest.raises(ResolveError) as exc_info:
            resolve("nonexistent.module.SomeClass")
        assert "nonexistent.module" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_resolve_nonexistent_class_raises_error(self) -> None:
        """Test that resolving a nonexistent class raises ResolveError."""
        with pytest.raises(ResolveError) as exc_info:
            resolve("hexdag.builtin.nodes.NonExistentClass")
        assert "NonExistentClass" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_resolve_short_name_without_alias_raises_error(self) -> None:
        """Test that a short name without alias raises ResolveError."""
        with pytest.raises(ResolveError) as exc_info:
            resolve("unknown_short_name")
        assert "Must be a full module path" in str(exc_info.value)

    def test_resolve_non_class_raises_error(self) -> None:
        """Test that resolving something that's not a class raises error."""
        # json.loads is a function, not a class
        with pytest.raises(ResolveError) as exc_info:
            resolve("json.loads")
        assert "is not a class" in str(exc_info.value)

    def test_resolve_stdlib_class(self) -> None:
        """Test resolving a standard library class."""
        from collections import OrderedDict

        result = resolve("collections.OrderedDict")
        assert result is OrderedDict

    def test_resolve_error_shows_available_names(self) -> None:
        """Test that error message shows available names from module."""
        with pytest.raises(ResolveError) as exc_info:
            resolve("hexdag.builtin.nodes.WrongClassName")
        error_msg = str(exc_info.value)
        assert "Available:" in error_msg
        # Should show some actual class names from the module
        assert "FunctionNode" in error_msg or "LLMNode" in error_msg


class TestResolveFunction:
    """Tests for the resolve_function() function."""

    def test_resolve_stdlib_function(self) -> None:
        """Test resolving a standard library function."""
        import json

        result = resolve_function("json.loads")
        assert result is json.loads

    def test_resolve_builtin_tool(self) -> None:
        """Test resolving a builtin tool function."""
        from hexdag.core.domain.agent_tools import tool_end

        result = resolve_function("hexdag.core.domain.agent_tools.tool_end")
        assert result is tool_end

    def test_resolve_function_without_dot_raises_error(self) -> None:
        """Test that a path without dots raises ResolveError."""
        with pytest.raises(ResolveError) as exc_info:
            resolve_function("loads")
        assert "Must be a full module path" in str(exc_info.value)

    def test_resolve_nonexistent_module_raises_error(self) -> None:
        """Test that resolving from nonexistent module raises error."""
        with pytest.raises(ResolveError) as exc_info:
            resolve_function("nonexistent.module.func")
        assert "not found" in str(exc_info.value)

    def test_resolve_nonexistent_function_raises_error(self) -> None:
        """Test that resolving nonexistent function raises error."""
        with pytest.raises(ResolveError) as exc_info:
            resolve_function("json.nonexistent_function")
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_resolve_non_callable_raises_error(self) -> None:
        """Test that resolving non-callable raises error."""
        # json.decoder is a module, not callable
        with pytest.raises(ResolveError) as exc_info:
            resolve_function("json.decoder")
        assert "is not callable" in str(exc_info.value)

    def test_resolve_lambda_in_module(self) -> None:
        """Test resolving a callable (like a class) works."""
        # Classes are callable
        import json

        result = resolve_function("json.JSONEncoder")
        assert result is json.JSONEncoder

    def test_resolve_nested_module_function(self) -> None:
        """Test resolving a function from nested module."""
        from os.path import join

        result = resolve_function("os.path.join")
        assert result is join


class TestResolveIntegration:
    """Integration tests for resolver with real hexDAG components."""

    def test_resolve_and_instantiate_node_factory(self) -> None:
        """Test resolving a node factory and using it."""
        FunctionNode = resolve("hexdag.builtin.nodes.FunctionNode")
        factory = FunctionNode()
        # The factory should be callable
        assert callable(factory)

    def test_resolve_and_instantiate_adapter(self) -> None:
        """Test resolving an adapter and instantiating it."""
        MockLLM = resolve("hexdag.builtin.adapters.mock.MockLLM")
        adapter = MockLLM()
        assert adapter is not None

    def test_resolve_function_and_call(self) -> None:
        """Test resolving a function and calling it."""
        loads = resolve_function("json.loads")
        result = loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_resolve_macro_class(self) -> None:
        """Test resolving a macro class."""
        from hexdag.builtin.macros import ReasoningAgentMacro

        result = resolve("hexdag.builtin.macros.ReasoningAgentMacro")
        assert result is ReasoningAgentMacro


class TestUserAliases:
    """Tests for user-defined alias functionality."""

    def setup_method(self) -> None:
        """Clear user aliases before each test."""
        clear_aliases()

    def teardown_method(self) -> None:
        """Clear user aliases after each test."""
        clear_aliases()

    def test_register_and_resolve_alias(self) -> None:
        """Test registering and resolving a user alias."""
        from hexdag.builtin.nodes import FunctionNode

        register_alias("my_func", "hexdag.builtin.nodes.FunctionNode")
        result = resolve("my_func")
        assert result is FunctionNode

    def test_alias_takes_precedence_over_builtin(self) -> None:
        """Test that user alias takes precedence over built-in short names."""
        from hexdag.builtin.nodes import LLMNode

        # Override the built-in "function_node" alias
        register_alias("function_node", "hexdag.builtin.nodes.LLMNode")
        result = resolve("function_node")
        assert result is LLMNode

    def test_unregister_alias(self) -> None:
        """Test unregistering an alias."""
        register_alias("my_alias", "hexdag.builtin.nodes.FunctionNode")
        assert "my_alias" in get_registered_aliases()

        result = unregister_alias("my_alias")
        assert result is True
        assert "my_alias" not in get_registered_aliases()

    def test_unregister_nonexistent_alias(self) -> None:
        """Test that unregistering nonexistent alias returns False."""
        result = unregister_alias("nonexistent")
        assert result is False

    def test_get_registered_aliases(self) -> None:
        """Test getting all registered aliases."""
        register_alias("alias1", "hexdag.builtin.nodes.FunctionNode")
        register_alias("alias2", "hexdag.builtin.nodes.LLMNode")

        aliases = get_registered_aliases()
        assert aliases == {
            "alias1": "hexdag.builtin.nodes.FunctionNode",
            "alias2": "hexdag.builtin.nodes.LLMNode",
        }

    def test_get_registered_aliases_returns_copy(self) -> None:
        """Test that get_registered_aliases returns a copy."""
        register_alias("alias1", "hexdag.builtin.nodes.FunctionNode")
        aliases = get_registered_aliases()

        # Modifying the returned dict shouldn't affect the internal state
        aliases["new_alias"] = "some.path"
        assert "new_alias" not in get_registered_aliases()

    def test_clear_aliases(self) -> None:
        """Test clearing all aliases."""
        register_alias("alias1", "hexdag.builtin.nodes.FunctionNode")
        register_alias("alias2", "hexdag.builtin.nodes.LLMNode")

        clear_aliases()
        assert get_registered_aliases() == {}

    def test_register_alias_empty_alias_raises(self) -> None:
        """Test that empty alias raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_alias("", "hexdag.builtin.nodes.FunctionNode")
        assert "empty" in str(exc_info.value).lower()

    def test_register_alias_empty_path_raises(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            register_alias("my_alias", "")
        assert "empty" in str(exc_info.value).lower()

    def test_resolve_alias_invalid_path_raises(self) -> None:
        """Test that alias pointing to invalid path raises error."""
        register_alias("bad_alias", "nonexistent.module.Class")
        with pytest.raises(ResolveError) as exc_info:
            resolve("bad_alias")
        assert "not found" in str(exc_info.value)

    def test_multiple_aliases_same_target(self) -> None:
        """Test multiple aliases pointing to same target."""
        from hexdag.builtin.nodes import FunctionNode

        register_alias("fn", "hexdag.builtin.nodes.FunctionNode")
        register_alias("func", "hexdag.builtin.nodes.FunctionNode")
        register_alias("function", "hexdag.builtin.nodes.FunctionNode")

        assert resolve("fn") is FunctionNode
        assert resolve("func") is FunctionNode
        assert resolve("function") is FunctionNode


class TestYamlAliasesIntegration:
    """Integration tests for aliases defined in YAML pipelines."""

    def setup_method(self) -> None:
        """Clear user aliases before each test."""
        clear_aliases()

    def teardown_method(self) -> None:
        """Clear user aliases after each test."""
        clear_aliases()

    def test_yaml_pipeline_with_aliases(self) -> None:
        """Test that aliases in YAML spec are registered."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-with-aliases
spec:
  aliases:
    fn: hexdag.builtin.nodes.FunctionNode
  nodes:
    - kind: fn
      metadata:
        name: my_node
      spec:
        fn: json.loads
      dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert "my_node" in graph.nodes
        # Alias should now be registered
        assert "fn" in get_registered_aliases()

    def test_yaml_pipeline_multiple_aliases(self) -> None:
        """Test multiple aliases in YAML."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-multiple-aliases
spec:
  aliases:
    fn: hexdag.builtin.nodes.FunctionNode
    llm: hexdag.builtin.nodes.LLMNode
  nodes:
    - kind: fn
      metadata:
        name: processor
      spec:
        fn: json.loads
      dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert "fn" in get_registered_aliases()
        assert "llm" in get_registered_aliases()

    def test_yaml_pipeline_without_aliases(self) -> None:
        """Test that pipeline without aliases section works."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-no-aliases
spec:
  nodes:
    - kind: hexdag.builtin.nodes.FunctionNode
      metadata:
        name: my_node
      spec:
        fn: json.loads
      dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert "my_node" in graph.nodes
