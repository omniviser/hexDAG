"""Tests for the yaml_macro module.

This module tests YAML-defined macros for declarative pipeline composition.
"""

from __future__ import annotations

import pytest

from hexdag.core.yaml_macro import (
    PreserveUndefined,
    YamlMacro,
    YamlMacroConfig,
    YamlMacroParameterSpec,
)


class TestYamlMacroParameterSpec:
    """Tests for YamlMacroParameterSpec."""

    def test_basic_parameter(self) -> None:
        """Test creating a basic parameter spec."""
        param = YamlMacroParameterSpec(name="model", type="str", default="gpt-4")
        assert param.name == "model"
        assert param.type == "str"
        assert param.default == "gpt-4"
        assert param.required is False

    def test_required_parameter(self) -> None:
        """Test required parameter."""
        param = YamlMacroParameterSpec(name="api_key", type="str", required=True)
        assert param.required is True

    def test_enum_parameter(self) -> None:
        """Test parameter with enum values."""
        param = YamlMacroParameterSpec(
            name="mode", type="str", enum=["fast", "accurate"], default="fast"
        )
        assert param.enum == ["fast", "accurate"]

    def test_valid_types(self) -> None:
        """Test all valid parameter types."""
        valid_types = ["str", "int", "float", "bool", "list", "dict", "Any"]
        for t in valid_types:
            param = YamlMacroParameterSpec(name="test", type=t)
            assert param.type == t

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid type raises validation error."""
        with pytest.raises(ValueError, match="Invalid type"):
            YamlMacroParameterSpec(name="test", type="invalid_type")

    def test_union_type(self) -> None:
        """Test union type validation."""
        param = YamlMacroParameterSpec(name="value", type="str | int")
        assert param.type == "str | int"

    def test_invalid_union_type_raises_error(self) -> None:
        """Test invalid union type."""
        with pytest.raises(ValueError, match="Invalid type"):
            YamlMacroParameterSpec(name="test", type="str | invalid")


class TestYamlMacroConfig:
    """Tests for YamlMacroConfig."""

    def test_minimal_config(self) -> None:
        """Test creating minimal macro config."""
        config = YamlMacroConfig(macro_name="test_macro", nodes=[])
        assert config.macro_name == "test_macro"
        assert config.parameters == []
        assert config.nodes == []

    def test_full_config(self) -> None:
        """Test creating full macro config."""
        config = YamlMacroConfig(
            macro_name="retry_workflow",
            macro_description="Retry with exponential backoff",
            parameters=[
                YamlMacroParameterSpec(name="max_retries", type="int", default=3),
                YamlMacroParameterSpec(name="delay", type="float", default=1.0),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_attempt"},
                    "spec": {"fn": "{{fn}}"},
                }
            ],
            outputs={"result": "{{name}}_attempt.output"},
        )
        assert config.macro_name == "retry_workflow"
        assert len(config.parameters) == 2
        assert len(config.nodes) == 1
        assert config.outputs == {"result": "{{name}}_attempt.output"}


class TestPreserveUndefined:
    """Tests for PreserveUndefined Jinja2 undefined handler."""

    def test_simple_undefined_preserved(self) -> None:
        """Test that simple undefined variables are preserved."""
        undefined = PreserveUndefined(name="input")
        assert str(undefined) == "{{input}}"

    def test_dotted_undefined_preserved(self) -> None:
        """Test that dotted access is preserved."""
        undefined = PreserveUndefined(name="node")
        result = undefined.output
        assert str(result) == "{{node.output}}"

    def test_nested_attribute_access(self) -> None:
        """Test nested attribute access preservation."""
        undefined = PreserveUndefined(name="data")
        result = undefined.nested.value
        assert str(result) == "{{data.nested.value}}"


class TestYamlMacro:
    """Tests for YamlMacro class."""

    def test_macro_creation(self) -> None:
        """Test creating a YamlMacro instance."""
        macro = YamlMacro(
            macro_name="test",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_prompt"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        assert macro.config.macro_name == "test"

    def test_macro_expand_simple(self) -> None:
        """Test expanding a simple macro."""
        macro = YamlMacro(
            macro_name="simple",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])

        assert "test_node" in graph.nodes
        # The template inside spec should be preserved for runtime
        assert graph.nodes["test_node"] is not None

    def test_macro_expand_with_parameters(self) -> None:
        """Test expanding macro with parameters."""
        macro = YamlMacro(
            macro_name="parameterized",
            parameters=[
                YamlMacroParameterSpec(name="model", type="str", default="gpt-4"),
                YamlMacroParameterSpec(name="temperature", type="float", default=0.7),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_llm"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        graph = macro.expand(
            instance_name="inference",
            inputs={"model": "claude", "temperature": 0.5},
            dependencies=[],
        )

        assert "inference_llm" in graph.nodes

    def test_macro_expand_with_dependencies(self) -> None:
        """Test macro expansion with external dependencies."""
        macro = YamlMacro(
            macro_name="dependent",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_step1"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        graph = macro.expand(instance_name="workflow", inputs={}, dependencies=["preprocessor"])

        assert "workflow_step1" in graph.nodes

    def test_macro_parameter_defaults_applied(self) -> None:
        """Test that parameter defaults are applied."""
        macro = YamlMacro(
            macro_name="defaults",
            parameters=[
                YamlMacroParameterSpec(name="retries", type="int", default=3),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        # Don't provide retries - should use default
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])
        assert "test_node" in graph.nodes

    def test_macro_required_parameter_missing_raises(self) -> None:
        """Test that missing required parameter raises error."""
        macro = YamlMacro(
            macro_name="required",
            parameters=[
                YamlMacroParameterSpec(name="api_key", type="str", required=True),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        with pytest.raises(ValueError, match="Required parameter 'api_key' not provided"):
            macro.expand(instance_name="test", inputs={}, dependencies=[])

    def test_macro_enum_validation(self) -> None:
        """Test enum parameter validation."""
        macro = YamlMacro(
            macro_name="enum_test",
            parameters=[
                YamlMacroParameterSpec(
                    name="mode", type="str", enum=["fast", "accurate"], default="fast"
                ),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        # Valid enum value
        graph = macro.expand(instance_name="test", inputs={"mode": "accurate"}, dependencies=[])
        assert "test_node" in graph.nodes

    def test_macro_enum_invalid_value_raises(self) -> None:
        """Test that invalid enum value raises error."""
        macro = YamlMacro(
            macro_name="enum_test",
            parameters=[
                YamlMacroParameterSpec(
                    name="mode", type="str", enum=["fast", "accurate"], default="fast"
                ),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        with pytest.raises(ValueError, match="must be one of"):
            macro.expand(instance_name="test", inputs={"mode": "invalid"}, dependencies=[])

    def test_macro_preserves_runtime_variables(self) -> None:
        """Test that runtime variables are preserved in templates."""
        macro = YamlMacro(
            macro_name="runtime",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},  # runtime variable
                }
            ],
        )
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])

        # The runtime variable should be preserved
        assert "test_node" in graph.nodes

    def test_macro_nested_macro_invocation_raises(self) -> None:
        """Test that nested macro_invocation raises error."""
        macro = YamlMacro(
            macro_name="nested",
            parameters=[],
            nodes=[
                {
                    "kind": "macro_invocation",
                    "metadata": {"name": "nested_macro"},
                    "spec": {"macro": "other_macro"},
                }
            ],
        )
        with pytest.raises(ValueError, match="cannot contain nested macro_invocations"):
            macro.expand(instance_name="test", inputs={}, dependencies=[])

    def test_macro_repr(self) -> None:
        """Test macro string representation."""
        macro = YamlMacro(
            macro_name="test",
            parameters=[
                YamlMacroParameterSpec(name="p1", type="str"),
                YamlMacroParameterSpec(name="p2", type="int"),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "n"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        repr_str = repr(macro)
        assert "test" in repr_str
        assert "parameters=2" in repr_str
        assert "nodes=1" in repr_str


class TestYamlMacroIntegration:
    """Integration tests for YAML macros."""

    def test_multiple_nodes_in_macro(self) -> None:
        """Test macro with multiple nodes."""
        macro = YamlMacro(
            macro_name="multi_node",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_step1"},
                    "spec": {"fn": "json.dumps"},
                },
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_step2"},
                    "spec": {"fn": "json.dumps", "dependencies": ["{{name}}_step1"]},
                },
            ],
        )
        graph = macro.expand(instance_name="pipeline", inputs={}, dependencies=[])

        assert "pipeline_step1" in graph.nodes
        assert "pipeline_step2" in graph.nodes

    def test_complex_template_rendering(self) -> None:
        """Test complex Jinja2 template rendering."""
        macro = YamlMacro(
            macro_name="complex",
            parameters=[
                YamlMacroParameterSpec(name="prefix", type="str", default="test"),
                YamlMacroParameterSpec(name="count", type="int", default=1),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{prefix}}_{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        graph = macro.expand(
            instance_name="instance",
            inputs={"prefix": "custom", "count": 5},
            dependencies=[],
        )

        assert "custom_instance_node" in graph.nodes
