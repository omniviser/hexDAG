"""Tests for the yaml_macro module.

This module tests YAML-defined macros for declarative pipeline composition,
including integration tests for macro_invocation through the pipeline builder.
"""

from __future__ import annotations

import textwrap

import pytest

from hexdag.kernel.exceptions import YamlPipelineBuilderError
from hexdag.kernel.yaml_macro import (
    PortRequirement,
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
        with pytest.raises(YamlPipelineBuilderError, match="Invalid type"):
            YamlMacroParameterSpec(name="test", type="invalid_type")

    def test_union_type(self) -> None:
        """Test union type validation."""
        param = YamlMacroParameterSpec(name="value", type="str | int")
        assert param.type == "str | int"

    def test_invalid_union_type_raises_error(self) -> None:
        """Test invalid union type."""
        with pytest.raises(YamlPipelineBuilderError, match="Invalid type"):
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
        with pytest.raises(
            YamlPipelineBuilderError, match="Required parameter 'api_key' not provided"
        ):
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
        with pytest.raises(YamlPipelineBuilderError, match="must be one of"):
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

    def test_macro_nested_macro_invocation_allowed(self) -> None:
        """Test that nested macro_invocation is allowed (expanded at build time)."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: inner_macro
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_inner"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: outer_macro
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_nested"
                spec:
                  macro: inner_macro
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: test-pipeline
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: top
                  spec:
                    macro: outer_macro
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        # The inner macro should have expanded: top_nested_inner
        assert "top_nested_inner" in graph.nodes

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


# ============================================================================
# Integration tests: macro_invocation through the full pipeline builder
# ============================================================================


class TestYamlMacroBuilderIntegration:
    """Test YAML macros through the full YamlPipelineBuilder pipeline."""

    def test_macro_definition_and_invocation_roundtrip(self) -> None:
        """Multi-doc YAML: define a macro, then invoke it in a pipeline."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: greet_macro
            parameters:
              - name: greeting
                type: str
                default: hello
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_say"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: test-pipeline
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: greeter
                  spec:
                    macro: greet_macro
                    inputs:
                      greeting: hi
        """)

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        assert "greeter_say" in graph.nodes

    def test_macro_followed_by_node_gets_correct_dependency(self) -> None:
        """Node after macro_invocation should depend on macro's exit node."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: two_step
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_step1"
                spec:
                  fn: "json.dumps"
              - kind: function_node
                metadata:
                  name: "{{name}}_step2"
                spec:
                  fn: "json.loads"
                dependencies:
                  - "{{name}}_step1"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: test-pipeline
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: workflow
                  spec:
                    macro: two_step
                - kind: function_node
                  metadata:
                    name: final_step
                  spec:
                    fn: "json.dumps"
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        assert "workflow_step1" in graph.nodes
        assert "workflow_step2" in graph.nodes
        assert "final_step" in graph.nodes
        # final_step should depend on the macro's exit node, not "workflow"
        final_deps = graph.get_dependencies("final_step")
        assert "workflow_step2" in final_deps
        assert "workflow" not in final_deps

    def test_config_and_inputs_overlap_raises_error(self) -> None:
        """Overlapping keys in config and inputs should raise."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: overlap_macro
            parameters:
              - name: model
                type: str
                default: gpt-4
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_node"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: test-pipeline
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: overlapper
                  spec:
                    macro: overlap_macro
                    config:
                      model: gpt-4
                    inputs:
                      model: gpt-3.5
        """)

        builder = YamlPipelineBuilder()
        with pytest.raises(YamlPipelineBuilderError, match="both 'config' and 'inputs'"):
            builder.build_from_yaml_string(yaml_content)

    def test_type_validation_rejects_wrong_type(self) -> None:
        """Passing wrong type for a macro parameter should raise."""
        macro = YamlMacro(
            macro_name="typed",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", required=True),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        with pytest.raises(YamlPipelineBuilderError, match="expected type 'int'"):
            macro.expand(
                instance_name="test",
                inputs={"count": "not_an_int"},
                dependencies=[],
            )

    def test_type_validation_accepts_correct_type(self) -> None:
        """Correct type should pass validation."""
        macro = YamlMacro(
            macro_name="typed",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", required=True),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        graph = macro.expand(
            instance_name="test",
            inputs={"count": 5},
            dependencies=[],
        )
        assert "test_node" in graph.nodes

    def test_type_validation_union_types(self) -> None:
        """Union type should accept either type."""
        macro = YamlMacro(
            macro_name="union",
            parameters=[
                YamlMacroParameterSpec(name="val", type="str | int"),
            ],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        # String should work
        graph = macro.expand(instance_name="t1", inputs={"val": "hello"}, dependencies=[])
        assert "t1_node" in graph.nodes

        # Int should work
        graph = macro.expand(instance_name="t2", inputs={"val": 42}, dependencies=[])
        assert "t2_node" in graph.nodes

        # List should fail
        with pytest.raises(YamlPipelineBuilderError, match="expected type"):
            macro.expand(instance_name="t3", inputs={"val": [1, 2]}, dependencies=[])

    def test_default_none_is_passed_to_context(self) -> None:
        """Parameter with explicit default=None should be in template context."""
        param = YamlMacroParameterSpec(name="opt", type="Any", default=None)
        assert param._has_explicit_default is True  # noqa: SLF001

        macro = YamlMacro(
            macro_name="none_default",
            parameters=[param],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_node"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
        )
        # Should NOT raise — None default is applied, not treated as missing
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])
        assert "test_node" in graph.nodes

    def test_no_explicit_default_is_not_in_context(self) -> None:
        """Parameter with no default and not required should not error."""
        param = YamlMacroParameterSpec(name="opt", type="str")
        assert param._has_explicit_default is False  # noqa: SLF001

    def test_macro_with_external_dependencies(self) -> None:
        """Macro entry nodes should get external dependencies."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: dep_macro
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: test-pipeline
            spec:
              nodes:
                - kind: function_node
                  metadata:
                    name: setup
                  spec:
                    fn: "json.dumps"
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: dep_macro
                    dependencies:
                      - setup
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        assert "setup" in graph.nodes
        assert "worker_work" in graph.nodes
        worker_deps = graph.get_dependencies("worker_work")
        assert "setup" in worker_deps

    def test_nested_macro_three_levels_deep(self) -> None:
        """Three levels of macro nesting: pipeline → outer → middle → inner."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: level3_macro
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_leaf"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: level2_macro
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_mid"
                spec:
                  macro: level3_macro
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: level1_macro
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_top"
                spec:
                  macro: level2_macro
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: deep-nesting
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: root
                  spec:
                    macro: level1_macro
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        # Expansion: root → root_top → root_top_mid → root_top_mid_leaf
        assert "root_top_mid_leaf" in graph.nodes

    def test_nested_macro_with_parameters_pass_through(self) -> None:
        """Parameters pass through nested macro expansion correctly."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: inner_param_macro
            parameters:
              - name: label
                type: str
                default: default_label
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_{{label}}"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: outer_param_macro
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_child"
                spec:
                  macro: inner_param_macro
                  inputs:
                    label: custom
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: param-pass
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: wrapper
                  spec:
                    macro: outer_param_macro
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        # inner macro gets label="custom"
        assert "wrapper_child_custom" in graph.nodes

    def test_circular_macro_raises_error(self) -> None:
        """Circular macro reference (A→B→A) should raise."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: macro_a
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_b"
                spec:
                  macro: macro_b
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: macro_b
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_a"
                spec:
                  macro: macro_a
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: circular-test
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: start
                  spec:
                    macro: macro_a
        """)

        builder = YamlPipelineBuilder()
        with pytest.raises(YamlPipelineBuilderError, match="Circular"):
            builder.build_from_yaml_string(yaml_content)

    def test_nested_macro_sequential_dependency(self) -> None:
        """Node after nested macro gets correct dependency on exit node."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: inner_seq
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_do"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: outer_seq
            nodes:
              - kind: macro_invocation
                metadata:
                  name: "{{name}}_inner"
                spec:
                  macro: inner_seq
              - kind: function_node
                metadata:
                  name: "{{name}}_after"
                spec:
                  fn: "json.loads"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: seq-test
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: flow
                  spec:
                    macro: outer_seq
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        assert "flow_inner_do" in graph.nodes
        assert "flow_after" in graph.nodes


# ============================================================================
# Phase 3: Dynamic node generation via nodes_raw
# ============================================================================


class TestYamlMacroDynamicNodes:
    """Tests for dynamic node generation using nodes_raw with Jinja2 block tags."""

    def test_for_loop_generates_n_nodes(self) -> None:
        """Macro with {% for %} generates N function_nodes."""
        macro = YamlMacro(
            macro_name="dynamic_for",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", default=3),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                {% for i in range(count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_step_{{i}}"
                  spec:
                    fn: "json.dumps"
                {% endfor %}
            """),
        )
        graph = macro.expand(
            instance_name="pipeline",
            inputs={"count": 4},
            dependencies=[],
        )

        assert len(graph) == 4
        for i in range(4):
            assert f"pipeline_step_{i}" in graph.nodes

    def test_for_loop_with_default_count(self) -> None:
        """Macro with {% for %} uses default count when not overridden."""
        macro = YamlMacro(
            macro_name="dynamic_default",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", default=2),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                {% for i in range(count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_node_{{i}}"
                  spec:
                    fn: "json.dumps"
                {% endfor %}
            """),
        )
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])

        assert len(graph) == 2
        assert "test_node_0" in graph.nodes
        assert "test_node_1" in graph.nodes

    def test_if_conditional_includes_nodes(self) -> None:
        """Macro with {% if %} conditionally includes nodes."""
        nodes_raw = textwrap.dedent("""\
            - kind: function_node
              metadata:
                name: "{{name}}_always"
              spec:
                fn: "json.dumps"
            {% if include_validator %}
            - kind: function_node
              metadata:
                name: "{{name}}_validator"
              spec:
                fn: "json.loads"
                dependencies:
                  - "{{name}}_always"
            {% endif %}
        """)

        # With validator
        macro = YamlMacro(
            macro_name="conditional",
            parameters=[
                YamlMacroParameterSpec(name="include_validator", type="bool", default=True),
            ],
            nodes=[],
            nodes_raw=nodes_raw,
        )
        graph = macro.expand(
            instance_name="test",
            inputs={"include_validator": True},
            dependencies=[],
        )
        assert "test_always" in graph.nodes
        assert "test_validator" in graph.nodes

        # Without validator
        graph2 = macro.expand(
            instance_name="test2",
            inputs={"include_validator": False},
            dependencies=[],
        )
        assert "test2_always" in graph2.nodes
        assert "test2_validator" not in graph2.nodes

    def test_for_loop_with_dependencies_between_generated_nodes(self) -> None:
        """Dynamically generated nodes can have dependencies on each other."""
        macro = YamlMacro(
            macro_name="chained",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", default=3),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                {% for i in range(count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_step_{{i}}"
                  spec:
                    fn: "json.dumps"
                {% if i > 0 %}
                    dependencies:
                      - "{{name}}_step_{{i - 1}}"
                {% endif %}
                {% endfor %}
            """),
        )
        graph = macro.expand(
            instance_name="chain",
            inputs={"count": 3},
            dependencies=[],
        )

        assert len(graph) == 3
        # step_1 depends on step_0, step_2 depends on step_1
        deps_1 = graph.get_dependencies("chain_step_1")
        assert "chain_step_0" in deps_1
        deps_2 = graph.get_dependencies("chain_step_2")
        assert "chain_step_1" in deps_2

    def test_nodes_raw_through_pipeline_builder(self) -> None:
        """Full roundtrip: kind: Macro with nodes_raw + Pipeline invocation."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: dynamic_macro
            parameters:
              - name: count
                type: int
                default: 2
            nodes_raw: |
              {% for i in range(count) %}
              - kind: function_node
                metadata:
                  name: "{{name}}_worker_{{i}}"
                spec:
                  fn: "json.dumps"
              {% endfor %}
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: dynamic-test
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: batch
                  spec:
                    macro: dynamic_macro
                    inputs:
                      count: 3
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        assert len(graph) == 3
        for i in range(3):
            assert f"batch_worker_{i}" in graph.nodes

    def test_nodes_raw_with_if_else_through_builder(self) -> None:
        """Conditional node generation through pipeline builder."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: mode_macro
            parameters:
              - name: mode
                type: str
                default: simple
            nodes_raw: |
              - kind: function_node
                metadata:
                  name: "{{name}}_start"
                spec:
                  fn: "json.dumps"
              {% if mode == 'detailed' %}
              - kind: function_node
                metadata:
                  name: "{{name}}_analyze"
                spec:
                  fn: "json.loads"
                  dependencies:
                    - "{{name}}_start"
              - kind: function_node
                metadata:
                  name: "{{name}}_report"
                spec:
                  fn: "json.dumps"
                  dependencies:
                    - "{{name}}_analyze"
              {% endif %}
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: mode-test
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: work
                  spec:
                    macro: mode_macro
                    inputs:
                      mode: detailed
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)

        assert "work_start" in graph.nodes
        assert "work_analyze" in graph.nodes
        assert "work_report" in graph.nodes
        assert "work_start" in graph.get_dependencies("work_analyze")

    def test_nodes_raw_preserves_runtime_variables(self) -> None:
        """Runtime variables ({{other_node.output}}) are preserved in nodes_raw."""
        macro = YamlMacro(
            macro_name="runtime_ref",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", default=1),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                {% for i in range(count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_node_{{i}}"
                  spec:
                    fn: "json.dumps"
                {% endfor %}
            """),
        )
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])
        assert "test_node_0" in graph.nodes

    def test_nodes_raw_invalid_yaml_after_render_raises(self) -> None:
        """Invalid YAML after rendering raises clear error."""
        macro = YamlMacro(
            macro_name="bad_yaml",
            parameters=[
                YamlMacroParameterSpec(name="count", type="int", default=1),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                {% for i in range(count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_node"
                  spec:
                  invalid yaml here: [[[
                {% endfor %}
            """),
        )
        with pytest.raises(YamlPipelineBuilderError, match="Failed to parse rendered YAML"):
            macro.expand(instance_name="test", inputs={}, dependencies=[])

    def test_nodes_raw_not_a_list_raises(self) -> None:
        """nodes_raw that renders to non-list YAML raises error."""
        macro = YamlMacro(
            macro_name="not_list",
            parameters=[
                YamlMacroParameterSpec(name="x", type="int", default=1),
            ],
            nodes=[],
            nodes_raw="{% if x %}key: value{% endif %}\n",
        )
        with pytest.raises(YamlPipelineBuilderError, match="must be a YAML list"):
            macro.expand(instance_name="test", inputs={}, dependencies=[])

    def test_nodes_raw_without_block_tags_uses_post_parse(self) -> None:
        """nodes_raw without {%...%} falls back to regular post-parse path."""
        # When nodes_raw has no block tags, the nodes list is used instead
        macro = YamlMacro(
            macro_name="no_blocks",
            parameters=[],
            nodes=[
                {
                    "kind": "function_node",
                    "metadata": {"name": "{{name}}_fallback"},
                    "spec": {"fn": "json.dumps"},
                }
            ],
            nodes_raw="- kind: function_node\n  metadata:\n    name: no_block\n",
        )
        graph = macro.expand(instance_name="test", inputs={}, dependencies=[])
        # Should use the nodes list (post-parse path) since nodes_raw has no {%
        assert "test_fallback" in graph.nodes

    def test_mixed_static_and_dynamic_nodes(self) -> None:
        """nodes_raw can contain both static and dynamically generated nodes."""
        macro = YamlMacro(
            macro_name="mixed",
            parameters=[
                YamlMacroParameterSpec(name="extra_count", type="int", default=2),
            ],
            nodes=[],
            nodes_raw=textwrap.dedent("""\
                - kind: function_node
                  metadata:
                    name: "{{name}}_setup"
                  spec:
                    fn: "json.dumps"
                {% for i in range(extra_count) %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_worker_{{i}}"
                  spec:
                    fn: "json.loads"
                    dependencies:
                      - "{{name}}_setup"
                {% endfor %}
                - kind: function_node
                  metadata:
                    name: "{{name}}_aggregate"
                  spec:
                    fn: "json.dumps"
            """),
        )
        graph = macro.expand(
            instance_name="job",
            inputs={"extra_count": 3},
            dependencies=[],
        )

        assert "job_setup" in graph.nodes
        for i in range(3):
            assert f"job_worker_{i}" in graph.nodes
            assert "job_setup" in graph.get_dependencies(f"job_worker_{i}")
        assert "job_aggregate" in graph.nodes


# ============================================================================
# Phase 4: Port requirements
# ============================================================================


class TestPortRequirement:
    """Tests for PortRequirement model."""

    def test_basic_port_requirement(self) -> None:
        """Test creating a basic port requirement."""
        req = PortRequirement(name="llm", protocol="LLM")
        assert req.name == "llm"
        assert req.protocol == "LLM"
        assert req.optional is False

    def test_optional_port_requirement(self) -> None:
        """Test creating an optional port requirement."""
        req = PortRequirement(name="memory", protocol="SupportsKeyValue", optional=True)
        assert req.optional is True


class TestYamlMacroPortRequirements:
    """Tests for port requirement validation through the pipeline builder."""

    def test_required_port_present_succeeds(self) -> None:
        """Macro with required port + Pipeline that provides it → success."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: port_macro_ok
            requires_ports:
              - name: llm
                protocol: LLM
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: port-test-ok
            spec:
              ports:
                llm:
                  adapter: hexdag.stdlib.adapters.mock.MockLLM
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: port_macro_ok
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)
        assert "worker_work" in graph.nodes

    def test_required_port_missing_raises(self) -> None:
        """Macro with required port + Pipeline missing it → build-time error."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: port_macro_missing
            requires_ports:
              - name: llm
                protocol: LLM
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: port-test-missing
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: port_macro_missing
        """)

        builder = YamlPipelineBuilder()
        with pytest.raises(YamlPipelineBuilderError, match="requires port 'llm'"):
            builder.build_from_yaml_string(yaml_content)

    def test_optional_port_missing_succeeds(self) -> None:
        """Macro with optional port + Pipeline missing it → success."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: port_macro_optional
            requires_ports:
              - name: memory
                protocol: SupportsKeyValue
                optional: true
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: port-test-optional
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: port_macro_optional
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)
        assert "worker_work" in graph.nodes

    def test_multiple_port_requirements(self) -> None:
        """Macro with multiple required ports — one missing raises."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: port_macro_multi
            requires_ports:
              - name: llm
                protocol: LLM
              - name: store
                protocol: SupportsKeyValue
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: port-test-multi
            spec:
              ports:
                llm:
                  adapter: hexdag.stdlib.adapters.mock.MockLLM
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: port_macro_multi
        """)

        builder = YamlPipelineBuilder()
        with pytest.raises(YamlPipelineBuilderError, match="requires port 'store'"):
            builder.build_from_yaml_string(yaml_content)

    def test_no_port_requirements_always_succeeds(self) -> None:
        """Macro without requires_ports — always succeeds."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = textwrap.dedent("""\
            ---
            apiVersion: hexdag/v1
            kind: Macro
            metadata:
              name: port_macro_none
            nodes:
              - kind: function_node
                metadata:
                  name: "{{name}}_work"
                spec:
                  fn: "json.dumps"
            ---
            apiVersion: hexdag/v1
            kind: Pipeline
            metadata:
              name: port-test-none
            spec:
              nodes:
                - kind: macro_invocation
                  metadata:
                    name: worker
                  spec:
                    macro: port_macro_none
        """)

        builder = YamlPipelineBuilder()
        graph, _config = builder.build_from_yaml_string(yaml_content)
        assert "worker_work" in graph.nodes
