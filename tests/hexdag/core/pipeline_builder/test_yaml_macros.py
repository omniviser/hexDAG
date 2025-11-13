"""Tests for YAML-defined macros (kind: Macro)."""

import pytest

from hexdag.core.pipeline_builder.yaml_builder import (
    YamlPipelineBuilder,
    YamlPipelineBuilderError,
)
from hexdag.core.registry import registry
from hexdag.core.yaml_macro import YamlMacro


class TestYamlMacroDefinition:
    """Tests for YAML macro definition (kind: Macro)."""

    def test_simple_macro_definition(self):
        """Test basic macro definition with parameters."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: simple_retry
  description: Simple retry pattern
  namespace: test
parameters:
  - name: max_retries
    type: int
    default: 3
  - name: delay
    type: float
    default: 1.0
nodes:
  - kind: function_node
    metadata:
      name: "{{name}}_attempt"
    spec:
      fn: "builtins.print"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Macro should be registered
        macro = registry.get("simple_retry", namespace="test")
        assert isinstance(macro, YamlMacro)
        assert macro.config.macro_name == "simple_retry"
        assert len(macro.config.parameters) == 2
        assert len(macro.config.nodes) == 1

    def test_macro_with_required_parameters(self):
        """Test macro definition with required parameters."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: required_params_macro
  namespace: test
parameters:
  - name: fn
    type: str
    required: true
  - name: timeout
    type: int
    default: 30
nodes:
  - kind: function_node
    metadata:
      name: "{{name}}_exec"
    spec:
      fn: "{{fn}}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes: []
"""
        builder = YamlPipelineBuilder()
        builder.build_from_yaml_string(yaml_content, use_cache=False)

        macro = registry.get("required_params_macro", namespace="test")
        assert macro.config.parameters[0].required is True
        assert macro.config.parameters[1].required is False

    def test_macro_with_enum_parameters(self):
        """Test macro definition with enum validation."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: enum_macro
  namespace: test
parameters:
  - name: mode
    type: str
    enum: [human, auto, hybrid]
    default: human
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_prompt"
    spec:
      template: "Mode: {{mode}}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes: []
"""
        builder = YamlPipelineBuilder()
        builder.build_from_yaml_string(yaml_content, use_cache=False)

        macro = registry.get("enum_macro", namespace="test")
        assert macro.config.parameters[0].enum == ["human", "auto", "hybrid"]

    def test_macro_definition_missing_name(self):
        """Test error when macro definition lacks name."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  description: Missing name
nodes:
  - kind: prompt_node
    metadata:
      name: test
    spec:
      template: "test"
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(YamlPipelineBuilderError, match="missing 'metadata.name'"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_macro_definition_no_nodes(self):
        """Test error when macro has no nodes."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: empty_macro
  namespace: test
parameters: []
nodes: []
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(YamlPipelineBuilderError, match="has no nodes"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)


class TestYamlMacroExpansion:
    """Tests for YAML macro expansion (invocation)."""

    @classmethod
    def setup_class(cls):
        """Register test macro before running tests."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: linear_chain
  namespace: test
  description: Linear chain of nodes
parameters:
  - name: steps
    type: int
    default: 2
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_step0"
    spec:
      template: "Step 0"
  - kind: prompt_node
    metadata:
      name: "{{name}}_step1"
    spec:
      template: "Step 1"
      dependencies: ["{{name}}_step0"]
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: setup-pipeline
spec:
  nodes: []
"""
        builder = YamlPipelineBuilder()
        builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_basic_macro_expansion(self):
        """Test invoking a YAML macro."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: my_chain
      spec:
        macro: test:linear_chain
        config:
          steps: 2
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 2 nodes from macro expansion
        assert len(graph.nodes) == 2
        assert "my_chain_step0" in graph.nodes
        assert "my_chain_step1" in graph.nodes

        # Check dependencies
        assert graph.get_dependencies("my_chain_step1") == frozenset({"my_chain_step0"})

    def test_macro_expansion_with_external_dependencies(self):
        """Test macro with external node dependencies."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: input
      spec:
        template: "Input"
    - kind: macro_invocation
      metadata:
        name: processor
      spec:
        macro: test:linear_chain
        config:
          steps: 2
        inputs: {}
        dependencies: [input]
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 3 nodes (1 regular + 2 from macro)
        assert len(graph.nodes) == 3
        assert "input" in graph.nodes
        assert "processor_step0" in graph.nodes
        assert "processor_step1" in graph.nodes

        # First macro node should depend on input
        assert "input" in graph.get_dependencies("processor_step0")

    def test_macro_parameter_validation_required(self):
        """Test error when required parameter is missing."""
        # Define macro with required parameter
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: required_macro
  namespace: test
parameters:
  - name: fn
    type: str
    required: true
nodes:
  - kind: function_node
    metadata:
      name: "{{name}}_exec"
    spec:
      fn: "{{fn}}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: bad_invocation
      spec:
        macro: test:required_macro
        config: {}
        inputs: {}
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(ValueError, match="Required parameter"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_macro_parameter_validation_enum(self):
        """Test enum parameter validation."""
        # Define macro with enum
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: enum_test_macro
  namespace: test
parameters:
  - name: mode
    type: str
    enum: [a, b, c]
    default: a
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_node"
    spec:
      template: "{{mode}}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: test_instance
      spec:
        macro: test:enum_test_macro
        config:
          mode: invalid_value
        inputs: {}
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(ValueError, match="must be one of"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_macro_jinja2_rendering(self):
        """Test Jinja2 template rendering in macro nodes."""
        # Define macro with complex templating
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: template_macro
  namespace: test
parameters:
  - name: prefix
    type: str
    default: "test"
  - name: count
    type: int
    default: 1
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_{{prefix}}_{{count}}"
    spec:
      template: "Prefix={{prefix}}, Count={{count}}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: tmpl
      spec:
        macro: test:template_macro
        config:
          prefix: "hello"
          count: 42
        inputs: {}
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Node name should be rendered
        assert "tmpl_hello_42" in graph.nodes

    def test_multiple_yaml_macros_same_pipeline(self):
        """Test multiple YAML macro invocations in same pipeline."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: chain1
      spec:
        macro: test:linear_chain
        config:
          steps: 2
        inputs: {}
        dependencies: []
    - kind: macro_invocation
      metadata:
        name: chain2
      spec:
        macro: test:linear_chain
        config:
          steps: 2
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 4 nodes total (2 from each macro)
        assert len(graph.nodes) == 4
        assert "chain1_step0" in graph.nodes
        assert "chain1_step1" in graph.nodes
        assert "chain2_step0" in graph.nodes
        assert "chain2_step1" in graph.nodes


class TestYamlMacroInclude:
    """Tests for including YAML macros from external files."""

    def test_macro_definition_via_include(self, tmp_path):
        """Test loading macro definition via !include."""
        # Create macro definition file
        macro_file = tmp_path / "my_macro.yaml"
        macro_file.write_text("""
kind: Macro
metadata:
  name: included_macro
  namespace: test
parameters:
  - name: value
    type: str
    default: "test"
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_node"
    spec:
      template: "{{value}}"
""")

        # Create pipeline that includes the macro
        pipeline_yaml = f"""
apiVersion: hexdag/v1
spec:
  nodes:
    - !include {macro_file.name}
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: test
      spec:
        macro: test:included_macro
        config:
          value: "hello"
        inputs: {{}}
"""
        # Build with temporary base path
        builder = YamlPipelineBuilder(base_path=tmp_path)
        graph, config = builder.build_from_yaml_string(pipeline_yaml, use_cache=False)

        # Macro should be registered and expanded
        assert "test_node" in graph.nodes


class TestYamlMacroMultiDocument:
    """Tests for multi-document YAML with macro definitions."""

    def test_multi_document_macro_definition_and_usage(self):
        """Test defining and using macro in same multi-document YAML."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: greeting
  namespace: test
parameters:
  - name: greeting_type
    type: str
    enum: [formal, casual]
    default: formal
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_greet"
    spec:
      template: "{% if greeting_type == 'formal' %}Good day{% else %}Hey{% endif %}"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: formal_greeting
      spec:
        macro: test:greeting
        config:
          greeting_type: formal
        inputs: {}
    - kind: macro_invocation
      metadata:
        name: casual_greeting
      spec:
        macro: test:greeting
        config:
          greeting_type: casual
        inputs: {}
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 2 nodes from 2 macro invocations
        assert len(graph.nodes) == 2
        assert "formal_greeting_greet" in graph.nodes
        assert "casual_greeting_greet" in graph.nodes

    def test_multiple_macro_definitions_before_pipeline(self):
        """Test multiple macro definitions in multi-document YAML."""
        yaml_content = """
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: macro_a
  namespace: test
parameters: []
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_a"
    spec:
      template: "A"
---
apiVersion: hexdag/v1
kind: Macro
metadata:
  name: macro_b
  namespace: test
parameters: []
nodes:
  - kind: prompt_node
    metadata:
      name: "{{name}}_b"
    spec:
      template: "B"
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: instance_a
      spec:
        macro: test:macro_a
        config: {}
        inputs: {}
    - kind: macro_invocation
      metadata:
        name: instance_b
      spec:
        macro: test:macro_b
        config: {}
        inputs: {}
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 2 nodes from 2 different macros
        assert len(graph.nodes) == 2
        assert "instance_a_a" in graph.nodes
        assert "instance_b_b" in graph.nodes
