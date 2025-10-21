"""Tests for YAML macro invocation support in YamlPipelineBuilder."""

import pytest

from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.pipeline_builder.yaml_builder import (
    YamlPipelineBuilder,
    YamlPipelineBuilderError,
)
from hexdag.core.registry import macro, registry
from hexdag.core.registry.models import ComponentType


class SimpleMacroConfig(MacroConfig):
    """Configuration for SimpleMacro."""

    steps: int = 3


@macro(name="simple_test_macro", namespace="test")
class SimpleMacro(ConfigurableMacro):
    """Simple test macro that expands to a linear chain of function nodes."""

    Config = SimpleMacroConfig

    def expand(self, instance_name: str, inputs: dict, dependencies: list[str]) -> DirectedGraph:
        """Expand to a simple chain of function nodes."""
        graph = DirectedGraph()

        # Create a chain of nodes
        for i in range(self.config.steps):
            node_name = f"{instance_name}_step{i}"

            node = NodeSpec(
                name=node_name,
                fn=lambda _input, step=i: {"result": f"step{step}", "input": _input},
            )

            if i == 0:
                # First node has no internal dependencies
                pass
            else:
                # Each subsequent node depends on previous
                node = node.after(f"{instance_name}_step{i - 1}")

            graph.add(node)

        return graph


class BranchingMacroConfig(MacroConfig):
    """Configuration for BranchingMacro."""

    branches: int = 2


@macro(name="branching_test_macro", namespace="test")
class BranchingMacro(ConfigurableMacro):
    """Test macro that expands to a branching graph structure."""

    Config = BranchingMacroConfig

    def expand(self, instance_name: str, inputs: dict, dependencies: list[str]) -> DirectedGraph:
        """Expand to a branching structure."""
        graph = DirectedGraph()

        # Create entry node
        entry = NodeSpec(
            name=f"{instance_name}_entry",
            fn=lambda _input: {"branches": self.config.branches},
        )
        graph.add(entry)

        # Create branch nodes
        for i in range(self.config.branches):
            branch = NodeSpec(
                name=f"{instance_name}_branch{i}",
                fn=lambda _input, b=i: {"branch": b},
            ).after(f"{instance_name}_entry")
            graph.add(branch)

        # Create merge node
        branch_names = [f"{instance_name}_branch{i}" for i in range(self.config.branches)]
        merge = NodeSpec(
            name=f"{instance_name}_merge",
            fn=lambda _input: {"merged": True},
        ).after(*branch_names)
        graph.add(merge)

        return graph


class TestYamlMacroInvocation:
    """Tests for macro_invocation kind in YAML pipelines."""

    @classmethod
    def setup_class(cls):
        """Register test macros before running tests."""
        # Manually register SimpleMacro
        registry.register(
            "simple_test_macro",
            SimpleMacro,
            namespace="test",
            component_type=ComponentType.MACRO,
            description="Simple test macro",
        )

        # Manually register BranchingMacro
        registry.register(
            "branching_test_macro",
            BranchingMacro,
            namespace="test",
            component_type=ComponentType.MACRO,
            description="Branching test macro",
        )

    def test_simple_macro_expansion(self):
        """Test basic macro expansion with default config."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: my_macro
      spec:
        macro: test:simple_test_macro
        config:
          steps: 3
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 3 nodes from macro expansion
        assert len(graph.nodes) == 3
        assert "my_macro_step0" in graph.nodes
        assert "my_macro_step1" in graph.nodes
        assert "my_macro_step2" in graph.nodes

        # Check dependencies
        assert graph.get_dependencies("my_macro_step1") == frozenset({"my_macro_step0"})
        assert graph.get_dependencies("my_macro_step2") == frozenset({"my_macro_step1"})

    def test_macro_with_custom_config(self):
        """Test macro expansion with custom configuration."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: custom_macro
      spec:
        macro: test:simple_test_macro
        config:
          steps: 5
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 5 nodes
        assert len(graph.nodes) == 5
        for i in range(5):
            assert f"custom_macro_step{i}" in graph.nodes

    def test_macro_with_external_dependencies(self):
        """Test macro expansion with dependencies on regular nodes."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: prompt_node
      metadata:
        name: input_node
      spec:
        template: "Input data"
    - kind: macro_invocation
      metadata:
        name: processing
      spec:
        macro: test:simple_test_macro
        config:
          steps: 2
        inputs: {}
        dependencies: [input_node]
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 3 nodes total (1 regular + 2 from macro)
        assert len(graph.nodes) == 3
        assert "input_node" in graph.nodes
        assert "processing_step0" in graph.nodes
        assert "processing_step1" in graph.nodes

        # First macro node should depend on input_node
        assert "input_node" in graph.get_dependencies("processing_step0")

    def test_branching_macro_expansion(self):
        """Test macro that expands to branching structure."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: parallel
      spec:
        macro: test:branching_test_macro
        config:
          branches: 3
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 5 nodes (1 entry + 3 branches + 1 merge)
        assert len(graph.nodes) == 5
        assert "parallel_entry" in graph.nodes
        assert "parallel_branch0" in graph.nodes
        assert "parallel_branch1" in graph.nodes
        assert "parallel_branch2" in graph.nodes
        assert "parallel_merge" in graph.nodes

        # Check dependencies
        assert graph.get_dependencies("parallel_branch0") == frozenset({"parallel_entry"})
        assert graph.get_dependencies("parallel_branch1") == frozenset({"parallel_entry"})
        assert graph.get_dependencies("parallel_branch2") == frozenset({"parallel_entry"})
        assert graph.get_dependencies("parallel_merge") == frozenset({
            "parallel_branch0",
            "parallel_branch1",
            "parallel_branch2",
        })

    def test_multiple_macro_invocations(self):
        """Test multiple macro invocations in same pipeline."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: first
      spec:
        macro: test:simple_test_macro
        config:
          steps: 2
        inputs: {}
        dependencies: []

    - kind: macro_invocation
      metadata:
        name: second
      spec:
        macro: test:simple_test_macro
        config:
          steps: 2
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content, use_cache=False)

        # Should have 4 nodes total (2 from each macro)
        assert len(graph.nodes) == 4
        assert "first_step0" in graph.nodes
        assert "first_step1" in graph.nodes
        assert "second_step0" in graph.nodes
        assert "second_step1" in graph.nodes

    def test_macro_invocation_missing_name(self):
        """Test error when macro_invocation lacks metadata.name."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata: {}
      spec:
        macro: test:simple_test_macro
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(YamlPipelineBuilderError, match="Missing 'metadata.name'"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_macro_invocation_missing_macro_ref(self):
        """Test error when macro_invocation lacks spec.macro."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: my_macro
      spec:
        config: {}
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(YamlPipelineBuilderError, match="must specify 'spec.macro' field"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_macro_not_found_in_registry(self):
        """Test error when macro doesn't exist in registry."""
        yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: macro_invocation
      metadata:
        name: my_macro
      spec:
        macro: test:nonexistent_macro
        config: {}
        inputs: {}
        dependencies: []
"""
        builder = YamlPipelineBuilder()

        with pytest.raises(YamlPipelineBuilderError, match="not found"):
            builder.build_from_yaml_string(yaml_content, use_cache=False)

    def test_load_macro_from_registry_success(self):
        """Test successful macro loading from registry."""
        # Test that registry.get() returns an instantiated macro
        macro_instance = registry.get("simple_test_macro", namespace="test")
        assert isinstance(macro_instance, SimpleMacro)
        assert isinstance(macro_instance, ConfigurableMacro)

    def test_load_macro_from_registry_not_found(self):
        """Test error when macro not found in registry."""
        from hexdag.core.registry.exceptions import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError):
            registry.get("nonexistent", namespace="test")
