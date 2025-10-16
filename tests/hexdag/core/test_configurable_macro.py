"""Tests for ConfigurableMacro base class."""

import pytest

from hexdag.core.configurable import ConfigurableMacro, MacroConfig
from hexdag.core.domain.dag import DirectedGraph, NodeSpec
from hexdag.core.registry import macro


class SimpleMacroConfig(MacroConfig):
    """Test configuration for simple macro."""

    depth: int = 3
    enable_feature: bool = True


class SimpleMacro(ConfigurableMacro):
    """Simple macro for testing."""

    Config = SimpleMacroConfig

    def expand(self, instance_name, inputs, dependencies):
        """Expand to simple two-node graph."""
        graph = DirectedGraph()

        node1 = NodeSpec(name=f"{instance_name}_step1", fn=lambda: "step1")
        node2 = NodeSpec(
            name=f"{instance_name}_step2",
            fn=lambda: "step2",
            deps=frozenset([f"{instance_name}_step1"]),
        )

        graph.add(node1)
        graph.add(node2)

        # Add external dependencies to entry node
        if dependencies:
            from dataclasses import replace

            node1_updated = replace(node1, deps=frozenset(dependencies))
            graph.nodes[node1.name] = node1_updated

        return graph


class TestMacroConfig:
    """Test suite for MacroConfig base class."""

    def test_macro_config_defaults(self):
        """Test MacroConfig with default values."""
        config = MacroConfig()
        assert hasattr(config, "model_config")

    def test_macro_config_frozen(self):
        """Test that MacroConfig is immutable (frozen)."""
        from pydantic import ValidationError

        config = MacroConfig()
        # Pydantic frozen models raise ValidationError on assignment
        with pytest.raises(ValidationError):
            config.some_field = "value"  # type: ignore

    def test_custom_macro_config(self):
        """Test custom MacroConfig subclass."""
        config = SimpleMacroConfig(depth=5, enable_feature=False)
        assert config.depth == 5
        assert config.enable_feature is False

    def test_custom_macro_config_defaults(self):
        """Test custom MacroConfig uses default values."""
        config = SimpleMacroConfig()
        assert config.depth == 3
        assert config.enable_feature is True


class TestConfigurableMacro:
    """Test suite for ConfigurableMacro base class."""

    def test_macro_initialization(self):
        """Test basic macro initialization."""
        macro = SimpleMacro()
        assert macro.config.depth == 3
        assert macro.config.enable_feature is True

    def test_macro_initialization_with_params(self):
        """Test macro initialization with custom parameters."""
        macro = SimpleMacro(depth=10, enable_feature=False)
        assert macro.config.depth == 10
        assert macro.config.enable_feature is False

    def test_macro_without_config_class_raises(self):
        """Test that macro without Config class raises error."""

        class BadMacro(ConfigurableMacro):
            pass

        with pytest.raises(AttributeError, match="must define a nested Config class"):
            BadMacro()

    def test_macro_expand_not_implemented(self):
        """Test that base expand() raises NotImplementedError."""

        class MinimalMacro(ConfigurableMacro):
            Config = MacroConfig

        macro = MinimalMacro()
        with pytest.raises(NotImplementedError, match="must implement expand"):
            macro.expand("test", {}, [])

    def test_macro_expand_basic(self):
        """Test basic macro expansion."""
        macro = SimpleMacro()
        graph = macro.expand("test_instance", {}, [])

        assert isinstance(graph, DirectedGraph)
        assert len(graph.nodes) == 2
        assert "test_instance_step1" in graph.nodes
        assert "test_instance_step2" in graph.nodes

    def test_macro_expand_with_dependencies(self):
        """Test macro expansion with external dependencies."""
        macro = SimpleMacro()
        graph = macro.expand("test_instance", {}, ["external_node"])

        # Check that entry node has external dependency
        step1 = graph.nodes["test_instance_step1"]
        assert "external_node" in step1.deps

    def test_macro_expand_preserves_config(self):
        """Test that macro expansion uses config values."""
        macro = SimpleMacro(depth=7, enable_feature=False)
        macro.expand("test", {}, [])

        # Config is accessible during expansion
        assert macro.config.depth == 7
        assert macro.config.enable_feature is False

    def test_validate_inputs_success(self):
        """Test validate_inputs with valid inputs."""
        macro = SimpleMacro()

        inputs = {"topic": "AI", "optional_param": "value"}
        required = ["topic"]
        optional = {"optional_param": "default"}

        result = macro.validate_inputs(inputs, required, optional)

        assert result["topic"] == "AI"
        assert result["optional_param"] == "value"

    def test_validate_inputs_missing_required(self):
        """Test validate_inputs raises on missing required inputs."""
        macro = SimpleMacro()

        inputs = {}
        required = ["topic", "depth"]
        optional = {}

        with pytest.raises(ValueError, match="Missing required inputs"):
            macro.validate_inputs(inputs, required, optional)

    def test_validate_inputs_applies_defaults(self):
        """Test validate_inputs applies default values."""
        macro = SimpleMacro()

        inputs = {"topic": "AI"}
        required = ["topic"]
        optional = {"depth": 3, "feature": True}

        result = macro.validate_inputs(inputs, required, optional)

        assert result["topic"] == "AI"
        assert result["depth"] == 3  # Default applied
        assert result["feature"] is True  # Default applied

    def test_macro_repr(self):
        """Test macro string representation."""
        macro = SimpleMacro(depth=5)
        repr_str = repr(macro)

        assert "SimpleMacro" in repr_str
        assert "depth" in repr_str or "5" in repr_str


class TestMacroDecorator:
    """Test suite for @macro decorator."""

    def test_macro_decorator_basic(self):
        """Test @macro decorator registration."""

        @macro(name="test_macro", namespace="test")
        class TestMacro(ConfigurableMacro):
            Config = MacroConfig

            def expand(self, instance_name, inputs, dependencies):
                return DirectedGraph()

        # Check metadata was added (uses internal attribute names from component decorator)
        assert hasattr(TestMacro, "_hexdag_type")
        assert TestMacro._hexdag_type.value == "macro"
        assert TestMacro._hexdag_name == "test_macro"
        assert TestMacro._hexdag_namespace == "test"

    def test_macro_decorator_validates_inheritance(self):
        """Test @macro decorator validates ConfigurableMacro inheritance."""
        from hexdag.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must inherit from ConfigurableMacro"):

            @macro(name="bad_macro")
            class NotAMacro:  # Doesn't inherit ConfigurableMacro
                pass

    def test_macro_decorator_validates_config(self):
        """Test @macro decorator validates Config class presence."""
        from hexdag.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must define a Config class"):

            @macro(name="no_config_macro")
            class NoConfigMacro(ConfigurableMacro):
                # Missing Config class
                def expand(self, instance_name, inputs, dependencies):
                    return DirectedGraph()

    def test_macro_decorator_with_description(self):
        """Test @macro decorator with description."""

        @macro(name="documented_macro", namespace="core", description="Test macro")
        class DocumentedMacro(ConfigurableMacro):
            Config = MacroConfig

            def expand(self, instance_name, inputs, dependencies):
                return DirectedGraph()

        assert DocumentedMacro._hexdag_description == "Test macro"


class TestMacroIntegration:
    """Integration tests for macro system."""

    def test_macro_expansion_with_multiple_instances(self):
        """Test expanding same macro multiple times with different names."""
        macro = SimpleMacro()

        # Expand first instance
        graph1 = macro.expand("instance1", {}, [])
        assert "instance1_step1" in graph1.nodes
        assert "instance1_step2" in graph1.nodes

        # Expand second instance
        graph2 = macro.expand("instance2", {}, [])
        assert "instance2_step1" in graph2.nodes
        assert "instance2_step2" in graph2.nodes

        # Instances are independent
        assert "instance1_step1" not in graph2.nodes
        assert "instance2_step1" not in graph1.nodes

    def test_macro_expansion_chain(self):
        """Test chaining macro expansions (macro depends on another macro's output)."""
        macro = SimpleMacro()

        # First macro instance
        graph = DirectedGraph()
        subgraph1 = macro.expand("macro1", {}, [])

        # Add first macro's nodes
        for node in subgraph1.nodes.values():
            graph.add(node)

        # Second macro depends on first macro's output
        subgraph2 = macro.expand("macro2", {}, ["macro1_step2"])

        # Add second macro's nodes
        for node in subgraph2.nodes.values():
            graph.add(node)

        # Verify dependency chain
        waves = graph.waves()
        # Each macro has 2 steps, macro2 depends on macro1_step2
        # So: wave0=[macro1_step1], wave1=[macro1_step2], wave2=[macro2_step1], wave3=[macro2_step2]
        assert len(waves) == 4
        assert "macro1_step1" in waves[0]
        assert "macro1_step2" in waves[1]
        assert "macro2_step1" in waves[2]
        assert "macro2_step2" in waves[3]

    def test_nested_macro_scenario(self):
        """Test scenario where macro output is used to parameterize another macro."""

        class ParameterizedMacro(ConfigurableMacro):
            """Macro that uses inputs to configure expansion."""

            Config = MacroConfig

            def expand(self, instance_name, inputs, dependencies):
                count = inputs.get("count", 2)
                graph = DirectedGraph()

                prev_node = None
                for i in range(count):
                    node = NodeSpec(name=f"{instance_name}_step{i}", fn=lambda i=i: f"step{i}")
                    if prev_node:
                        from dataclasses import replace

                        node = replace(node, deps=frozenset([prev_node]))
                    elif dependencies:
                        from dataclasses import replace

                        node = replace(node, deps=frozenset(dependencies))
                    graph.add(node)
                    prev_node = node.name

                return graph

        macro = ParameterizedMacro()

        # Expand with count=3
        graph = macro.expand("dynamic", {"count": 3}, [])
        assert len(graph.nodes) == 3
        assert "dynamic_step0" in graph.nodes
        assert "dynamic_step1" in graph.nodes
        assert "dynamic_step2" in graph.nodes

        # Verify chain
        waves = graph.waves()
        assert len(waves) == 3
