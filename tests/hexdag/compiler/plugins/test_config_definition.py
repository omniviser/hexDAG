"""Tests for the ConfigDefinitionPlugin."""

from __future__ import annotations

from hexdag.compiler.plugins.config_definition import ConfigDefinitionPlugin
from hexdag.compiler.yaml_builder import YamlPipelineBuilder
from hexdag.kernel.domain.dag import DirectedGraph


class TestConfigDefinitionPlugin:
    """Tests for ConfigDefinitionPlugin."""

    def test_can_handle_config(self) -> None:
        """Test can_handle returns True for kind: Config."""
        plugin = ConfigDefinitionPlugin()
        assert plugin.can_handle({"kind": "Config"}) is True

    def test_can_handle_not_config(self) -> None:
        """Test can_handle returns False for other kinds."""
        plugin = ConfigDefinitionPlugin()
        assert plugin.can_handle({"kind": "Pipeline"}) is False
        assert plugin.can_handle({"kind": "Macro"}) is False
        assert plugin.can_handle({}) is False

    def test_build_parses_spec(self) -> None:
        """Test build() extracts spec and creates HexDAGConfig."""
        plugin = ConfigDefinitionPlugin()
        builder = YamlPipelineBuilder()
        graph = DirectedGraph()

        config_doc = {
            "kind": "Config",
            "metadata": {"name": "test-config"},
            "spec": {
                "modules": ["myapp.nodes"],
                "kernel": {"max_concurrent_nodes": 3},
                "limits": {"max_llm_calls": 50},
                "caps": {"deny": ["secret"]},
            },
        }

        result = plugin.build(config_doc, builder, graph)

        assert result is None
        assert builder._inline_config is not None
        assert builder._inline_config.modules == ["myapp.nodes"]
        assert builder._inline_config.orchestrator.max_concurrent_nodes == 3
        assert builder._inline_config.limits.max_llm_calls == 50
        assert builder._inline_config.caps.deny == ["secret"]

    def test_build_returns_none(self) -> None:
        """Test build() returns None (no nodes added)."""
        plugin = ConfigDefinitionPlugin()
        builder = YamlPipelineBuilder()
        graph = DirectedGraph()

        result = plugin.build(
            {"kind": "Config", "metadata": {"name": "test"}, "spec": {}},
            builder,
            graph,
        )

        assert result is None

    def test_build_empty_spec(self) -> None:
        """Test build() handles empty spec gracefully."""
        plugin = ConfigDefinitionPlugin()
        builder = YamlPipelineBuilder()
        graph = DirectedGraph()

        plugin.build(
            {"kind": "Config", "metadata": {"name": "empty"}, "spec": {}},
            builder,
            graph,
        )

        assert builder._inline_config is not None
        assert builder._inline_config.modules == []


class TestConfigInMultiDocYaml:
    """Tests for kind: Config in multi-document YAML."""

    def test_config_with_pipeline(self) -> None:
        """Test Config + Pipeline in same YAML file."""
        yaml_content = """\
---
kind: Config
metadata:
  name: inline-config
spec:
  kernel:
    max_concurrent_nodes: 3
---
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: step1
      spec:
        fn: "json.loads"
"""
        builder = YamlPipelineBuilder()
        graph, pipeline_config = builder.build_from_yaml_string(yaml_content)

        assert builder._inline_config is not None
        assert builder._inline_config.orchestrator.max_concurrent_nodes == 3
        assert len(graph) > 0

    def test_config_filtered_from_pipeline_selection(self) -> None:
        """Test kind: Config is filtered out in _select_environment."""
        yaml_content = """\
---
kind: Config
metadata:
  name: cfg
spec:
  modules: [myapp.nodes]
---
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: step1
      spec:
        fn: "json.loads"
"""
        builder = YamlPipelineBuilder()
        graph, pipeline_config = builder.build_from_yaml_string(yaml_content)

        # Pipeline should be selected, not the Config
        assert pipeline_config.metadata.get("name") == "my-pipeline"
