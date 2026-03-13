"""Tests for AdapterDefinitionPlugin (kind: Adapter)."""

from __future__ import annotations

import pytest

from hexdag.compiler.plugins.adapter_definition import (
    AdapterDefinitionPlugin,
    clear_adapter_registry,
    get_adapter_definition,
)
from hexdag.kernel.domain.dag import DirectedGraph
from hexdag.kernel.exceptions import YamlPipelineBuilderError


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Clear adapter registry before each test."""
    clear_adapter_registry()


class TestAdapterDefinitionPlugin:
    """Tests for the AdapterDefinitionPlugin class."""

    def setup_method(self) -> None:
        self.plugin = AdapterDefinitionPlugin()
        self.graph = DirectedGraph()

    def test_can_handle_adapter(self) -> None:
        assert self.plugin.can_handle({"kind": "Adapter"}) is True

    def test_can_handle_not_adapter(self) -> None:
        assert self.plugin.can_handle({"kind": "Pipeline"}) is False
        assert self.plugin.can_handle({"kind": "Middleware"}) is False
        assert self.plugin.can_handle({}) is False

    def test_build_registers_adapter(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "test-adapter"},
            "spec": {
                "class": "hexdag.stdlib.adapters.mock.MockLLM",
                "config": {"model": "gpt-4o", "temperature": 0.5},
            },
        }
        result = self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]
        assert result is None

        adapter_def = get_adapter_definition("test-adapter")
        assert adapter_def is not None
        assert adapter_def["class"] == "hexdag.stdlib.adapters.mock.MockLLM"
        assert adapter_def["config"] == {"model": "gpt-4o", "temperature": 0.5}

    def test_build_registers_adapter_no_config(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "minimal-adapter"},
            "spec": {"class": "hexdag.stdlib.adapters.mock.MockLLM"},
        }
        self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

        adapter_def = get_adapter_definition("minimal-adapter")
        assert adapter_def is not None
        assert adapter_def["config"] == {}

    def test_build_missing_name(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {},
            "spec": {"class": "some.Adapter"},
        }
        with pytest.raises(YamlPipelineBuilderError, match="missing 'metadata.name'"):
            self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

    def test_build_missing_class(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "test"},
            "spec": {"config": {"key": "val"}},
        }
        with pytest.raises(YamlPipelineBuilderError, match="missing 'spec.class'"):
            self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

    def test_build_class_not_string(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "test"},
            "spec": {"class": 42},
        }
        with pytest.raises(YamlPipelineBuilderError, match="must be a module path string"):
            self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

    def test_build_config_not_dict(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "test"},
            "spec": {"class": "some.Adapter", "config": "not-a-dict"},
        }
        with pytest.raises(YamlPipelineBuilderError, match="must be a dict"):
            self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

    def test_build_returns_none(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "test"},
            "spec": {"class": "some.Adapter"},
        }
        assert self.plugin.build(config, None, self.graph) is None  # type: ignore[arg-type]

    def test_build_with_capabilities(self) -> None:
        config = {
            "kind": "Adapter",
            "metadata": {"name": "cap-adapter"},
            "spec": {
                "class": "some.Adapter",
                "config": {},
                "capabilities": {"requires": ["port.llm"]},
            },
        }
        self.plugin.build(config, None, self.graph)  # type: ignore[arg-type]

        adapter_def = get_adapter_definition("cap-adapter")
        assert adapter_def is not None
        assert adapter_def["capabilities"] == {"requires": ["port.llm"]}


class TestGetAdapterDefinition:
    """Tests for the adapter registry lookup."""

    def test_not_found_returns_none(self) -> None:
        assert get_adapter_definition("nonexistent") is None

    def test_clear_registry(self) -> None:
        plugin = AdapterDefinitionPlugin()
        graph = DirectedGraph()
        config = {
            "kind": "Adapter",
            "metadata": {"name": "to-clear"},
            "spec": {"class": "some.Adapter"},
        }
        plugin.build(config, None, graph)  # type: ignore[arg-type]
        assert get_adapter_definition("to-clear") is not None

        clear_adapter_registry()
        assert get_adapter_definition("to-clear") is None


class TestAdapterInMultiDocYaml:
    """Integration tests for kind: Adapter in multi-document YAML."""

    def test_adapter_with_pipeline(self) -> None:
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = """\
apiVersion: hexdag/v1
kind: Adapter
metadata:
  name: mock-llm
spec:
  class: hexdag.stdlib.adapters.mock.MockLLM
  config:
    responses: '{"result": "test"}'
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  ports:
    llm:
      ref: mock-llm
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
      dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, pipeline_config = builder.build_from_yaml_string(yaml_content)

        assert len(graph) > 0
        # Verify the ref was stored in pipeline_config ports
        assert "llm" in pipeline_config.ports
        assert pipeline_config.ports["llm"]["ref"] == "mock-llm"

    def test_adapter_filtered_from_environment_selection(self) -> None:
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder

        yaml_content = """\
apiVersion: hexdag/v1
kind: Adapter
metadata:
  name: test-adapter
spec:
  class: hexdag.stdlib.adapters.mock.MockLLM
  config: {}
---
apiVersion: hexdag/v1
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
      dependencies: []
"""
        builder = YamlPipelineBuilder()
        graph, pipeline_config = builder.build_from_yaml_string(yaml_content)

        # Should build the Pipeline, not the Adapter
        assert pipeline_config.metadata.get("name") == "my-pipeline"
        assert len(graph) > 0
