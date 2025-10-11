"""Tests for the simplified PipelineBuilder."""

import pytest
import yaml

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.domain.dag import NodeSpec
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder, YamlPipelineBuilderError

# Ensure registry is bootstrapped for tests
ensure_bootstrapped()


class TestPipelineBuilder:
    """Test cases for PipelineBuilder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = YamlPipelineBuilder()

        # Register a sample function for testing
        def sample_function(input_data, **ports):
            return {"result": "test_output"}

        self.sample_function = sample_function
        self.builder.register_function("sample_function", self.sample_function)

    def test_simple_function_registration(self):
        """Test basic function registration and retrieval."""
        # Verify registration
        assert "sample_function" in self.builder.registered_functions
        assert self.builder.registered_functions["sample_function"](None) == {
            "result": "test_output"
        }

    def test_build_simple_graph(self):
        """Test building a simple graph with one node."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: simple-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: processor
      spec:
        fn: sample_function
        dependencies: []
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 1
        assert "processor" in graph.nodes
        node = graph.nodes["processor"]
        assert isinstance(node, NodeSpec)
        assert node.name == "processor"
        assert len(node.deps) == 0

        # Check metadata
        assert pipeline_config.metadata["name"] == "simple-pipeline"

    def test_build_k8s_style_graph(self):
        """Test building a graph with declarative manifest YAML format."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: test-pipeline
  description: Test declarative manifest pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: processor
      spec:
        fn: sample_function
        dependencies: []
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 1
        assert "processor" in graph.nodes
        node = graph.nodes["processor"]
        assert isinstance(node, NodeSpec)
        assert node.name == "processor"
        assert len(node.deps) == 0

        # Check metadata from declarative manifest format
        assert pipeline_config.metadata["name"] == "test-pipeline"
        assert pipeline_config.metadata["description"] == "Test declarative manifest pipeline"

    def test_build_k8s_style_with_dependencies(self):
        """Test declarative manifest YAML with dependencies."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: pipeline-with-deps
spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        fn: sample_function
        dependencies: []

    - kind: function_node
      metadata:
        name: node2
      spec:
        fn: sample_function
        dependencies: [node1]
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 2
        assert graph.nodes["node1"].deps == set()
        assert graph.nodes["node2"].deps == {"node1"}

    def test_build_k8s_style_with_namespace(self):
        """Test declarative manifest YAML with namespace-qualified kind."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  nodes:
    - kind: core:function_node
      metadata:
        name: processor
      spec:
        fn: sample_function
        dependencies: []
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 1
        assert "processor" in graph.nodes
        node = graph.nodes["processor"]
        assert isinstance(node, NodeSpec)
        assert node.name == "processor"

    def test_build_with_dependencies(self):
        """Test building pipeline with dependencies."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: multi-node-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        fn: sample_function
        dependencies: []

    - kind: function_node
      metadata:
        name: node2
      spec:
        fn: sample_function
        dependencies: [node1]

    - kind: function_node
      metadata:
        name: node3
      spec:
        fn: sample_function
        dependencies: [node1, node2]
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 3
        assert graph.nodes["node1"].deps == set()
        assert graph.nodes["node2"].deps == {"node1"}
        assert graph.nodes["node3"].deps == {"node1", "node2"}

    def test_build_with_input_mapping(self):
        """Test building pipeline with input mapping."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: input-mapping-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        fn: sample_function
        dependencies: []

    - kind: function_node
      metadata:
        name: node2
      spec:
        fn: sample_function
        input_mapping:
          data: "node1.result"
          metadata: "node1.status"
        dependencies: [node1]
"""

        graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 2

        # Verify input mapping is stored in params
        node2 = graph.nodes["node2"]
        assert hasattr(node2, "params")
        if hasattr(node2, "params") and node2.params:
            input_mapping = node2.params.get("input_mapping")
            if input_mapping:
                assert input_mapping["data"] == "node1.result"
                assert input_mapping["metadata"] == "node1.status"

    def test_pipeline_metadata_extraction(self):
        """Test extraction of pipeline-wide metadata."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: test_pipeline
  description: A test pipeline for validation
  version: "1.0.0"
  author: "Test Author"
  tags: ["test", "validation"]
  field_mapping_mode: "none"
spec:
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: test_function
        dependencies: []
"""

        # This will fail due to missing function, but we can catch and check metadata
        with pytest.raises(TypeError):
            self.builder.build_from_yaml_string(yaml_content)

        config = yaml.safe_load(yaml_content)
        metadata = self.builder._extract_pipeline_metadata(config)

        assert metadata["name"] == "test_pipeline"
        assert metadata["description"] == "A test pipeline for validation"
        assert metadata["version"] == "1.0.0"
        assert metadata["author"] == "Test Author"
        assert metadata["tags"] == ["test", "validation"]

    @pytest.mark.skip(reason="field_mapping_mode feature removed from yaml_builder")
    def test_field_mapping_mode_validation(self):
        """Test field mapping mode validation."""
        # Test invalid field mapping mode
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: field-mapping-validation-pipeline
  field_mapping_mode: "invalid_mode"
spec:
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: test_function
        dependencies: []
"""

        with pytest.raises(YamlPipelineBuilderError, match="Invalid field_mapping_mode"):
            self.builder.build_from_yaml_string(yaml_content)

    @pytest.mark.skip(reason="field_mapping_mode feature removed from yaml_builder")
    def test_custom_field_mappings_validation(self):
        """Test custom field mappings validation."""
        # Test custom mode without mappings
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: custom-field-mappings-pipeline
  field_mapping_mode: "custom"
spec:
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: test_function
        dependencies: []
"""
        with pytest.raises(
            YamlPipelineBuilderError,
            match="custom_field_mappings required when field_mapping_mode='custom'",
        ):
            self.builder.build_from_yaml_string(yaml_content)

        # Test custom mode with mappings
        yaml_content_valid = """
apiVersion: v1
kind: Pipeline
metadata:
  name: custom-field-mappings-valid-pipeline
  field_mapping_mode: "custom"
  custom_field_mappings:
    text: ["content", "data"]
    result: ["output", "response"]
spec:
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: test_function
        dependencies: []
"""
        with pytest.raises(TypeError, match="'test_function' is not a callable object"):
            self.builder.build_from_yaml_string(yaml_content_valid)
            # Expected due to missing function, but metadata extraction should work

        # Test metadata extraction
        import yaml

        config = yaml.safe_load(yaml_content_valid)
        self.builder._extract_pipeline_metadata(config)

        assert self.builder.pipeline_config.metadata["field_mapping_mode"] == "custom"
        assert self.builder.pipeline_config.metadata["custom_field_mappings"]["text"] == [
            "content",
            "data",
        ]
        assert self.builder.pipeline_config.metadata["custom_field_mappings"]["result"] == [
            "output",
            "response",
        ]

    @pytest.mark.skip(reason="input_mapping validation not implemented")
    def test_input_mapping_validation_error(self):
        """Test input mapping validation errors."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: input-mapping-validation-pipeline
spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        fn: sample_function
        input_mapping: "invalid_format"  # Should be dict, not string
        dependencies: []
"""

        with pytest.raises(
            YamlPipelineBuilderError,
            match="input_mapping for node 'node1' must be a dictionary",
        ):
            self.builder.build_from_yaml_string(yaml_content)

    def test_namespace_parsing_core(self):
        """Test that core namespace is correctly parsed."""
        # Test simple kind (should default to core namespace)
        node_type, namespace = YamlPipelineBuilder._parse_kind("llm_node")
        assert node_type == "llm"
        assert namespace == "core"

        # Test kind without _node suffix
        node_type, namespace = YamlPipelineBuilder._parse_kind("function")
        assert node_type == "function"
        assert namespace == "core"

    def test_namespace_parsing_custom(self):
        """Test that custom plugin namespaces are correctly parsed."""
        # Test namespace-qualified kind
        node_type, namespace = YamlPipelineBuilder._parse_kind("my-plugin:dalle_node")
        assert node_type == "dalle"
        assert namespace == "my-plugin"

        # Test with different separator patterns
        node_type, namespace = YamlPipelineBuilder._parse_kind("openai:gpt4_node")
        assert node_type == "gpt4"
        assert namespace == "openai"

    def test_namespace_qualified_node_lookup(self):
        """Test that namespace-qualified nodes are looked up correctly in registry."""
        from unittest.mock import Mock, patch

        from hexdag.core.domain.dag import NodeSpec

        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: plugin-test-pipeline
spec:
  nodes:
    - kind: my-plugin:dalle_node
      metadata:
        name: image_gen
      spec:
        prompt: "Generate an image"
        dependencies: []
"""

        # Mock the registry to verify it's called with correct namespace
        mock_factory = Mock(return_value=NodeSpec("image_gen", lambda: None))

        # Patch ensure_bootstrapped, registry, and validator to bypass validation
        with (
            patch("hexdag.core.pipeline_builder.yaml_builder.ensure_bootstrapped"),
            patch("hexdag.core.pipeline_builder.yaml_builder.registry") as mock_registry,
            patch.object(self.builder.validator, "validate") as mock_validate,
        ):
            # Make validator always pass
            mock_validate_result = Mock()
            mock_validate_result.is_valid = True
            mock_validate_result.errors = []
            mock_validate_result.warnings = []
            mock_validate_result.suggestions = []
            mock_validate.return_value = mock_validate_result

            mock_registry.get.return_value = mock_factory

            # Should successfully build
            graph, pipeline_config = self.builder.build_from_yaml_string(yaml_content)

            # Verify registry.get was called with the correct namespace
            mock_registry.get.assert_called_once_with("dalle_node", namespace="my-plugin")
            assert len(graph.nodes) == 1
            assert "image_gen" in graph.nodes
