"""Tests for the simplified PipelineBuilder."""

import pytest
import yaml

from hexai.agent_factory.yaml_builder import YamlPipelineBuilder, YamlPipelineBuilderError
from hexai.core.domain.dag import NodeSpec


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
        # Register a function
        self.builder.register_function(
            "sample_function",
            lambda input_data, **ports: {"result": "test"},
        )

        # Verify registration
        assert "sample_function" in self.builder.registered_functions
        assert callable(self.builder.registered_functions["sample_function"])

    def test_build_simple_graph(self):
        """Test building a simple graph with one node."""
        # Register required function
        self.builder.register_function(
            "sample_function",
            lambda input_data, **ports: {"result": "test"},
        )

        yaml_content = """
nodes:
  - id: processor
    type: function
    params:
      fn: sample_function
    depends_on: []
"""

        graph, metadata = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 1
        assert "processor" in graph.nodes
        node = graph.nodes["processor"]
        assert isinstance(node, NodeSpec)
        assert node.name == "processor"
        assert len(node.deps) == 0

        # Check default metadata
        assert metadata["field_mapping_mode"] == "default"
        assert metadata["name"] is None
        assert metadata["description"] is None

    def test_build_with_dependencies(self):
        """Test building pipeline with dependencies."""
        # Register required function
        self.builder.register_function(
            "sample_function",
            lambda input_data, **ports: {"result": "test"},
        )

        yaml_content = """
nodes:
  - id: node1
    type: function
    params:
      fn: sample_function
    depends_on: []

  - id: node2
    type: function
    params:
      fn: sample_function
    depends_on: [node1]

  - id: node3
    type: function
    params:
      fn: sample_function
    depends_on: [node1, node2]
"""

        graph, metadata = self.builder.build_from_yaml_string(yaml_content)

        assert len(graph.nodes) == 3
        assert graph.nodes["node1"].deps == set()
        assert graph.nodes["node2"].deps == {"node1"}
        assert graph.nodes["node3"].deps == {"node1", "node2"}

    def test_build_with_input_mapping(self):
        """Test building pipeline with input mapping."""
        # Register required function
        self.builder.register_function(
            "sample_function",
            lambda input_data, **ports: {"result": "test"},
        )

        yaml_content = """
nodes:
  - id: node1
    type: function
    params:
      fn: sample_function
    depends_on: []

  - id: node2
    type: function
    params:
      fn: sample_function
      input_mapping:
        data: "node1.result"
        metadata: "node1.status"
    depends_on: [node1]
"""

        graph, metadata = self.builder.build_from_yaml_string(yaml_content)

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
name: test_pipeline
description: A test pipeline for validation
field_mapping_mode: "none"
version: "1.0.0"
author: "Test Author"
tags: ["test", "validation"]

nodes:
  - id: test_node
    type: function
    params:
      fn: test_function
"""

        # This will fail due to missing function, but we can catch and check metadata
        with pytest.raises(TypeError, match="not a callable object"):
            self.builder.build_from_yaml_string(yaml_content)
            # Expected due to missing function registration

        # Test metadata extraction separately

        config = yaml.safe_load(yaml_content)
        metadata = self.builder._extract_pipeline_metadata(config)

        assert metadata["name"] == "test_pipeline"
        assert metadata["description"] == "A test pipeline for validation"
        assert metadata["field_mapping_mode"] == "none"
        assert metadata["version"] == "1.0.0"
        assert metadata["author"] == "Test Author"
        assert metadata["tags"] == ["test", "validation"]

    def test_field_mapping_mode_validation(self):
        """Test field mapping mode validation."""
        # Test invalid field mapping mode
        yaml_content = """
field_mapping_mode: "invalid_mode"

nodes:
  - id: test_node
    type: function
    params:
      fn: test_function
"""

        with pytest.raises(YamlPipelineBuilderError, match="Invalid field_mapping_mode"):
            self.builder.build_from_yaml_string(yaml_content)

    def test_custom_field_mappings_validation(self):
        """Test custom field mappings validation."""
        # Test custom mode without mappings
        yaml_content = """
field_mapping_mode: "custom"

nodes:
  - id: test_node
    type: function
    params:
      fn: test_function
"""

        with pytest.raises(YamlPipelineBuilderError, match="custom_field_mappings required"):
            self.builder.build_from_yaml_string(yaml_content)

        # Test custom mode with mappings
        yaml_content_valid = """
field_mapping_mode: "custom"
custom_field_mappings:
  text: ["content", "data"]
  result: ["output", "response"]

nodes:
  - id: test_node
    type: function
    params:
      fn: test_function
"""

        with pytest.raises(TypeError, match="not a callable object"):
            self.builder.build_from_yaml_string(yaml_content_valid)
            # Expected due to missing function, but metadata extraction should work

        # Test metadata extraction
        import yaml

        config = yaml.safe_load(yaml_content_valid)
        metadata = self.builder._extract_pipeline_metadata(config)

        assert metadata["field_mapping_mode"] == "custom"
        assert metadata["custom_field_mappings"]["text"] == ["content", "data"]
        assert metadata["custom_field_mappings"]["result"] == ["output", "response"]

    def test_input_mapping_validation_error(self):
        """Test input mapping validation errors."""
        yaml_content = """
nodes:
  - id: node1
    type: function
    params:
      fn: sample_function
      input_mapping: "invalid_format"  # Should be dict, not string
"""

        with pytest.raises(
            YamlPipelineBuilderError,
            match="input_mapping for node 'node1' must be a dictionary",
        ):
            self.builder.build_from_yaml_string(yaml_content)

    def test_data_mapping_validation(self):
        """Test data mapping validation warnings."""
        # Test with missing dependency reference
        yaml_content = """
nodes:
  - id: node1
    type: function
    params:
      fn: sample_function
    depends_on: []

  - id: node2
    type: function
    params:
      fn: sample_function
      input_mapping:
        data: "missing_node.result"  # References non-existent node
    depends_on: [node1]
"""

        import yaml

        config = yaml.safe_load(yaml_content)
        warnings = self.builder.validate_data_mapping(config)

        assert len(warnings) > 0
        assert any("missing_node" in warning for warning in warnings)

    def test_data_mapping_dependency_validation(self):
        """Test data mapping dependency validation warnings."""
        yaml_content = """
nodes:
  - id: node1
    type: function
    params:
      fn: sample_function
    depends_on: []

  - id: node2
    type: function
    params:
      fn: sample_function
      input_mapping:
        data: "node3.result"  # References node3 but not in dependencies
    depends_on: [node1]

  - id: node3
    type: function
    params:
      fn: sample_function
    depends_on: [node1]
"""

        import yaml

        config = yaml.safe_load(yaml_content)
        warnings = self.builder.validate_data_mapping(config)

        assert len(warnings) > 0
        assert any("not in dependencies" in warning for warning in warnings)
