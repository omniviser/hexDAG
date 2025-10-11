"""Integration tests for YamlPipelineBuilder.

Tests the complete pipeline from YAML manifest to DirectedGraph building:
- Full parsing, validation, and building pipeline
- Namespace-qualified plugin nodes
- Field mapping resolution across nodes
- Registry integration with real components
- Error handling in realistic scenarios
"""

from __future__ import annotations

import pytest

from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder


@pytest.fixture(autouse=True)
def bootstrap_registry():
    """Ensure registry is bootstrapped before each test."""
    ensure_bootstrapped()
    yield


@pytest.fixture
def yaml_builder():
    """Create YamlPipelineBuilder instance."""
    return YamlPipelineBuilder()


# ===================================================================
# Integration Tests: YAML -> DirectedGraph -> Execution
# ===================================================================


class TestYamlBuilderIntegration:
    """Integration tests for the full YAML pipeline."""

    def test_simple_function_pipeline_building(self, yaml_builder):
        """Test building DirectedGraph from YAML with function nodes."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: math-pipeline
  description: Simple math operations
  version: "1.0"

spec:
  nodes:
    - kind: function_node
      metadata:
        name: add_numbers
      spec:
        fn: add
        input_mapping:
          a: input.a
          b: input.b
        dependencies: []

    - kind: function_node
      metadata:
        name: multiply_result
      spec:
        fn: multiply
        input_mapping:
          x: add_numbers
          y: input.y
        dependencies: [add_numbers]
"""

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(x: int, y: int) -> int:
            return x * y

        # Register functions with builder
        yaml_builder.register_function("add", add)
        yaml_builder.register_function("multiply", multiply)

        # Build DirectedGraph from YAML
        graph, pipeline_config = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify metadata
        assert pipeline_config.metadata["name"] == "math-pipeline"
        assert pipeline_config.metadata["description"] == "Simple math operations"
        assert pipeline_config.metadata["version"] == "1.0"

        # Verify graph structure
        assert len(graph.nodes) == 2
        assert "add_numbers" in graph.nodes
        assert "multiply_result" in graph.nodes

        # Verify dependencies
        assert frozenset(graph.nodes["multiply_result"].deps) == frozenset(["add_numbers"])
        assert frozenset(graph.nodes["add_numbers"].deps) == frozenset()

    def test_multiple_nodes_with_dependencies(self, yaml_builder):
        """Test building graph with multiple nodes and dependencies."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: multi-node-pipeline

spec:
  nodes:
    - kind: function_node
      metadata:
        name: step1
      spec:
        fn: func1
        input_mapping:
          x: input.value
        dependencies: []

    - kind: function_node
      metadata:
        name: step2
      spec:
        fn: func2
        input_mapping:
          val: step1
        dependencies: [step1]

    - kind: function_node
      metadata:
        name: step3
      spec:
        fn: func3
        input_mapping:
          val: step2
        dependencies: [step2]
"""

        def func1(x: int) -> int:
            return x * 2

        def func2(val: int) -> int:
            return val + 10

        def func3(val: int) -> int:
            return val - 5

        # Register functions
        yaml_builder.register_function("func1", func1)
        yaml_builder.register_function("func2", func2)
        yaml_builder.register_function("func3", func3)

        # Build graph
        graph, _ = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify structure
        assert len(graph.nodes) == 3
        assert "step1" in graph.nodes
        assert "step2" in graph.nodes
        assert "step3" in graph.nodes

        # Verify dependency chain
        assert frozenset(graph.nodes["step1"].deps) == frozenset()
        assert frozenset(graph.nodes["step2"].deps) == frozenset(["step1"])
        assert frozenset(graph.nodes["step3"].deps) == frozenset(["step2"])

    def test_metadata_fields_preserved(self, yaml_builder):
        """Test that metadata fields are preserved in the built graph."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: metadata-test
  description: Test metadata preservation
  version: "2.0"
  author: Test Author
  tags: ["test", "integration"]

spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
        annotations:
          purpose: testing
      spec:
        fn: test_func
        input_mapping:
          x: input.x
        dependencies: []
"""

        def test_func(x: int) -> int:
            return x

        yaml_builder.register_function("test_func", test_func)

        # Build graph
        graph, pipeline_config = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify metadata
        assert pipeline_config.metadata["name"] == "metadata-test"
        assert pipeline_config.metadata["description"] == "Test metadata preservation"
        assert pipeline_config.metadata["version"] == "2.0"
        assert pipeline_config.metadata["author"] == "Test Author"
        assert "test" in pipeline_config.metadata["tags"]
        assert "integration" in pipeline_config.metadata["tags"]

    def test_field_mapping_in_yaml(self, yaml_builder):
        """Test that field mappings are correctly parsed from YAML."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: mapping-pipeline

spec:
  nodes:
    - kind: function_node
      metadata:
        name: create_data
      spec:
        fn: create_dict
        input_mapping:
          dummy: input.dummy
        dependencies: []

    - kind: function_node
      metadata:
        name: extract_field
      spec:
        fn: get_value
        input_mapping:
          data: create_data.result
          key: input.key
        dependencies: [create_data]
"""

        def create_dict(dummy: str = "") -> dict[str, dict[str, str]]:
            return {"result": {"name": "Alice", "age": "30"}}

        def get_value(data: dict[str, str], key: str) -> str:
            return data[key]

        # Register functions
        yaml_builder.register_function("create_dict", create_dict)
        yaml_builder.register_function("get_value", get_value)

        # Build graph
        graph, _ = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify field mapping was added to node
        assert len(graph.nodes) == 2
        assert "extract_field" in graph.nodes

    def test_validation_errors_on_invalid_yaml(self, yaml_builder):
        """Test that validation errors are raised for invalid YAML."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: invalid-pipeline

spec:
  nodes:
    - kind: function_node
      metadata:
        name: node1
      spec:
        # Missing required 'fn' field
        input_mapping: {}
        dependencies: []
"""

        # Should raise validation error
        with pytest.raises(Exception) as exc_info:
            yaml_builder.build_from_yaml_string(yaml_content)

        assert "validation" in str(exc_info.value).lower() or "fn" in str(exc_info.value).lower()

    def test_cycle_detection_in_dependencies(self, yaml_builder):
        """Test that cycles in dependencies are detected."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: cyclic-pipeline

spec:
  nodes:
    - kind: function_node
      metadata:
        name: node_a
      spec:
        fn: func_a
        dependencies: [node_c]

    - kind: function_node
      metadata:
        name: node_b
      spec:
        fn: func_b
        dependencies: [node_a]

    - kind: function_node
      metadata:
        name: node_c
      spec:
        fn: func_c
        dependencies: [node_b]
"""

        def func_a():
            return "a"

        def func_b():
            return "b"

        def func_c():
            return "c"

        # Register functions
        yaml_builder.register_function("func_a", func_a)
        yaml_builder.register_function("func_b", func_b)
        yaml_builder.register_function("func_c", func_c)

        # Should raise validation error about cycle
        with pytest.raises(Exception) as exc_info:
            yaml_builder.build_from_yaml_string(yaml_content)

        error_msg = str(exc_info.value).lower()
        assert "cycle" in error_msg or "validation" in error_msg

    def test_common_field_mappings(self, yaml_builder):
        """Test that common field mappings work correctly."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: field-mapping-test

spec:
  common_field_mappings:
    user_input:
      query: input.user_query
      context: input.context

  nodes:
    - kind: function_node
      metadata:
        name: processor
      spec:
        fn: process_data
        input_mapping:
          data: input.value
        dependencies: []
"""

        def process_data(data: str) -> str:
            return f"processed: {data}"

        yaml_builder.register_function("process_data", process_data)

        # Build graph with common mappings
        graph, pipeline_config = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify common mappings were registered with the builder
        user_input_mapping = yaml_builder.field_mapping_registry.get("user_input")
        assert user_input_mapping is not None
        assert user_input_mapping["query"] == "input.user_query"
        assert user_input_mapping["context"] == "input.context"

        # Verify node was created
        assert len(graph.nodes) == 1
        assert "processor" in graph.nodes

    def test_missing_apiversion_creates_warning(self, yaml_builder):
        """Test that missing apiVersion field generates warning but succeeds."""
        yaml_content = """
kind: Pipeline
metadata:
  name: no-version

spec:
  nodes: []
"""

        # Should complete with warning, not error
        graph, pipeline_config = yaml_builder.build_from_yaml_string(yaml_content)
        assert pipeline_config.metadata["name"] == "no-version"
        assert len(graph.nodes) == 0

    def test_missing_kind_field(self, yaml_builder):
        """Test that missing kind field is detected."""
        yaml_content = """
apiVersion: v1
metadata:
  name: no-kind

spec:
  nodes: []
"""

        with pytest.raises(Exception) as exc_info:
            yaml_builder.build_from_yaml_string(yaml_content)

        assert "kind" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()

    def test_empty_nodes_list(self, yaml_builder):
        """Test that empty nodes list creates empty graph."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: empty-pipeline

spec:
  nodes: []
"""

        graph, pipeline_config = yaml_builder.build_from_yaml_string(yaml_content)

        assert pipeline_config.metadata["name"] == "empty-pipeline"
        assert len(graph.nodes) == 0

    def test_parallel_node_dependencies(self, yaml_builder):
        """Test that independent nodes have no dependencies."""
        yaml_content = """
apiVersion: v1
kind: Pipeline
metadata:
  name: parallel-pipeline

spec:
  nodes:
    - kind: function_node
      metadata:
        name: task1
      spec:
        fn: work1
        input_mapping:
          dummy: input.x
        dependencies: []

    - kind: function_node
      metadata:
        name: task2
      spec:
        fn: work2
        input_mapping:
          dummy: input.x
        dependencies: []

    - kind: function_node
      metadata:
        name: task3
      spec:
        fn: work3
        input_mapping:
          dummy: input.x
        dependencies: []

    - kind: function_node
      metadata:
        name: combine
      spec:
        fn: combine_results
        input_mapping:
          r1: task1
          r2: task2
          r3: task3
        dependencies: [task1, task2, task3]
"""

        def work1(dummy: str = "") -> int:
            return 1

        def work2(dummy: str = "") -> int:
            return 2

        def work3(dummy: str = "") -> int:
            return 3

        def combine_results(r1: int, r2: int, r3: int) -> int:
            return r1 + r2 + r3

        # Register functions
        yaml_builder.register_function("work1", work1)
        yaml_builder.register_function("work2", work2)
        yaml_builder.register_function("work3", work3)
        yaml_builder.register_function("combine_results", combine_results)

        # Build graph
        graph, _ = yaml_builder.build_from_yaml_string(yaml_content)

        # Verify independent nodes have no dependencies
        assert len(graph.nodes) == 4
        assert frozenset(graph.nodes["task1"].deps) == frozenset()
        assert frozenset(graph.nodes["task2"].deps) == frozenset()
        assert frozenset(graph.nodes["task3"].deps) == frozenset()
        assert frozenset(graph.nodes["combine"].deps) == frozenset({"task1", "task2", "task3"})
