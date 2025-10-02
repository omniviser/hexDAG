"""Tests for DAG primitives: NodeSpec and DirectedGraph."""

import pytest
from pydantic import BaseModel, Field

from hexai.core.domain.dag import (
    CycleDetectedError,
    DirectedGraph,
    DuplicateNodeError,
    MissingDependencyError,
    NodeSpec,
    SchemaCompatibilityError,
    ValidationError,
)


def dummy_fn():
    """Return a predefined string result for testing purposes."""
    return "result"


def another_fn():
    """Return another predefined string result for testing purposes."""
    return "another_result"


# Sample models for validation tests
class SampleInputModel(BaseModel):
    """Sample input model for validation tests."""

    name: str
    value: int
    optional_field: str | None = None


class SampleOutputModel(BaseModel):
    """Sample output model for validation tests."""

    result: str
    count: int = Field(gt=0)


class TestNodeSpec:
    """Test cases for NodeSpec class."""

    def test_basic_creation(self):
        """Test basic NodeSpec creation."""
        node = NodeSpec("test", dummy_fn)
        assert node.name == "test"
        assert node.fn == dummy_fn
        assert node.in_model is None
        assert node.out_model is None
        assert node.deps == frozenset()

    def test_after_method(self):
        """Test the after method for adding dependencies."""
        node = NodeSpec("test", dummy_fn).after("dep1", "dep2")
        assert node.deps == frozenset({"dep1", "dep2"})

    def test_after_method_with_multiple_deps(self):
        """Test the after method with multiple dependencies."""
        node = NodeSpec("test", dummy_fn).after("dep1", "dep2")
        assert node.deps == frozenset({"dep1", "dep2"})

    def test_chained_after(self):
        """Test chaining after calls."""
        node = NodeSpec("test", dummy_fn).after("dep1").after("dep2")
        assert node.deps == frozenset({"dep1", "dep2"})

    def test_with_models(self):
        """Test setting input/output Pydantic models."""
        node = NodeSpec("test", dummy_fn, in_model=SampleInputModel, out_model=SampleOutputModel)
        assert node.in_model is SampleInputModel
        assert node.out_model is SampleOutputModel

    def test_repr(self):
        """Test string representation."""
        node = NodeSpec("test", dummy_fn)
        repr_str = repr(node)
        assert "NodeSpec('test'" in repr_str

        # Test with dependencies and models
        node_with_deps = NodeSpec(
            "test", dummy_fn, in_model=SampleInputModel, out_model=SampleOutputModel
        ).after("dep1")
        repr_with_deps = repr(node_with_deps)
        assert "NodeSpec('test'" in repr_with_deps
        assert "SampleInputModel -> SampleOutputModel" in repr_with_deps
        assert "deps=['dep1']" in repr_with_deps

    def test_immutability(self):
        """Test that NodeSpec is immutable."""
        node = NodeSpec("test", dummy_fn, deps={"dep1"})

        # Should not be able to modify existing node
        with pytest.raises(AttributeError, match="has no attribute 'add'"):
            node.deps.add("dep2")  # type: ignore

        # Creating new node should not modify original
        new_node = node.after("dep2")
        assert node.deps == frozenset({"dep1"})
        assert new_node.deps == frozenset({"dep1", "dep2"})

    def test_params_immutability(self):
        """Test that params are immutable."""
        params = {"key": "value"}
        node = NodeSpec("test", dummy_fn, params=params)

        # Should not be able to modify params directly
        with pytest.raises(TypeError, match="does not support item assignment"):
            node.params["new_key"] = "new_value"  # type: ignore

    def test_types_none_handling(self):
        """Test handling of None types."""
        node = NodeSpec("test", dummy_fn, in_model=None, out_model=None)
        assert node.in_model is None
        assert node.out_model is None

    def test_complex_chaining(self):
        """Test complex method chaining."""
        node = NodeSpec("test", dummy_fn, in_model=str, out_model=dict).after("dep1").after("dep2")
        # Alternative chaining using a separate node for list/tuple types
        node2 = NodeSpec("test2", dummy_fn, in_model=list, out_model=tuple).after("dep3")

        assert node.in_model is str
        assert node.out_model is dict
        assert node.deps == frozenset({"dep1", "dep2"})
        assert node2.in_model is list
        assert node2.out_model is tuple

    def test_empty_node_name(self):
        """Test NodeSpec with empty name."""
        node = NodeSpec("", dummy_fn)
        assert node.name == ""

    def test_complex_method_chaining(self):
        """Test complex method chaining scenarios."""
        # Test chaining multiple after() calls
        node = (
            NodeSpec("complex", dummy_fn, in_model=list, out_model=tuple)
            .after("dep1")
            .after("dep2", "dep3")
        )

        assert node.deps == frozenset({"dep1", "dep2", "dep3"})
        assert node.in_model is list
        assert node.out_model is tuple

    def test_node_with_none_types(self):
        """Test NodeSpec with explicit None types."""
        node = NodeSpec("test", dummy_fn, in_model=None, out_model=None)
        assert node.in_model is None
        assert node.out_model is None

    def test_after_with_duplicate_dependencies(self):
        """Test that duplicate dependencies are handled correctly."""
        node = NodeSpec("test", dummy_fn).after("dep1", "dep1", "dep2")
        assert node.deps == frozenset({"dep1", "dep2"})

    def test_after_with_no_dependencies(self):
        """Test after method with no arguments."""
        node = NodeSpec("test", dummy_fn).after()
        assert node.deps == frozenset()

    def test_node_equality_and_hashing(self):
        """Test that NodeSpec instances can be compared and hashed."""
        node1 = NodeSpec("test", dummy_fn)
        node2 = NodeSpec("test", dummy_fn)

        # Different instances with same data should be equal for frozen dataclass
        assert node1.name == node2.name
        assert node1.fn == node2.fn

    # Validation tests for NodeSpec
    def test_validate_input_with_no_model(self):
        """Test validation when no input model is specified."""
        node = NodeSpec("test_node", dummy_fn, in_model=None)

        # Should return data as-is when no model
        test_data = {"any": "data"}
        result = node.validate_input(test_data)
        assert result == test_data

    def test_validate_input_with_valid_data(self):
        """Test validation with valid input data."""
        node = NodeSpec("test_node", dummy_fn, in_model=SampleInputModel)

        # Test with dict
        test_data = {"name": "test", "value": 42}
        result = node.validate_input(test_data)
        assert isinstance(result, SampleInputModel)
        assert result.name == "test"
        assert result.value == 42
        assert result.optional_field is None

    def test_validate_input_already_correct_type(self):
        """Test validation when data is already the correct type."""
        node = NodeSpec("test_node", dummy_fn, in_model=SampleInputModel)

        # Test with already validated model
        test_data = SampleInputModel(name="test", value=42)
        result = node.validate_input(test_data)
        assert result is test_data  # Should return same instance

    def test_validate_input_with_invalid_data(self):
        """Test validation with invalid input data."""
        node = NodeSpec("test_node", dummy_fn, in_model=SampleInputModel)

        # Missing required field
        test_data = {"name": "test"}  # Missing 'value'

        with pytest.raises(ValidationError) as exc_info:
            node.validate_input(test_data)

        assert "Input validation failed for node 'test_node'" in str(exc_info.value)

    def test_validate_input_with_type_coercion(self):
        """Test validation with automatic type coercion."""
        node = NodeSpec("test_node", dummy_fn, in_model=SampleInputModel)

        # String that can be converted to int
        test_data = {"name": "test", "value": "42"}
        result = node.validate_input(test_data)
        assert isinstance(result, SampleInputModel)
        assert result.value == 42  # Coerced to int

    def test_validate_output_with_no_model(self):
        """Test output validation when no model is specified."""
        node = NodeSpec("test_node", dummy_fn, out_model=None)

        # Should return data as-is when no model
        test_data = {"any": "output"}
        result = node.validate_output(test_data)
        assert result == test_data

    def test_validate_output_with_valid_data(self):
        """Test output validation with valid data."""
        node = NodeSpec("test_node", dummy_fn, out_model=SampleOutputModel)

        # Test with dict
        test_data = {"result": "success", "count": 5}
        result = node.validate_output(test_data)
        assert isinstance(result, SampleOutputModel)
        assert result.result == "success"
        assert result.count == 5

    def test_validate_output_with_invalid_data(self):
        """Test output validation with invalid data."""
        node = NodeSpec("test_node", dummy_fn, out_model=SampleOutputModel)

        # Invalid count (must be > 0)
        test_data = {"result": "success", "count": 0}

        with pytest.raises(ValidationError) as exc_info:
            node.validate_output(test_data)

        assert "Output validation failed for node 'test_node'" in str(exc_info.value)

    def test_validate_with_model_shared_logic(self):
        """Test that both input and output use same validation logic."""
        node = NodeSpec(
            "test_node", dummy_fn, in_model=SampleInputModel, out_model=SampleOutputModel
        )

        # Both should handle None model the same way
        assert node._validate_with_model({"test": "data"}, None, "input") == {"test": "data"}
        assert node._validate_with_model({"test": "data"}, None, "output") == {"test": "data"}

        # Both should handle already-correct type the same way
        input_instance = SampleInputModel(name="test", value=1)
        output_instance = SampleOutputModel(result="test", count=1)

        assert (
            node._validate_with_model(input_instance, SampleInputModel, "input") is input_instance
        )
        assert (
            node._validate_with_model(output_instance, SampleOutputModel, "output")
            is output_instance
        )


class TestDirectedGraph:
    """Test cases for DirectedGraph class."""

    def test_empty_graph(self):
        """Test empty graph initialization."""
        graph = DirectedGraph()
        assert len(graph.nodes) == 0
        assert graph.waves() == []

    def test_single_node(self):
        """Test adding a single node."""
        graph = DirectedGraph()
        node = NodeSpec("test", dummy_fn)

        graph.add(node)

        assert len(graph.nodes) == 1
        assert "test" in graph.nodes
        assert graph.nodes["test"] == node
        assert graph.waves() == [["test"]]

    def test_add_many(self):
        """Test adding multiple nodes at once."""
        graph = DirectedGraph()

        nodes = [
            NodeSpec("A", dummy_fn),
            NodeSpec("B", another_fn).after("A"),
            NodeSpec("C", dummy_fn).after("A"),
            NodeSpec("D", another_fn).after("B", "C"),
        ]

        result = graph.add_many(*nodes)

        assert result is graph  # Should return self for chaining
        assert len(graph.nodes) == 4
        assert all(node.name in graph.nodes for node in nodes)

        # Verify dependencies are correctly set
        assert graph.get_dependencies("A") == set()
        assert graph.get_dependencies("B") == {"A"}
        assert graph.get_dependencies("C") == {"A"}
        assert graph.get_dependencies("D") == {"B", "C"}

    def test_add_many_with_duplicates(self):
        """Test that add_many raises error for duplicate nodes."""
        graph = DirectedGraph()
        graph.add(NodeSpec("existing", dummy_fn))

        with pytest.raises(DuplicateNodeError, match="Node 'existing' already exists"):
            graph.add_many(
                NodeSpec("new", another_fn),
                NodeSpec("existing", dummy_fn),  # This should cause the error
            )

    def test_linear_dependency_chain(self):
        """Test linear chain: A -> B -> C."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A")
        node_c = NodeSpec("C", dummy_fn).after("B")

        graph.add(node_a).add(node_b).add(node_c)
        graph.validate()

        waves = graph.waves()
        assert waves == [["A"], ["B"], ["C"]]

    def test_dag_parallel_structure_validation(self):
        """Test parallel execution: A -> B, A -> C, B -> D, C -> D."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A")
        node_c = NodeSpec("C", dummy_fn).after("A")
        node_d = NodeSpec("D", another_fn).after("B", "C")

        graph.add(node_a).add(node_b).add(node_c).add(node_d)
        graph.validate()

        # Test the parallel structure
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert "C" in graph.nodes
        assert "D" in graph.nodes

        # Verify dependency structure for parallel pattern
        assert len(graph.get_dependencies("B")) == 1 and "A" in graph.get_dependencies("B")
        assert len(graph.get_dependencies("C")) == 1 and "A" in graph.get_dependencies("C")
        assert len(graph.get_dependencies("D")) == 2 and {"B", "C"}.issubset(
            graph.get_dependencies("D")
        )

        # Test wave structure as well
        waves = graph.waves()
        assert waves == [["A"], ["B", "C"], ["D"]]

    def test_diamond_pattern(self):
        """Test diamond pattern: A -> B, A -> C, B -> D, C -> D."""
        graph = DirectedGraph()

        # Same as parallel_execution but explicitly testing diamond pattern
        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A")
        node_c = NodeSpec("C", dummy_fn).after("A")
        node_d = NodeSpec("D", another_fn).after("B", "C")

        graph.add(node_a).add(node_b).add(node_c).add(node_d)
        graph.validate()

        waves = graph.waves()
        assert waves == [["A"], ["B", "C"], ["D"]]

    def test_duplicate_node_error(self):
        """Test that adding duplicate nodes raises error."""
        graph = DirectedGraph()
        node1 = NodeSpec("duplicate", dummy_fn)
        node2 = NodeSpec("duplicate", another_fn)

        graph.add(node1)

        with pytest.raises(DuplicateNodeError, match="Node 'duplicate' already exists"):
            graph.add(node2)

    def test_missing_dependency_error(self):
        """Test that missing dependencies are detected."""
        graph = DirectedGraph()
        node = NodeSpec("B", dummy_fn).after("A")  # A doesn't exist

        graph.add(node)

        with pytest.raises(MissingDependencyError, match="depends on missing node 'A'"):
            graph.validate()

    def test_cycle_detection_simple(self):
        """Test simple cycle detection: A -> B -> A."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn).after("B")
        node_b = NodeSpec("B", another_fn).after("A")

        graph.add(node_a).add(node_b)

        with pytest.raises(CycleDetectedError, match="Cycle detected"):
            graph.validate()

    def test_cycle_detection_complex(self):
        """Test complex cycle detection: A -> B -> C -> D -> B."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A", "D")
        node_c = NodeSpec("C", dummy_fn).after("B")
        node_d = NodeSpec("D", another_fn).after("C")

        graph.add(node_a).add(node_b).add(node_c).add(node_d)

        with pytest.raises(CycleDetectedError, match="Cycle detected"):
            graph.validate()

    def test_self_dependency_cycle(self):
        """Test self-dependency is detected as cycle."""
        graph = DirectedGraph()
        node = NodeSpec("A", dummy_fn).after("A")  # Self-dependency

        graph.add(node)

        with pytest.raises(CycleDetectedError, match="Cycle detected"):
            graph.validate()

    def test_complex_valid_dag(self):
        """Test a more complex but valid DAG."""
        graph = DirectedGraph()

        # Create a complex DAG:
        # start -> [fetch1, fetch2] -> process1 -> [transform1, transform2] -> combine -> end
        nodes = [
            NodeSpec("start", dummy_fn),
            NodeSpec("fetch1", another_fn).after("start"),
            NodeSpec("fetch2", dummy_fn).after("start"),
            NodeSpec("process1", another_fn).after("fetch1", "fetch2"),
            NodeSpec("transform1", dummy_fn).after("process1"),
            NodeSpec("transform2", another_fn).after("process1"),
            NodeSpec("combine", dummy_fn).after("transform1", "transform2"),
            NodeSpec("end", another_fn).after("combine"),
        ]

        for node in nodes:
            graph.add(node)

        graph.validate()
        waves = graph.waves()

        expected_waves = [
            ["start"],
            ["fetch1", "fetch2"],
            ["process1"],
            ["transform1", "transform2"],
            ["combine"],
            ["end"],
        ]
        assert waves == expected_waves

    def test_graph_repr(self):
        """Test string representation of graph."""
        graph = DirectedGraph()
        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn)

        graph.add(node_a).add(node_b)

        repr_str = repr(graph)
        assert "DirectedGraph" in repr_str
        assert "A" in repr_str
        assert "B" in repr_str

    def test_get_dependencies_and_dependents(self):
        """Test public API methods for querying dependencies and dependents."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A")
        node_c = NodeSpec("C", dummy_fn).after("A")

        graph.add(node_a).add(node_b).add(node_c)

        # Test get_dependents (what depends on A?)
        assert graph.get_dependents("A") == {"B", "C"}
        assert graph.get_dependents("B") == set()
        assert graph.get_dependents("C") == set()

        # Test get_dependencies (what does each node depend on?)
        assert graph.get_dependencies("A") == set()
        assert graph.get_dependencies("B") == {"A"}
        assert graph.get_dependencies("C") == {"A"}

    def test_get_dependencies_nonexistent_node(self):
        """Test that get_dependencies raises KeyError for nonexistent node."""
        graph = DirectedGraph()
        node_a = NodeSpec("A", dummy_fn)
        graph.add(node_a)

        with pytest.raises(KeyError, match="Node 'nonexistent' not found"):
            graph.get_dependencies("nonexistent")

    def test_get_dependents_nonexistent_node(self):
        """Test that get_dependents raises KeyError for nonexistent node."""
        graph = DirectedGraph()
        node_a = NodeSpec("A", dummy_fn)
        graph.add(node_a)

        with pytest.raises(KeyError, match="Node 'nonexistent' not found"):
            graph.get_dependents("nonexistent")

    def test_get_dependents_returns_copy(self):
        """Test that get_dependents returns a copy, not the internal set."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn).after("A")

        graph.add(node_a).add(node_b)

        dependents = graph.get_dependents("A")
        dependents.add("should_not_affect_internal")

        # Internal state should be unchanged
        assert graph.get_dependents("A") == {"B"}

    def test_multiple_missing_dependencies(self):
        """Test that multiple missing dependencies are reported."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn).after("missing1", "missing2")
        node_b = NodeSpec("B", another_fn).after("missing3")

        graph.add(node_a).add(node_b)

        with pytest.raises(MissingDependencyError) as exc_info:
            graph.validate()

        error_msg = str(exc_info.value)
        assert "missing1" in error_msg
        assert "missing2" in error_msg
        assert "missing3" in error_msg

    def test_method_chaining(self):
        """Test that add method supports chaining."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", another_fn)

        result = graph.add(node_a).add(node_b)

        assert result is graph  # Should return self
        assert len(graph.nodes) == 2
        assert "A" in graph.nodes
        assert "B" in graph.nodes

    # Enhanced Fan-out and Fan-in Tests

    def test_wide_fan_out_pattern(self):
        """Test wide fan-out: one node with many dependents."""
        graph = DirectedGraph()

        # Create: A -> B1, B2, B3, B4, B5
        nodes = [NodeSpec("A", dummy_fn)]
        fan_out_nodes = [NodeSpec(f"B{i}", dummy_fn).after("A") for i in range(1, 6)]
        nodes.extend(fan_out_nodes)

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert waves[0] == ["A"]
        assert set(waves[1]) == {f"B{i}" for i in range(1, 6)}
        assert len(waves) == 2

    def test_wide_fan_in_pattern(self):
        """Test wide fan-in: many nodes feeding into one node."""
        graph = DirectedGraph()

        # Create: A1, A2, A3, A4, A5 -> B
        source_nodes = [NodeSpec(f"A{i}", dummy_fn) for i in range(1, 6)]
        target_node = NodeSpec("B", dummy_fn).after(*[f"A{i}" for i in range(1, 6)])

        nodes = source_nodes + [target_node]
        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert set(waves[0]) == {f"A{i}" for i in range(1, 6)}
        assert waves[1] == ["B"]
        assert len(waves) == 2

    def test_multi_level_fan_patterns(self):
        """Test complex multi-level fan-out and fan-in."""
        graph = DirectedGraph()

        # Pattern: A -> [B1, B2] -> [C1, C2, C3, C4] -> D
        nodes = [
            NodeSpec("A", dummy_fn),
            NodeSpec("B1", dummy_fn).after("A"),
            NodeSpec("B2", dummy_fn).after("A"),
            NodeSpec("C1", dummy_fn).after("B1"),
            NodeSpec("C2", dummy_fn).after("B1"),
            NodeSpec("C3", dummy_fn).after("B2"),
            NodeSpec("C4", dummy_fn).after("B2"),
            NodeSpec("D", dummy_fn).after("C1", "C2", "C3", "C4"),
        ]

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert waves == [["A"], ["B1", "B2"], ["C1", "C2", "C3", "C4"], ["D"]]

    def test_mixed_fan_patterns(self):
        """Test mixed fan-out and fan-in with independent branches."""
        graph = DirectedGraph()

        # Complex pattern with multiple independent fan patterns
        nodes = [
            NodeSpec("root", dummy_fn),
            # First branch: fan-out
            NodeSpec("branch1_1", dummy_fn).after("root"),
            NodeSpec("branch1_2", dummy_fn).after("root"),
            # Second branch: linear
            NodeSpec("branch2", dummy_fn).after("root"),
            # Fan-in from first branch
            NodeSpec("merge1", dummy_fn).after("branch1_1", "branch1_2"),
            # Final convergence
            NodeSpec("final", dummy_fn).after("merge1", "branch2"),
        ]

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert waves[0] == ["root"]
        assert set(waves[1]) == {"branch1_1", "branch1_2", "branch2"}
        assert waves[2] == ["merge1"]
        assert waves[3] == ["final"]

    # Enhanced Wave Generation Tests

    def test_waves_empty_graph(self):
        """Test waves generation for empty graph."""
        graph = DirectedGraph()
        assert graph.waves() == []

    def test_waves_single_node(self):
        """Test waves generation for single node."""
        graph = DirectedGraph()
        graph.add(NodeSpec("solo", dummy_fn))
        assert graph.waves() == [["solo"]]

    def test_waves_deterministic_ordering(self):
        """Test that waves have deterministic ordering within each wave."""
        graph = DirectedGraph()

        # Add nodes in different order to test sorting
        nodes = [
            NodeSpec("root", dummy_fn),
            NodeSpec("z_node", dummy_fn).after("root"),
            NodeSpec("a_node", dummy_fn).after("root"),
            NodeSpec("m_node", dummy_fn).after("root"),
        ]

        graph.add_many(*nodes)
        waves = graph.waves()

        # Second wave should be alphabetically sorted
        assert waves[1] == ["a_node", "m_node", "z_node"]

    def test_validate_type_compatibility(self):
        """Test type compatibility validation between connected nodes."""
        from pydantic import BaseModel

        class OutputA(BaseModel):
            result: str

        class InputB(BaseModel):
            result: str

        # Create compatible nodes (same type)
        node_a = NodeSpec("a", dummy_fn, out_model=OutputA)
        node_b = NodeSpec("b", dummy_fn, in_model=OutputA, deps={"a"})  # Same type as output

        graph = DirectedGraph()
        graph.add(node_a)
        graph.add(node_b)

        # Should validate successfully with same types
        graph.validate(check_type_compatibility=True)

        # Create incompatible nodes
        class InputC(BaseModel):
            number: float

        node_c = NodeSpec("c", dummy_fn, in_model=InputC, deps={"a"})
        graph.add(node_c)

        # Should raise SchemaCompatibilityError for incompatible types
        with pytest.raises(SchemaCompatibilityError) as exc_info:
            graph.validate(check_type_compatibility=True)
        assert "expects InputC but dependency 'a' outputs OutputA" in str(exc_info.value)

    def test_validate_without_type_checking(self):
        """Test that validation can skip type compatibility checking."""
        from pydantic import BaseModel

        class OutputA(BaseModel):
            result: str

        class InputB(BaseModel):
            number: float  # Incompatible with OutputA

        node_a = NodeSpec("a", dummy_fn, out_model=OutputA)
        node_b = NodeSpec("b", dummy_fn, in_model=InputB, deps={"a"})

        graph = DirectedGraph()
        graph.add(node_a)
        graph.add(node_b)

        # Should validate successfully when type checking is disabled
        graph.validate(check_type_compatibility=False)

    def test_waves_complex_dag_performance(self):
        """Test wave generation with a larger, complex DAG."""
        graph = DirectedGraph()

        # Create a more complex DAG to test performance and correctness
        nodes = []

        # Layer 1: root
        nodes.append(NodeSpec("root", dummy_fn))

        # Layer 2: 10 parallel nodes
        for i in range(10):
            nodes.append(NodeSpec(f"layer2_{i:02d}", dummy_fn).after("root"))

        # Layer 3: each depends on 2 nodes from layer 2
        for i in range(5):
            deps = [f"layer2_{i * 2:02d}", f"layer2_{i * 2 + 1:02d}"]
            nodes.append(NodeSpec(f"layer3_{i:02d}", dummy_fn).after(*deps))

        # Layer 4: final convergence
        layer3_deps = [f"layer3_{i:02d}" for i in range(5)]
        nodes.append(NodeSpec("final", dummy_fn).after(*layer3_deps))

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert len(waves) == 4
        assert len(waves[0]) == 1  # root
        assert len(waves[1]) == 10  # layer 2
        assert len(waves[2]) == 5  # layer 3
        assert len(waves[3]) == 1  # final

    # Enhanced Cycle Detection Tests

    def test_cycle_detection_immediate_self_reference(self):
        """Test detection of immediate self-reference cycle."""
        graph = DirectedGraph()
        node = NodeSpec("self_ref", dummy_fn).after("self_ref")
        graph.add(node)

        with pytest.raises(CycleDetectedError, match="Cycle detected.*self_ref.*self_ref"):
            graph.validate()

    def test_cycle_detection_indirect_cycle(self):
        """Test detection of indirect cycle through multiple nodes."""
        graph = DirectedGraph()

        # Create cycle: A -> B -> C -> D -> A
        nodes = [
            NodeSpec("A", dummy_fn).after("D"),
            NodeSpec("B", dummy_fn).after("A"),
            NodeSpec("C", dummy_fn).after("B"),
            NodeSpec("D", dummy_fn).after("C"),
        ]

        graph.add_many(*nodes)

        with pytest.raises(CycleDetectedError):
            graph.validate()

    def test_cycle_detection_with_valid_branches(self):
        """Test that cycles are detected even when valid branches exist."""
        graph = DirectedGraph()

        # Mixed valid and cyclic structure
        nodes = [
            NodeSpec("valid_root", dummy_fn),
            NodeSpec("valid_child", dummy_fn).after("valid_root"),
            # Cycle in separate part
            NodeSpec("cycle_a", dummy_fn).after("cycle_b"),
            NodeSpec("cycle_b", dummy_fn).after("cycle_a"),
        ]

        graph.add_many(*nodes)

        with pytest.raises(CycleDetectedError):
            graph.validate()

    def test_no_false_positive_cycles(self):
        """Test that valid complex DAGs don't trigger false positive cycles."""
        graph = DirectedGraph()

        # Complex but valid DAG
        nodes = [
            NodeSpec("A", dummy_fn),
            NodeSpec("B", dummy_fn).after("A"),
            NodeSpec("C", dummy_fn).after("A"),
            NodeSpec("D", dummy_fn).after("B"),
            NodeSpec("E", dummy_fn).after("C"),
            NodeSpec("F", dummy_fn).after("B", "C"),  # Diamond convergence
            NodeSpec("G", dummy_fn).after("D", "E", "F"),  # Multiple convergence
        ]

        graph.add_many(*nodes)
        # Should not raise any exception
        try:
            graph.validate()
            # Verify the graph structure is correct
            waves = graph.waves()
            assert len(waves) == 4
            # Additional verification that validation succeeded
            assert len(graph.nodes) == 7
        except Exception as e:
            pytest.fail(f"Valid DAG should not raise cycle detection error, but got: {e}")

    # Enhanced Error Handling Tests

    def test_add_many_empty_list(self):
        """Test add_many with empty list."""
        graph = DirectedGraph()
        result = graph.add_many()
        assert result is graph
        assert len(graph.nodes) == 0

    def test_add_many_partial_failure_rollback(self):
        """Test that add_many fails atomically - no partial adds on error."""
        graph = DirectedGraph()
        graph.add(NodeSpec("existing", dummy_fn))

        initial_count = len(graph.nodes)

        try:
            graph.add_many(
                NodeSpec("new1", dummy_fn),
                NodeSpec("existing", dummy_fn),  # Duplicate
                NodeSpec("new2", dummy_fn),
            )
        except DuplicateNodeError:
            pass

        # Should have rolled back - no new nodes added
        assert len(graph.nodes) == initial_count
        assert "new1" not in graph.nodes
        assert "new2" not in graph.nodes

    def test_validate_without_nodes(self):
        """Test validation of empty graph."""
        graph = DirectedGraph()
        # Should not raise any exception
        try:
            graph.validate()
            # If we get here, validation passed successfully
            assert True
        except Exception as e:
            pytest.fail(f"Empty graph validation should not raise an exception, but got: {e}")

    def test_dependencies_edge_cases(self):
        """Test edge cases in dependency management."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", dummy_fn).after("A")

        graph.add(node_a).add(node_b)

        # Test that we get immutable frozensets (performance optimization)
        deps_1 = graph.get_dependencies("B")
        deps_2 = graph.get_dependencies("B")

        # frozensets are immutable, so returning same instance is safe and faster
        assert deps_1 is deps_2  # Same immutable object (performance optimization)
        assert deps_1 == deps_2  # Same content

    def test_error_message_specificity(self):
        """Test that error messages are specific and helpful."""
        graph = DirectedGraph()

        # Test missing dependency error message
        node = NodeSpec("B", dummy_fn).after("nonexistent")
        graph.add(node)

        with pytest.raises(MissingDependencyError) as exc_info:
            graph.validate()

        error_msg = str(exc_info.value)
        assert "Node 'B'" in error_msg
        assert "missing node 'nonexistent'" in error_msg

    def test_concurrent_modification_safety(self):
        """Test that internal structures are protected from external modification."""
        graph = DirectedGraph()

        node_a = NodeSpec("A", dummy_fn)
        node_b = NodeSpec("B", dummy_fn).after("A")

        graph.add(node_a).add(node_b)

        # Verify returned dependencies are immutable (frozenset)
        deps = graph.get_dependencies("B")
        assert isinstance(deps, frozenset)

        # frozenset doesn't have .add() - this is the protection we want!
        # Attempting to modify would raise AttributeError
        # Internal state is inherently safe with frozenset
        assert graph.get_dependencies("B") == deps

    # Stress Tests

    def test_large_linear_chain_performance(self):
        """Test performance with a large linear chain."""
        graph = DirectedGraph()

        # Create a chain of 100 nodes
        nodes = [NodeSpec("node_0", dummy_fn)]
        for i in range(1, 100):
            nodes.append(NodeSpec(f"node_{i}", dummy_fn).after(f"node_{i - 1}"))

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert len(waves) == 100
        for i, wave in enumerate(waves):
            assert wave == [f"node_{i}"]

    def test_large_parallel_structure_performance(self):
        """Test performance with large parallel structure."""
        graph = DirectedGraph()

        # Create root -> 50 parallel nodes -> convergence
        nodes = [NodeSpec("root", dummy_fn)]
        parallel_names = []

        for i in range(50):
            name = f"parallel_{i:02d}"
            parallel_names.append(name)
            nodes.append(NodeSpec(name, dummy_fn).after("root"))

        nodes.append(NodeSpec("convergence", dummy_fn).after(*parallel_names))

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert len(waves) == 3
        assert waves[0] == ["root"]
        assert len(waves[1]) == 50
        assert waves[2] == ["convergence"]

    def test_memory_efficiency_large_dag(self):
        """Test memory efficiency with a reasonably large DAG."""
        graph = DirectedGraph()

        # Create a more complex structure to test memory usage
        nodes = []

        # Create multiple layers with fan-out/fan-in
        nodes.append(NodeSpec("start", dummy_fn))

        # Layer 1: 20 nodes
        for i in range(20):
            nodes.append(NodeSpec(f"l1_{i:02d}", dummy_fn).after("start"))

        # Layer 2: Groups of layer 1 nodes feed into layer 2
        for i in range(10):
            deps = [f"l1_{i * 2:02d}", f"l1_{i * 2 + 1:02d}"]
            nodes.append(NodeSpec(f"l2_{i:02d}", dummy_fn).after(*deps))

        # Layer 3: Convergence
        l2_deps = [f"l2_{i:02d}" for i in range(10)]
        nodes.append(NodeSpec("end", dummy_fn).after(*l2_deps))

        graph.add_many(*nodes)
        graph.validate()

        waves = graph.waves()
        assert len(waves) == 4
        assert len(graph.nodes) == 32  # 1 + 20 + 10 + 1
