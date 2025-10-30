"""Tests for InputMapper component."""

import pytest

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.components.input_mapper import InputMapper


class TestInputMapper:
    """Test InputMapper component."""

    @pytest.fixture
    def mapper(self):
        """Create an InputMapper instance."""
        return InputMapper()

    def test_no_dependencies_returns_initial_input(self, mapper):
        """Test that nodes with no dependencies get the initial input."""
        node_spec = NodeSpec("start", lambda x: x)  # No dependencies
        node_results = {}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == "hello"

    def test_no_dependencies_with_existing_results(self, mapper):
        """Test that no-dep nodes get initial input even if results exist."""
        node_spec = NodeSpec("start", lambda x: x)
        node_results = {"other_node": "other_result"}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == "hello"

    def test_single_dependency_returns_dependency_result(self, mapper):
        """Test that single-dependency nodes get direct pass-through."""
        node_spec = NodeSpec("process", lambda x: x, deps={"start"})
        node_results = {"start": "HELLO"}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == "HELLO"

    def test_single_dependency_missing_returns_initial_input(self, mapper):
        """Test that missing dependency falls back to initial input."""
        node_spec = NodeSpec("process", lambda x: x, deps={"start"})
        node_results = {}  # Dependency not executed yet
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == "hello"

    def test_multiple_dependencies_returns_dict(self, mapper):
        """Test that multiple dependencies return a dict with node names as keys."""
        node_spec = NodeSpec("combine", lambda x: x, deps={"start", "process"})
        node_results = {"start": "HELLO", "process": "HELLO!"}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert isinstance(result, dict)
        assert result == {"start": "HELLO", "process": "HELLO!"}

    def test_multiple_dependencies_partial_results(self, mapper):
        """Test that only available dependencies are included in dict."""
        node_spec = NodeSpec("combine", lambda x: x, deps={"start", "process", "missing"})
        node_results = {"start": "HELLO", "process": "HELLO!"}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert isinstance(result, dict)
        assert result == {"start": "HELLO", "process": "HELLO!"}
        assert "missing" not in result

    def test_multiple_dependencies_no_results(self, mapper):
        """Test that multiple dependencies with no results returns empty dict."""
        node_spec = NodeSpec("combine", lambda x: x, deps={"start", "process"})
        node_results = {}
        initial_input = "hello"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert isinstance(result, dict)
        assert result == {}

    def test_preserves_node_name_keys(self, mapper):
        """Test that dependency dict preserves node names as keys."""
        node_spec = NodeSpec("report", lambda x: x, deps={"analyzer", "validator", "formatter"})
        node_results = {
            "analyzer": {"score": 0.95},
            "validator": {"valid": True},
            "formatter": {"formatted": "Report"},
        }
        initial_input = "data"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert "analyzer" in result
        assert "validator" in result
        assert "formatter" in result
        assert result["analyzer"] == {"score": 0.95}
        assert result["validator"] == {"valid": True}
        assert result["formatter"] == {"formatted": "Report"}

    def test_complex_initial_input(self, mapper):
        """Test with complex initial input (dict)."""
        node_spec = NodeSpec("start", lambda x: x)
        node_results = {}
        initial_input = {"key": "value", "nested": {"data": 123}}

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == {"key": "value", "nested": {"data": 123}}

    def test_complex_dependency_results(self, mapper):
        """Test with complex dependency results."""
        node_spec = NodeSpec("combine", lambda x: x, deps={"node1", "node2"})
        node_results = {"node1": {"data": [1, 2, 3]}, "node2": {"status": "complete", "count": 42}}
        initial_input = "ignored"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == {
            "node1": {"data": [1, 2, 3]},
            "node2": {"status": "complete", "count": 42},
        }

    def test_pipeline_flow_simulation(self, mapper):
        """Test simulating a full pipeline flow."""
        # Pipeline: start -> process -> analyze -> combine
        initial_input = "hello"
        node_results = {}

        # 1. Start node (no deps)
        start_spec = NodeSpec("start", lambda x: x.upper())
        start_input = mapper.prepare_node_input(start_spec, node_results, initial_input)
        assert start_input == "hello"
        node_results["start"] = "HELLO"

        # 2. Process node (depends on start)
        process_spec = NodeSpec("process", lambda x: x + "!", deps={"start"})
        process_input = mapper.prepare_node_input(process_spec, node_results, initial_input)
        assert process_input == "HELLO"
        node_results["process"] = "HELLO!"

        # 3. Analyze node (depends on process)
        analyze_spec = NodeSpec("analyze", lambda x: len(x), deps={"process"})
        analyze_input = mapper.prepare_node_input(analyze_spec, node_results, initial_input)
        assert analyze_input == "HELLO!"
        node_results["analyze"] = 6

        # 4. Combine node (depends on start and process)
        combine_spec = NodeSpec("combine", lambda x: x, deps={"start", "process", "analyze"})
        combine_input = mapper.prepare_node_input(combine_spec, node_results, initial_input)
        assert combine_input == {"start": "HELLO", "process": "HELLO!", "analyze": 6}

    def test_empty_deps_set(self, mapper):
        """Test with explicitly empty deps set."""
        node_spec = NodeSpec("start", lambda x: x, deps=set())
        node_results = {"other": "data"}
        initial_input = "input"

        result = mapper.prepare_node_input(node_spec, node_results, initial_input)

        assert result == "input"
