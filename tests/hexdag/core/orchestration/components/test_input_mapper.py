"""Tests for InputMapper component."""

import pytest

from hexdag.core.domain.dag import NodeSpec
from hexdag.core.orchestration.components.execution_coordinator import ExecutionCoordinator
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


class TestInputMappingWithExecutionCoordinator:
    """Tests for input_mapping functionality in ExecutionCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create an ExecutionCoordinator instance."""
        return ExecutionCoordinator()

    def test_input_mapping_with_dollar_input_syntax(self, coordinator):
        """Test $input.field syntax for accessing initial pipeline input."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"producer"},
            params={
                "input_mapping": {
                    "load_id": "$input.load_id",
                    "carrier_mc": "$input.carrier_mc",
                }
            },
        )
        initial_input = {"load_id": "LOAD123", "carrier_mc": "MC456", "extra": "ignored"}
        node_results = {"producer": {"result": "some_data"}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result == {"load_id": "LOAD123", "carrier_mc": "MC456"}

    def test_input_mapping_with_dependency_path(self, coordinator):
        """Test node_name.field syntax for accessing dependency output."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"analyzer"},
            params={
                "input_mapping": {
                    "analysis_result": "analyzer.output",
                    "score": "analyzer.metadata.score",
                }
            },
        )
        initial_input = {"original": "data"}
        node_results = {
            "analyzer": {
                "output": "Analysis complete",
                "metadata": {"score": 0.95, "confidence": "high"},
            }
        }

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["analysis_result"] == "Analysis complete"
        assert result["score"] == 0.95

    def test_input_mapping_mixed_sources(self, coordinator):
        """Test combining $input and dependency mappings."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"processor"},
            params={
                "input_mapping": {
                    "original_id": "$input.request_id",
                    "processed_data": "processor.result",
                }
            },
        )
        initial_input = {"request_id": "REQ001", "user": "test_user"}
        node_results = {"processor": {"result": {"status": "processed"}}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["original_id"] == "REQ001"
        assert result["processed_data"] == {"status": "processed"}

    def test_input_mapping_entire_initial_input(self, coordinator):
        """Test $input to reference entire initial input."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"producer"},
            params={
                "input_mapping": {
                    "full_context": "$input",
                }
            },
        )
        initial_input = {"key1": "value1", "key2": "value2"}
        node_results = {"producer": {"result": "data"}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["full_context"] == initial_input

    def test_input_mapping_nested_dependency_path(self, coordinator):
        """Test deeply nested path extraction from dependencies."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"deep_producer"},
            params={
                "input_mapping": {
                    "deep_value": "deep_producer.level1.level2.level3.value",
                }
            },
        )
        initial_input = {}
        node_results = {"deep_producer": {"level1": {"level2": {"level3": {"value": "found_it"}}}}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["deep_value"] == "found_it"

    def test_input_mapping_missing_path_returns_none(self, coordinator):
        """Test that missing paths result in None values."""
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"producer"},
            params={
                "input_mapping": {
                    "missing_field": "$input.nonexistent",
                    "also_missing": "producer.nonexistent.path",
                }
            },
        )
        initial_input = {"existing": "value"}
        node_results = {"producer": {"existing": "data"}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["missing_field"] is None
        assert result["also_missing"] is None

    def test_input_mapping_no_mapping_uses_standard_behavior(self, coordinator):
        """Test that nodes without input_mapping use standard dependency behavior."""
        # Single dependency - should pass through
        node_spec = NodeSpec(
            "consumer",
            lambda x: x,
            deps={"producer"},
        )
        initial_input = {"original": "data"}
        node_results = {"producer": {"result": "processed"}}

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result == {"result": "processed"}

    def test_input_mapping_from_multiple_dependencies(self, coordinator):
        """Test mapping from multiple different dependencies."""
        node_spec = NodeSpec(
            "merger",
            lambda x: x,
            deps={"analyzer", "validator", "enricher"},
            params={
                "input_mapping": {
                    "analysis": "analyzer.result",
                    "validation_status": "validator.is_valid",
                    "extra_data": "enricher.metadata",
                    "request_id": "$input.id",
                }
            },
        )
        initial_input = {"id": "REQ123"}
        node_results = {
            "analyzer": {"result": {"score": 0.9}},
            "validator": {"is_valid": True},
            "enricher": {"metadata": {"source": "external"}},
        }

        result = coordinator.prepare_node_input(node_spec, node_results, initial_input)

        assert result["analysis"] == {"score": 0.9}
        assert result["validation_status"] is True
        assert result["extra_data"] == {"source": "external"}
        assert result["request_id"] == "REQ123"

    def test_input_mapping_pipeline_simulation(self, coordinator):
        """Test realistic pipeline with input_mapping throughout."""
        # Simulating: request -> extract -> process -> validate -> respond
        initial_input = {
            "load_id": "LOAD001",
            "carrier_mc": "MC123",
            "rate": 1500.00,
        }
        node_results = {}

        # 1. Extract node - uses initial input via $input
        extract_spec = NodeSpec(
            "extract",
            lambda x: x,
            params={
                "input_mapping": {
                    "load_id": "$input.load_id",
                    "carrier_mc": "$input.carrier_mc",
                }
            },
        )
        extract_input = coordinator.prepare_node_input(extract_spec, node_results, initial_input)
        assert extract_input == {"load_id": "LOAD001", "carrier_mc": "MC123"}
        node_results["extract"] = {"load_data": {"id": "LOAD001"}, "carrier_data": {"mc": "MC123"}}

        # 2. Process node - uses extract output
        process_spec = NodeSpec(
            "process",
            lambda x: x,
            deps={"extract"},
            params={
                "input_mapping": {
                    "load": "extract.load_data",
                    "carrier": "extract.carrier_data",
                    "original_rate": "$input.rate",
                }
            },
        )
        process_input = coordinator.prepare_node_input(process_spec, node_results, initial_input)
        assert process_input["load"] == {"id": "LOAD001"}
        assert process_input["carrier"] == {"mc": "MC123"}
        assert process_input["original_rate"] == 1500.00
