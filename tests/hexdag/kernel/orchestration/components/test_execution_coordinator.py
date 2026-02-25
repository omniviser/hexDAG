"""Tests for auto-wired input forwarding in ExecutionCoordinator.

When a single-dep node has ``in_model`` and no explicit ``input_mapping``,
fields with matching names are automatically extracted from the upstream dict.
"""

import pytest
from pydantic import BaseModel

from hexdag.kernel.domain.dag import NodeSpec
from hexdag.kernel.orchestration.components.execution_coordinator import ExecutionCoordinator


async def noop_fn(input_data):
    return input_data


class UpstreamOutput(BaseModel):
    name: str
    score: float
    extra: str


class DownstreamInput(BaseModel):
    name: str
    score: float


class TestAutoWireInputForwarding:
    """Tests for auto-wire logic in prepare_node_input."""

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_matching_fields_auto_extracted(self, coordinator):
        """Matching fields are extracted from upstream dict."""
        node = NodeSpec(
            "downstream", noop_fn, in_model=DownstreamInput, deps=frozenset({"upstream"})
        )
        results = {"upstream": {"name": "Alice", "score": 0.95, "extra": "ignored"}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        assert result == {"name": "Alice", "score": 0.95}
        assert "extra" not in result

    def test_no_in_model_full_passthrough(self, coordinator):
        """Without in_model, full upstream result is passed through (unchanged)."""
        node = NodeSpec("downstream", noop_fn, deps=frozenset({"upstream"}))
        upstream_data = {"name": "Alice", "score": 0.95, "extra": "kept"}
        results = {"upstream": upstream_data}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        assert result == upstream_data

    def test_explicit_input_mapping_takes_precedence(self, coordinator):
        """Explicit input_mapping overrides auto-wiring."""
        node = NodeSpec(
            "downstream",
            noop_fn,
            in_model=DownstreamInput,
            deps=frozenset({"upstream"}),
            params={"input_mapping": {"name": "upstream.extra"}},
        )
        results = {"upstream": {"name": "Alice", "score": 0.95, "extra": "mapped_value"}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # input_mapping was used, not auto-wire
        assert result["name"] == "mapped_value"

    def test_partial_field_match(self, coordinator):
        """Only matching fields are extracted; missing ones are not included."""

        class WantsThree(BaseModel):
            name: str
            score: float
            missing_field: str = "default"

        node = NodeSpec("downstream", noop_fn, in_model=WantsThree, deps=frozenset({"upstream"}))
        results = {"upstream": {"name": "Bob", "score": 0.8}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # Only the two matching fields are extracted
        assert result == {"name": "Bob", "score": 0.8}

    def test_non_dict_upstream_no_auto_wire(self, coordinator):
        """Non-dict upstream result skips auto-wiring."""
        node = NodeSpec(
            "downstream", noop_fn, in_model=DownstreamInput, deps=frozenset({"upstream"})
        )
        results = {"upstream": "just a string"}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # Pass-through unchanged since upstream is not a dict
        assert result == "just a string"

    def test_multi_dep_no_auto_wire(self, coordinator):
        """Multi-dep nodes don't get auto-wiring (only single-dep)."""
        node = NodeSpec("downstream", noop_fn, in_model=DownstreamInput, deps=frozenset({"a", "b"}))
        results = {"a": {"name": "Alice"}, "b": {"score": 0.9}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # Multi-dep returns namespace dict, no auto-wire
        assert result == {"a": {"name": "Alice"}, "b": {"score": 0.9}}

    def test_no_matching_fields_full_passthrough(self, coordinator):
        """When no fields match, the full upstream dict is passed through."""

        class UnrelatedInput(BaseModel):
            x: int
            y: int

        node = NodeSpec(
            "downstream", noop_fn, in_model=UnrelatedInput, deps=frozenset({"upstream"})
        )
        upstream_data = {"name": "Alice", "score": 0.95}
        results = {"upstream": upstream_data}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # No matching fields → unchanged
        assert result == upstream_data
