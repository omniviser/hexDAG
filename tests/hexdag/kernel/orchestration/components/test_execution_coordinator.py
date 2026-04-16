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


class TestAdditiveInputMapping:
    """Tests for n8n-like additive input_mapping behavior.

    When input_mapping is present, the result now includes the full upstream
    namespace (node_results) PLUS the explicitly mapped fields overlaid on top.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_upstream_namespace_available_with_mapping(self, coordinator):
        """Upstream node results are available alongside mapped fields."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer", "scorer"}),
            params={"input_mapping": {"rate": "producer.rate"}},
        )
        node_results = {
            "producer": {"rate": 100, "name": "test"},
            "scorer": {"score": 0.95},
        }

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        # Explicit mapping works
        assert result["rate"] == 100
        # Upstream namespace also available
        assert result["producer"] == {"rate": 100, "name": "test"}
        assert result["scorer"] == {"score": 0.95}

    def test_explicit_mapping_overrides_upstream(self, coordinator):
        """Explicit input_mapping values override upstream namespace keys."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"producer": "producer.inner"}},
        )
        node_results = {"producer": {"inner": "extracted_value"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        # The explicit mapping overrides the namespace key
        assert result["producer"] == "extracted_value"

    def test_initial_input_available_as_input_key(self, coordinator):
        """Initial pipeline input is available as 'input' in the namespace."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"rate": "producer.rate"}},
        )
        initial_input = {"load_id": "LOAD123"}
        node_results = {"producer": {"rate": 100}}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["input"] == {"load_id": "LOAD123"}

    def test_missing_first_segment_raises_error(self, coordinator):
        """MISSING first path segment in input_mapping raises ValueError."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"bad": "nonexistent_node.field"}},
        )
        node_results = {"producer": {"data": "ok"}}

        with pytest.raises(ValueError, match="does not exist"):
            coordinator.prepare_node_input(node, node_results, initial_input={})

    def test_missing_deep_path_returns_none(self, coordinator):
        """Missing deep path (after first segment) returns None, not error."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"val": "producer.nonexistent_deep"}},
        )
        node_results = {"producer": {"data": "ok"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        # Deep path missing → None (optional field), not error
        assert result["val"] is None


class TestMissingSentinelInExpressions:
    """Tests for MISSING sentinel behavior in expression evaluation."""

    def test_missing_root_name_in_evaluate_expression(self):
        """evaluate_expression raises ExpressionError for missing root names."""
        from hexdag.kernel.expression_parser import ExpressionError, evaluate_expression

        with pytest.raises(ExpressionError, match="missing reference"):
            evaluate_expression("nonexistent_var", {"known": 42}, {})

    def test_coalesce_skips_missing(self):
        """coalesce() treats MISSING like None — skips to next arg."""
        from hexdag.kernel.expression_parser import evaluate_expression

        result = evaluate_expression(
            "coalesce(missing_var, fallback)",
            {"fallback": "ok"},
            {},
        )
        # missing_var resolves to MISSING → coalesce skips it → returns "ok"
        assert result == "ok"

    def test_default_handles_missing(self):
        """default() treats MISSING like None — returns the default."""
        from hexdag.kernel.expression_parser import evaluate_expression

        result = evaluate_expression(
            "default(missing_var, 42)",
            {},
            {},
        )
        assert result == 42

    def test_deep_path_none_not_missing(self):
        """Deep path missing returns None (not MISSING) — doesn't error."""
        from hexdag.kernel.expression_parser import evaluate_expression

        result = evaluate_expression(
            "default(data.nonexistent_field, 'fallback')",
            {"data": {"other": "value"}},
            {},
        )
        assert result == "fallback"

    def test_known_root_with_none_value(self):
        """A key that exists but is None passes through correctly."""
        from hexdag.kernel.expression_parser import evaluate_expression

        result = evaluate_expression("val", {"val": None}, {})
        assert result is None
