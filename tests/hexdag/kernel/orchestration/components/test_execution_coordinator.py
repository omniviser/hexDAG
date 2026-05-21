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


class TestBareNodeNameResolution:
    """Tests for resolving bare node names (no dot) from node_results."""

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_bare_name_resolves_from_node_results(self, coordinator):
        """input_mapping 'check: guardrail_check' resolves the whole node result."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"guardrail_check"}),
            params={"input_mapping": {"check": "guardrail_check"}},
        )
        node_results = {"guardrail_check": {"classification": "safe", "score": 0.95}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result["check"] == {"classification": "safe", "score": 0.95}

    def test_bare_name_scalar_node_result(self, coordinator):
        """Bare name resolves to scalar node result."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"counter"}),
            params={"input_mapping": {"count": "counter"}},
        )
        node_results = {"counter": 42}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result["count"] == 42

    def test_bare_name_not_in_results_falls_to_base_input(self, coordinator):
        """Bare name not in node_results falls back to base_input field extraction."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"val": "some_field"}},
        )
        node_results = {"producer": {"some_field": "hello"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result["val"] == "hello"


class TestQuotedStringLiteralResolution:
    """Tests for quoted string literals in input_mapping."""

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_single_quoted_string_literal(self, coordinator):
        """YAML "'approved'" resolves to string 'approved'."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"status": "'approved'"}},
        )
        node_results = {"producer": {"data": "x"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result["status"] == "approved"

    def test_is_expression_detects_quoted_string(self, coordinator):
        """_is_expression returns True for quoted string literals."""
        assert coordinator._is_expression("'approved'") is True
        assert coordinator._is_expression("'hello world'") is True
        assert coordinator._is_expression('"double quoted"') is True

    def test_is_expression_does_not_match_unquoted(self, coordinator):
        """_is_expression returns False for regular field names."""
        assert coordinator._is_expression("some_field") is False
        assert coordinator._is_expression("node.field") is False


class TestExpressionDetectionFalsePositives:
    """Regression tests for _is_expression false positives.

    Field paths containing substrings of allowed function names
    (e.g., 'my_len(data).result' matching 'len(') should NOT be
    classified as expressions.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_field_with_len_suffix_not_expression(self, coordinator):
        """'my_len' should not match allowed function 'len'."""
        assert coordinator._is_expression("my_len(data)") is False

    def test_field_with_coalesce_prefix_not_expression(self, coordinator):
        """'coalesce_results.field' should not match 'coalesce('."""
        assert coordinator._is_expression("coalesce_results.field") is False

    def test_actual_len_still_detected(self, coordinator):
        """Actual 'len(items)' is still detected as expression."""
        assert coordinator._is_expression("len(items)") is True

    def test_actual_coalesce_still_detected(self, coordinator):
        """Actual 'coalesce(a, b)' is still detected as expression."""
        assert coordinator._is_expression("coalesce(a, b)") is True


class TestUnknownNodeFallback:
    """Regression tests for unknown node.field resolution.

    When input_mapping references a node name not in node_results,
    the coordinator should return MISSING (raising an error) instead
    of silently extracting from base_input.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_unknown_node_raises_error(self, coordinator):
        """Referencing a non-existent node in input_mapping raises ValueError."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"val": "typo_node.field"}},
        )
        node_results = {"producer": {"data": "ok"}}

        with pytest.raises(ValueError, match="does not exist"):
            coordinator.prepare_node_input(node, node_results, initial_input={})

    def test_known_node_name_in_base_input_still_resolves(self, coordinator):
        """When node_name is in base_input (upstream dict), it still resolves."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"val": "producer.data"}},
        )
        node_results = {"producer": {"data": "ok"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result["val"] == "ok"


class TestCtxInExpressions:
    """Test that ctx is injected into expression evaluation."""

    def test_ctx_available_in_expression(self):
        """ctx fields are accessible in _evaluate_expression."""
        from hexdag.kernel.context import (
            ExecutionContext,
            clear_execution_context,
        )

        coordinator = ExecutionCoordinator()

        with ExecutionContext(
            run_id="test-run-123",
            pipeline_name="my-pipeline",
        ):
            result = coordinator._evaluate_expression(
                "ctx.run_id",
                base_input={},
                initial_input={},
                node_results={},
            )
            assert result == "test-run-123"

        clear_execution_context()

    def test_ctx_pipeline_name_in_expression(self):
        """ctx.pipeline_name resolves in expression evaluation."""
        from hexdag.kernel.context import (
            ExecutionContext,
            clear_execution_context,
        )

        coordinator = ExecutionCoordinator()

        with ExecutionContext(pipeline_name="order-processing"):
            result = coordinator._evaluate_expression(
                "ctx.pipeline_name == 'order-processing'",
                base_input={},
                initial_input={},
                node_results={},
            )
            assert result is True

        clear_execution_context()

    def test_ctx_coexists_with_node_results(self):
        """ctx doesn't interfere with node result access."""
        from hexdag.kernel.context import (
            ExecutionContext,
            clear_execution_context,
        )

        coordinator = ExecutionCoordinator()

        with ExecutionContext(run_id="run-1", pipeline_name="test"):
            result = coordinator._evaluate_expression(
                "analyzer.score",
                base_input={},
                initial_input={},
                node_results={"analyzer": {"score": 0.95}},
            )
            assert result == 0.95

        clear_execution_context()


class TestExpressionValuesInInputMapping:
    """Runtime tests for expression-valued input_mapping entries."""

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_arithmetic_expression(self, coordinator):
        """Arithmetic expression evaluates in input_mapping."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"total": "order.price * order.quantity"},
            initial_input={},
            node_results={"order": {"price": 10, "quantity": 5}},
        )
        assert result["total"] == 50

    def test_function_call_expression(self, coordinator):
        """Function call in input_mapping evaluates correctly."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"count": "len(analyzer.items)"},
            initial_input={},
            node_results={"analyzer": {"items": [1, 2, 3]}},
        )
        assert result["count"] == 3

    def test_comparison_expression(self, coordinator):
        """Comparison expression returns boolean."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"valid": "analyzer.score > 0.5"},
            initial_input={},
            node_results={"analyzer": {"score": 0.8}},
        )
        assert result["valid"] is True

    def test_mixed_node_and_input_expression(self, coordinator):
        """Expression mixing node ref and initial input."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"adjusted": "order.price * input.discount"},
            initial_input={"discount": 0.9},
            node_results={"order": {"price": 100}},
        )
        assert result["adjusted"] == pytest.approx(90.0)

    def test_coalesce_expression(self, coordinator):
        """coalesce() works in input_mapping."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"val": "coalesce(order.notes, 'none')"},
            initial_input={},
            node_results={"order": {"notes": None}},
        )
        assert result["val"] == "none"

    def test_string_concatenation_expression(self, coordinator):
        """String concatenation via + in input_mapping."""
        result = coordinator._apply_input_mapping(
            base_input={},
            input_mapping={"label": "'Order: ' + order.name"},
            initial_input={},
            node_results={"order": {"name": "ABC-123"}},
        )
        assert result["label"] == "Order: ABC-123"
