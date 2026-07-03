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


class TestAutoInferParams:
    """Tests for accepted_params-based auto-inference in prepare_node_input.

    When a node declares ``accepted_params``, the coordinator searches
    upstream result dicts for matching parameter names — same mechanism
    as service_call_node, but unified in the coordinator.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_infers_from_upstream_dicts(self, coordinator):
        """Params found in upstream result dicts are auto-mapped."""
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"fetch_order", "compute"}),
            accepted_params=frozenset({"order_id", "amount"}),
        )
        results = {
            "fetch_order": {"order_id": "ORD-1", "customer": "Alice"},
            "compute": {"amount": 99.5, "tax": 5.0},
        }

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        assert result == {"order_id": "ORD-1", "amount": 99.5}

    def test_ambiguous_param_skipped(self, coordinator):
        """Param found in multiple upstream sources is skipped."""
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"source_a", "source_b"}),
            accepted_params=frozenset({"order_id", "amount"}),
        )
        results = {
            "source_a": {"order_id": "A1", "amount": 10.0},
            "source_b": {"order_id": "B1"},
        }

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # order_id is ambiguous (in both sources) → skipped
        # amount is only in source_a → mapped
        assert result == {"amount": 10.0}

    def test_top_level_keys_take_priority(self, coordinator):
        """Top-level keys matching accepted_params are used directly."""
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"upstream"}),
            accepted_params=frozenset({"order_id", "amount"}),
        )
        results = {
            "upstream": {"order_id": "TOP", "amount": 50.0, "extra": "ignored"},
        }

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # Single dep → base_input is the upstream dict itself, not nested
        # So order_id and amount are top-level matches
        assert result["order_id"] == "TOP"
        assert result["amount"] == 50.0
        assert "extra" not in result

    def test_explicit_input_mapping_overrides_accepted_params(self, coordinator):
        """input_mapping takes precedence over accepted_params."""
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"upstream"}),
            accepted_params=frozenset({"order_id"}),
            params={"input_mapping": {"order_id": "upstream.custom_field"}},
        )
        results = {"upstream": {"order_id": "DIRECT", "custom_field": "MAPPED"}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # input_mapping was applied, not auto-infer
        assert result["order_id"] == "MAPPED"

    def test_no_accepted_params_no_inference(self, coordinator):
        """Without accepted_params, no auto-inference happens."""
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"a", "b"}),
        )
        results = {"a": {"x": 1}, "b": {"y": 2}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # Raw namespace dict, no filtering
        assert result == {"a": {"x": 1}, "b": {"y": 2}}

    def test_accepted_params_supersedes_in_model(self, coordinator):
        """accepted_params takes priority over in_model auto-wire."""
        node = NodeSpec(
            "process",
            noop_fn,
            in_model=DownstreamInput,  # expects name, score
            deps=frozenset({"upstream"}),
            accepted_params=frozenset({"name"}),  # only wants name
        )
        results = {"upstream": {"name": "Alice", "score": 0.95, "extra": "x"}}

        result = coordinator.prepare_node_input(node, results, initial_input=None)

        # accepted_params wins: only "name" extracted, not "score"
        assert result == {"name": "Alice"}

    def test_infers_from_initial_input(self, coordinator):
        """Params not in upstream dicts arrive via the ambient input tier.

        Auto-infer no longer searches the pipeline input; instead every
        top-level input field is filled ambiently after inference. Fields
        the fn doesn't declare (``extra``) are present in the prepared
        input — the fn boundary (signature filter) drops them at binding.
        """
        node = NodeSpec(
            "process",
            noop_fn,
            deps=frozenset({"compute"}),
            accepted_params=frozenset({"amount", "user_id"}),
        )
        results = {"compute": {"amount": 99.5}}

        result = coordinator.prepare_node_input(
            node, results, initial_input={"user_id": "U42", "extra": "ignored"}
        )

        assert result == {"amount": 99.5, "user_id": "U42", "extra": "ignored"}


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


class TestStrictInputMapping:
    """Tests for strict_mapping mode: only explicitly mapped fields are passed.

    When ``strict_mapping: true`` is set alongside ``input_mapping``, the
    result dict starts empty and contains ONLY the explicitly mapped fields.
    References (``node.field``, ``$input.field``) still resolve normally.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_strict_mapping_only_mapped_fields(self, coordinator):
        """With strict_mapping, result contains only explicitly mapped fields."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer", "scorer"}),
            params={
                "input_mapping": {"rate": "producer.rate"},
                "strict_mapping": True,
            },
        )
        node_results = {
            "producer": {"rate": 100, "name": "test"},
            "scorer": {"score": 0.95},
        }

        result = coordinator.prepare_node_input(node, node_results, initial_input={"load_id": "L1"})

        assert result == {"rate": 100}

    def test_strict_mapping_no_upstream_namespace(self, coordinator):
        """Upstream node results are NOT present in strict mode."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {"val": "producer.data"},
                "strict_mapping": True,
            },
        )
        node_results = {"producer": {"data": "ok", "extra": "noise"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={"x": 1})

        assert "producer" not in result
        assert "input" not in result
        assert result == {"val": "ok"}

    def test_strict_mapping_empty_mapping_returns_empty(self, coordinator):
        """strict_mapping + empty input_mapping returns empty dict."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {},
                "strict_mapping": True,
            },
        )
        node_results = {"producer": {"data": "ok"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result == {}

    def test_strict_mapping_false_is_additive(self, coordinator):
        """strict_mapping=False preserves current additive behavior."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {"rate": "producer.rate"},
                "strict_mapping": False,
            },
        )
        node_results = {"producer": {"rate": 100, "name": "test"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={"load_id": "L1"})

        # Additive: upstream namespace present
        assert result["producer"] == {"rate": 100, "name": "test"}
        assert result["input"] == {"load_id": "L1"}
        # Explicit mapping also present
        assert result["rate"] == 100

    def test_strict_mapping_resolves_references_normally(self, coordinator):
        """$input.field and node.field references resolve even in strict mode."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {
                    "from_input": "$input.load_id",
                    "from_node": "producer.rate",
                },
                "strict_mapping": True,
            },
        )
        node_results = {"producer": {"rate": 42}}

        result = coordinator.prepare_node_input(
            node, node_results, initial_input={"load_id": "LOAD99"}
        )

        assert result == {"from_input": "LOAD99", "from_node": 42}

    def test_strict_mapping_with_modifiers(self, coordinator):
        """Safe path modifiers (| default) work in strict mode."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {
                    "val": "producer.missing | default('fallback')",
                },
                "strict_mapping": True,
            },
        )
        node_results = {"producer": {"data": "ok"}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        assert result == {"val": "fallback"}

    def test_strict_mapping_default_is_false(self, coordinator):
        """Without strict_mapping in params, behavior is additive (default)."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"rate": "producer.rate"}},
        )
        node_results = {"producer": {"rate": 100}}

        result = coordinator.prepare_node_input(node, node_results, initial_input={})

        # Default additive: upstream namespace present
        assert "producer" in result
        assert result["rate"] == 100


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


class TestAmbientInput:
    """The run's input is ambient: every top-level input field is available
    to every node unless something closer provides that name.

    Precedence (closest wins): explicit input_mapping > upstream data >
    ambient input. Coalesce semantics: ambient fills keys that are absent
    or None. strict_mapping opts a node out entirely.
    """

    @pytest.fixture
    def coordinator(self):
        return ExecutionCoordinator()

    def test_mapped_node_receives_absent_input_fields(self, coordinator):
        """A mapped node gets input fields it never declared."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"rate": "producer.rate"}},
        )
        node_results = {"producer": {"rate": 100}}
        initial_input = {"conversation_id": "c1", "email_subject": "RE: load"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["rate"] == 100
        assert result["conversation_id"] == "c1"
        assert result["email_subject"] == "RE: load"

    def test_non_none_value_never_overwritten(self, coordinator):
        """An explicit mapping pin beats the ambient input value."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"conversation_id": "producer.conversation_id"}},
        )
        node_results = {"producer": {"conversation_id": "from-upstream"}}
        initial_input = {"conversation_id": "from-input"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["conversation_id"] == "from-upstream"

    def test_none_value_coalesced_from_input(self, coordinator):
        """A mapped value that resolved to None is rescued by ambient input.

        None in hexDAG usually means a failed path resolution — protecting
        it would preserve broken wiring instead of real data.
        """
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={"input_mapping": {"conversation_id": "producer.missing_field"}},
        )
        node_results = {"producer": {"other": 1}}
        initial_input = {"conversation_id": "from-input"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["conversation_id"] == "from-input"

    def test_strict_node_gets_nothing_ambient(self, coordinator):
        """strict_mapping: the node receives ONLY its mapped fields."""
        node = NodeSpec(
            "consumer",
            noop_fn,
            deps=frozenset({"producer"}),
            params={
                "input_mapping": {"amount": "producer.amount"},
                "strict_mapping": True,
            },
        )
        node_results = {"producer": {"amount": 42}}
        initial_input = {"conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result == {"amount": 42}

    def test_single_dep_flat_upstream_key_beats_ambient(self, coordinator):
        """Closest wins: a field in the flat single-dep output shadows input."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        node_results = {"producer": {"status": "from-upstream", "x": 1}}
        initial_input = {"status": "from-input", "conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["status"] == "from-upstream"
        assert result["conversation_id"] == "c1"  # true gap filled ambiently

    def test_auto_infer_upstream_unique_beats_ambient(self, coordinator):
        """A param found in exactly one upstream source wins over $input.

        This is the old ambiguity trap resolved: name in $input + one
        upstream no longer skips/TypeErrors — upstream wins (closest).
        """
        node = NodeSpec(
            "route",
            noop_fn,
            deps=frozenset({"get_context", "decide"}),
            accepted_params=frozenset({"conversation_id", "decision"}),
        )
        node_results = {
            "get_context": {"conversation_id": "from-context"},
            "decide": {"decision": "accept"},
        }
        initial_input = {"conversation_id": "from-input"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["conversation_id"] == "from-context"
        assert result["decision"] == "accept"

    def test_no_deps_node_input_unchanged(self, coordinator):
        """A root node receives the initial input as-is (no duplication)."""
        node = NodeSpec("start", noop_fn)
        initial_input = {"conversation_id": "c1", "email_subject": "hello"}

        result = coordinator.prepare_node_input(node, {}, initial_input)

        assert result == initial_input

    def test_non_dict_initial_input_untouched(self, coordinator):
        """A non-dict pipeline input has no fields to make ambient."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        node_results = {"producer": {"x": 1}}

        result = coordinator.prepare_node_input(node, node_results, "plain string")

        assert result == {"x": 1}

    def test_non_dict_prepared_input_untouched(self, coordinator):
        """A flat non-dict upstream output passes through unchanged."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        node_results = {"producer": "scalar output"}
        initial_input = {"conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result == "scalar output"

    def test_skip_marker_not_polluted(self, coordinator):
        """Upstream-skipped markers are not enriched with ambient fields."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        node_results = {"producer": {"_skipped": True, "reason": "when clause"}}
        initial_input = {"conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert "conversation_id" not in result

    def test_ambient_does_not_mutate_shared_state(self, coordinator):
        """Ambient fill is copy-on-write: stored upstream results and the
        initial input must never gain injected keys."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        producer_result = {"x": 1}
        node_results = {"producer": producer_result}
        initial_input = {"conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert result["conversation_id"] == "c1"
        assert producer_result == {"x": 1}
        assert initial_input == {"conversation_id": "c1"}

    def test_none_input_field_not_injected(self, coordinator):
        """A None-valued input field injects nothing (never inject None)."""
        node = NodeSpec("consumer", noop_fn, deps=frozenset({"producer"}))
        node_results = {"producer": {"x": 1}}
        initial_input = {"resolved_by": None, "conversation_id": "c1"}

        result = coordinator.prepare_node_input(node, node_results, initial_input)

        assert "resolved_by" not in result
        assert result["conversation_id"] == "c1"
