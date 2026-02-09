"""Tests for CompositeNode control flow patterns.

These tests cover the CompositeNode unified control flow API which replaces
the deprecated ConditionalNode (mode='switch') and LoopNode (mode='while').
"""

import pytest

from hexdag.builtin.nodes.composite_node import CompositeNode


class TestCompositeNodeSwitchExpressions:
    """Test CompositeNode switch mode with string expression predicates."""

    def test_switch_with_string_expression(self) -> None:
        """Test switch mode accepts string expressions."""
        node_spec = CompositeNode()(
            name="test_cond",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        assert node_spec.name == "test_cond"

    def test_switch_with_nested_expression(self) -> None:
        """Test switch mode with nested attribute access."""
        node_spec = CompositeNode()(
            name="test_cond",
            mode="switch",
            branches=[
                {"condition": "result.action == 'ACCEPT'", "action": "approve"},
                {"condition": "result.action == 'REJECT'", "action": "reject"},
            ],
        )
        assert node_spec.name == "test_cond"

    def test_switch_with_complex_expression(self) -> None:
        """Test switch mode with complex boolean expressions."""
        node_spec = CompositeNode()(
            name="test_cond",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT' and confidence > 0.8", "action": "approve"},
                {"condition": "action == 'REJECT' or confidence < 0.2", "action": "reject"},
            ],
        )
        assert node_spec.name == "test_cond"

    def test_switch_with_membership(self) -> None:
        """Test switch mode with in operator."""
        node_spec = CompositeNode()(
            name="test_cond",
            mode="switch",
            branches=[
                {"condition": "status in ['active', 'pending']", "action": "process"},
                {"condition": "status not in ['active', 'pending']", "action": "skip"},
            ],
        )
        assert node_spec.name == "test_cond"

    def test_switch_invalid_string_raises_error(self) -> None:
        """Test that invalid string expression raises error at compile time."""
        from hexdag.core.expression_parser import ExpressionError

        with pytest.raises(ExpressionError):
            CompositeNode()(
                name="test_cond",
                mode="switch",
                branches=[{"condition": "action == ==", "action": "invalid"}],
            )


class TestCompositeNodeSwitchExecution:
    """Test CompositeNode switch mode execution with string expressions."""

    @pytest.mark.asyncio
    async def test_simple_expression_execution(self) -> None:
        """Test executing switch with simple expression."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        result = await node_spec.fn({"action": "ACCEPT"})

        assert result["result"] == "approve"
        assert result["metadata"]["matched_branch"] == 0

    @pytest.mark.asyncio
    async def test_nested_attribute_execution(self) -> None:
        """Test executing switch with nested attribute access."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "node.result.action == 'ACCEPT'", "action": "approve"},
                {"condition": "node.result.action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        result = await node_spec.fn({"node": {"result": {"action": "REJECT"}}})

        assert result["result"] == "reject"
        assert result["metadata"]["matched_branch"] == 1

    @pytest.mark.asyncio
    async def test_complex_expression_execution(self) -> None:
        """Test executing switch with complex expression."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {
                    "condition": "action == 'ACCEPT' and confidence > 0.8",
                    "action": "high_confidence_approve",
                },
                {
                    "condition": "action == 'ACCEPT' and confidence <= 0.8",
                    "action": "low_confidence_approve",
                },
            ],
            else_action="review",
        )

        # High confidence
        result = await node_spec.fn({"action": "ACCEPT", "confidence": 0.95})
        assert result["result"] == "high_confidence_approve"

        # Low confidence
        result = await node_spec.fn({"action": "ACCEPT", "confidence": 0.5})
        assert result["result"] == "low_confidence_approve"

    @pytest.mark.asyncio
    async def test_else_action_execution(self) -> None:
        """Test fallback to else_action."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "status == 'done'", "action": "complete"},
            ],
            else_action="pending",
        )
        result = await node_spec.fn({"status": "processing"})

        assert result["result"] == "pending"
        assert result["metadata"]["matched_branch"] is None

    @pytest.mark.asyncio
    async def test_membership_execution(self) -> None:
        """Test executing switch with membership test."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "category in ['urgent', 'critical']", "action": "priority"},
                {"condition": "category in ['normal', 'low']", "action": "standard"},
            ],
            else_action="review",
        )

        result = await node_spec.fn({"category": "urgent"})
        assert result["result"] == "priority"

        result = await node_spec.fn({"category": "normal"})
        assert result["result"] == "standard"

        result = await node_spec.fn({"category": "unknown"})
        assert result["result"] == "review"


class TestCompositeNodeWhileExpressions:
    """Test CompositeNode while mode with string expression conditions."""

    def test_condition_with_string_expression(self) -> None:
        """Test while condition accepts string expressions."""
        node_spec = CompositeNode()(
            name="test_loop",
            mode="while",
            condition="state.iteration < 10",
            initial_state={"iteration": 0},
            state_update={"iteration": "state.iteration + 1"},
            body=lambda d, **ports: {"processed": True},
        )
        assert node_spec.name == "test_loop"

    def test_condition_with_complex_expression(self) -> None:
        """Test while condition with complex boolean expression."""
        node_spec = CompositeNode()(
            name="test_loop",
            mode="while",
            condition="state.iteration < 10 and not state.done",
            initial_state={"iteration": 0, "done": False},
            state_update={"iteration": "state.iteration + 1"},
            body=lambda d, **ports: {"processed": True},
        )
        assert node_spec.name == "test_loop"

    def test_condition_with_data_and_state(self) -> None:
        """Test while condition referencing both data and state."""
        node_spec = CompositeNode()(
            name="test_loop",
            mode="while",
            condition="items_processed < total_items and state.iteration < 100",
            initial_state={"iteration": 0},
            state_update={"iteration": "state.iteration + 1"},
            body=lambda d, **ports: {"processed": True},
        )
        assert node_spec.name == "test_loop"


class TestCompositeNodeWhileExecution:
    """Test CompositeNode while mode execution with string expressions.

    Note: These tests use state.iteration which is automatically set to 0,
    then incremented by CompositeNode after each loop iteration.
    """

    @pytest.mark.asyncio
    async def test_simple_loop_execution(self) -> None:
        """Test loop with simple iteration condition using built-in iteration tracking."""
        node_spec = CompositeNode()(
            name="counter",
            mode="while",
            condition="state.iteration < 3",
            initial_state={"iteration": 0},
            body=lambda d, **ports: d["iteration"],
            max_iterations=10,
            collect="list",
        )
        result = await node_spec.fn({})

        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3

    @pytest.mark.asyncio
    async def test_string_expression_with_init_state(self) -> None:
        """Test loop with string expression and initialized state."""
        node_spec = CompositeNode()(
            name="counter_expr",
            mode="while",
            condition="state.iteration < 3",
            initial_state={"iteration": 0},
            body=lambda d, **ports: d["iteration"],
            max_iterations=10,
            collect="list",
        )
        result = await node_spec.fn({})

        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3

    @pytest.mark.asyncio
    async def test_loop_with_data_condition(self) -> None:
        """Test loop that checks data values with string expression."""
        node_spec = CompositeNode()(
            name="processor",
            mode="while",
            condition="state.iteration < max_iterations",
            initial_state={"iteration": 0},
            body=lambda d, **ports: d["max_iterations"] - d["iteration"],
            max_iterations=100,
            collect="list",
        )
        result = await node_spec.fn({"max_iterations": 3})

        assert result["result"] == [3, 2, 1]
        assert result["metadata"]["iterations"] == 3


class TestYamlIntegration:
    """Test scenarios that would come from YAML pipelines."""

    @pytest.mark.asyncio
    async def test_llm_output_routing(self) -> None:
        """Test routing based on simulated LLM output structure."""
        node_spec = CompositeNode()(
            name="llm_router",
            mode="switch",
            branches=[
                {
                    "condition": "determine_action.result.action == 'ACCEPT'",
                    "action": "approve_path",
                },
                {
                    "condition": "determine_action.result.action == 'REJECT'",
                    "action": "reject_path",
                },
                {
                    "condition": "determine_action.result.action == 'NEEDS_REVIEW'",
                    "action": "review_path",
                },
            ],
            else_action="error_path",
        )

        # Test ACCEPT
        result = await node_spec.fn({
            "determine_action": {"result": {"action": "ACCEPT", "confidence": 0.95}}
        })
        assert result["result"] == "approve_path"

        # Test REJECT
        result = await node_spec.fn({
            "determine_action": {"result": {"action": "REJECT", "reason": "Missing docs"}}
        })
        assert result["result"] == "reject_path"

        # Test unknown action
        result = await node_spec.fn({"determine_action": {"result": {"action": "UNKNOWN"}}})
        assert result["result"] == "error_path"

    @pytest.mark.asyncio
    async def test_confidence_threshold_routing(self) -> None:
        """Test routing based on confidence scores."""
        node_spec = CompositeNode()(
            name="confidence_router",
            mode="switch",
            branches=[
                {"condition": "score >= 0.9", "action": "high_confidence"},
                {"condition": "score >= 0.7 and score < 0.9", "action": "medium_confidence"},
                {"condition": "score >= 0.5 and score < 0.7", "action": "low_confidence"},
            ],
            else_action="reject",
        )

        assert (await node_spec.fn({"score": 0.95}))["result"] == "high_confidence"
        assert (await node_spec.fn({"score": 0.85}))["result"] == "medium_confidence"
        assert (await node_spec.fn({"score": 0.6}))["result"] == "low_confidence"
        assert (await node_spec.fn({"score": 0.3}))["result"] == "reject"


class TestCompositeNodeSwitchYamlMode:
    """Test CompositeNode switch mode with YAML-style direct parameter passing.

    These tests verify the YAML mode where branches and else_action are passed
    directly to __call__().
    """

    def test_yaml_branches_basic(self) -> None:
        """Test __call__ with branches parameter (YAML mode)."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        assert node_spec.name == "router"

    def test_yaml_branches_without_else(self) -> None:
        """Test YAML mode works without else_action."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "status == 'done'", "action": "complete"},
            ],
        )
        assert node_spec.name == "router"

    def test_yaml_branches_empty_raises_error(self) -> None:
        """Test that empty branches list raises validation error."""
        with pytest.raises(ValueError, match="branches"):
            CompositeNode()(
                name="router",
                mode="switch",
                branches=[],
                else_action="fallback",
            )

    def test_yaml_branch_missing_condition_raises(self) -> None:
        """Test that branch without condition raises error."""
        with pytest.raises(ValueError, match="condition"):
            CompositeNode()(
                name="router",
                mode="switch",
                branches=[{"action": "approve"}],
            )

    def test_yaml_branch_without_action_allowed(self) -> None:
        """Test that branch without action is allowed (uses body or routing mode)."""
        # CompositeNode allows branches without action for execution mode
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[{"condition": "x == 1"}],  # No action, will return None
        )
        assert node_spec.name == "router"

    def test_yaml_invalid_expression_raises(self) -> None:
        """Test that invalid expression raises error at build time."""
        from hexdag.core.expression_parser import ExpressionError

        with pytest.raises(ExpressionError):
            CompositeNode()(
                name="router",
                mode="switch",
                branches=[{"condition": "invalid ==== syntax", "action": "fail"}],
            )

    @pytest.mark.asyncio
    async def test_yaml_execution_matches_first(self) -> None:
        """Test YAML mode execution matches first branch."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        result = await node_spec.fn({"action": "ACCEPT"})
        assert result["result"] == "approve"
        assert result["metadata"]["matched_branch"] == 0

    @pytest.mark.asyncio
    async def test_yaml_execution_matches_second(self) -> None:
        """Test YAML mode execution matches second branch."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        result = await node_spec.fn({"action": "REJECT"})
        assert result["result"] == "reject"
        assert result["metadata"]["matched_branch"] == 1

    @pytest.mark.asyncio
    async def test_yaml_execution_else_fallback(self) -> None:
        """Test YAML mode falls back to else_action."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "status == 'done'", "action": "complete"},
            ],
            else_action="pending",
        )
        result = await node_spec.fn({"status": "processing"})
        assert result["result"] == "pending"
        assert result["metadata"]["matched_branch"] is None

    @pytest.mark.asyncio
    async def test_yaml_execution_no_match_no_else(self) -> None:
        """Test YAML mode returns None when no match and no else_action."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "status == 'done'", "action": "complete"},
            ],
        )
        result = await node_spec.fn({"status": "processing"})
        assert result["result"] is None
        assert result["metadata"]["matched_branch"] is None

    @pytest.mark.asyncio
    async def test_yaml_complex_expressions(self) -> None:
        """Test YAML mode with complex boolean expressions."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "score >= 0.9 and confidence > 0.8", "action": "high"},
                {"condition": "score >= 0.5", "action": "medium"},
            ],
            else_action="low",
        )
        assert (await node_spec.fn({"score": 0.95, "confidence": 0.9}))["result"] == "high"
        assert (await node_spec.fn({"score": 0.7, "confidence": 0.5}))["result"] == "medium"
        assert (await node_spec.fn({"score": 0.3, "confidence": 0.2}))["result"] == "low"

    @pytest.mark.asyncio
    async def test_yaml_nested_attribute_access(self) -> None:
        """Test YAML mode with nested attribute expressions."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "result.action == 'ACCEPT'", "action": "approve"},
                {"condition": "result.action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        result = await node_spec.fn({"result": {"action": "ACCEPT"}})
        assert result["result"] == "approve"

    @pytest.mark.asyncio
    async def test_yaml_membership_operators(self) -> None:
        """Test YAML mode with in/not in operators."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "status in ['active', 'pending']", "action": "process"},
                {"condition": "status not in ['active', 'pending', 'done']", "action": "skip"},
            ],
            else_action="archive",
        )
        assert (await node_spec.fn({"status": "active"}))["result"] == "process"
        assert (await node_spec.fn({"status": "unknown"}))["result"] == "skip"
        assert (await node_spec.fn({"status": "done"}))["result"] == "archive"

    @pytest.mark.asyncio
    async def test_yaml_deep_nested_access(self) -> None:
        """Test YAML mode with deeply nested attribute access."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "llm.response.decision.action == 'ACCEPT'", "action": "approve"},
            ],
            else_action="manual_review",
        )
        result = await node_spec.fn({
            "llm": {"response": {"decision": {"action": "ACCEPT", "confidence": 0.95}}}
        })
        assert result["result"] == "approve"


class TestYamlPipelineCompositeNode:
    """Test CompositeNode through YamlPipelineBuilder."""

    def test_yaml_pipeline_switch_builds(self) -> None:
        """Test composite_node builds correctly from YAML."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        branches:
          - condition: "action == 'ACCEPT'"
            action: approve
          - condition: "action == 'REJECT'"
            action: reject
        else_action: review
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "router" in graph.nodes
        assert graph.nodes["router"].name == "router"

    @pytest.mark.asyncio
    async def test_yaml_pipeline_switch_executes(self) -> None:
        """Test composite_node executes correctly from YAML pipeline."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        branches:
          - condition: "action == 'ACCEPT'"
            action: approve
          - condition: "action == 'REJECT'"
            action: reject
        else_action: review
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        router = graph.nodes["router"]

        # Test ACCEPT routing
        result = await router.fn({"action": "ACCEPT"})
        assert result["result"] == "approve"

        # Test REJECT routing
        result = await router.fn({"action": "REJECT"})
        assert result["result"] == "reject"

        # Test else fallback
        result = await router.fn({"action": "UNKNOWN"})
        assert result["result"] == "review"

    def test_yaml_pipeline_switch_without_else(self) -> None:
        """Test composite_node in YAML without else_action."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional-no-else
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        branches:
          - condition: "status == 'done'"
            action: complete
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "router" in graph.nodes

    def test_yaml_pipeline_switch_complex_expressions(self) -> None:
        """Test composite_node in YAML with complex expressions."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional-complex
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: confidence_router
      spec:
        mode: switch
        branches:
          - condition: "score >= 0.9 and confidence > 0.8"
            action: high_confidence
          - condition: "score >= 0.5"
            action: medium_confidence
        else_action: low_confidence
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "confidence_router" in graph.nodes

    @pytest.mark.asyncio
    async def test_yaml_pipeline_switch_with_dependencies(self) -> None:
        """Test composite_node in YAML with dependencies."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional-deps
spec:
  nodes:
    - kind: function_node
      metadata:
        name: analyzer
      spec:
        fn: "json.loads"
    - kind: composite_node
      metadata:
        name: router
      spec:
        mode: switch
        branches:
          - condition: "action == 'ACCEPT'"
            action: approve
        else_action: review
      dependencies:
        - analyzer
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        router = graph.nodes["router"]
        assert "analyzer" in router.deps


class TestCompositeNodeSwitchAdvancedExpressions:
    """Test CompositeNode switch mode with advanced expression features."""

    @pytest.mark.asyncio
    async def test_switch_with_function_calls(self) -> None:
        """Test conditions using whitelisted function calls."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "len(items) > 0", "action": "process"},
                {"condition": "len(items) == 0", "action": "skip"},
            ],
            else_action="error",
        )

        result = await node_spec.fn({"items": [1, 2, 3]})
        assert result["result"] == "process"

        result = await node_spec.fn({"items": []})
        assert result["result"] == "skip"

    @pytest.mark.asyncio
    async def test_switch_with_string_functions(self) -> None:
        """Test conditions using string functions."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "upper(status) == 'ACTIVE'", "action": "process"},
                {"condition": "lower(status) == 'pending'", "action": "wait"},
            ],
            else_action="skip",
        )

        result = await node_spec.fn({"status": "active"})
        assert result["result"] == "process"

        result = await node_spec.fn({"status": "PENDING"})
        assert result["result"] == "wait"

    @pytest.mark.asyncio
    async def test_switch_with_default_function(self) -> None:
        """Test conditions using default() for null handling."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "default(value, 0) > 10", "action": "high"},
                {"condition": "default(value, 0) > 0", "action": "low"},
            ],
            else_action="zero",
        )

        result = await node_spec.fn({"value": 20})
        assert result["result"] == "high"

        result = await node_spec.fn({"value": 5})
        assert result["result"] == "low"

        result = await node_spec.fn({"value": None})
        assert result["result"] == "zero"

        result = await node_spec.fn({})  # Missing key
        assert result["result"] == "zero"

    @pytest.mark.asyncio
    async def test_switch_with_arithmetic(self) -> None:
        """Test conditions with arithmetic expressions."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "price * quantity > 1000", "action": "large_order"},
                {"condition": "price * quantity > 100", "action": "medium_order"},
            ],
            else_action="small_order",
        )

        result = await node_spec.fn({"price": 50, "quantity": 30})
        assert result["result"] == "large_order"

        result = await node_spec.fn({"price": 10, "quantity": 20})
        assert result["result"] == "medium_order"

        result = await node_spec.fn({"price": 5, "quantity": 10})
        assert result["result"] == "small_order"

    @pytest.mark.asyncio
    async def test_switch_with_min_max(self) -> None:
        """Test conditions using min/max functions."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "max(score_a, score_b, score_c) >= 90", "action": "excellent"},
                {"condition": "min(score_a, score_b, score_c) >= 70", "action": "passing"},
            ],
            else_action="needs_improvement",
        )

        result = await node_spec.fn({"score_a": 85, "score_b": 95, "score_c": 80})
        assert result["result"] == "excellent"

        result = await node_spec.fn({"score_a": 75, "score_b": 80, "score_c": 70})
        assert result["result"] == "passing"

        result = await node_spec.fn({"score_a": 75, "score_b": 65, "score_c": 70})
        assert result["result"] == "needs_improvement"

    @pytest.mark.asyncio
    async def test_switch_negotiation_scenario(self) -> None:
        """Test CompositeNode switch with realistic negotiation conditions."""
        node_spec = CompositeNode()(
            name="route_action",
            mode="switch",
            branches=[
                # Winner already locked - reject
                {"condition": "get_context.load.winner_locked == True", "action": "reject"},
                # Low confidence - escalate
                {"condition": "extract_offer.confidence < 0.5", "action": "escalate"},
                # Negotiation not active - reject
                {"condition": "get_context.negotiation.status != 'ACTIVE'", "action": "reject"},
                # Rate acceptable - accept
                {
                    "condition": "extract_offer.rate <= get_context.load.target_rate",
                    "action": "accept",
                },
            ],
            else_action="counter",
        )

        # Test: Winner locked
        result = await node_spec.fn({
            "get_context": {
                "load": {"winner_locked": True, "target_rate": 2.5},
                "negotiation": {"status": "ACTIVE"},
            },
            "extract_offer": {"rate": 2.0, "confidence": 0.9},
        })
        assert result["result"] == "reject"

        # Test: Low confidence
        result = await node_spec.fn({
            "get_context": {
                "load": {"winner_locked": False, "target_rate": 2.5},
                "negotiation": {"status": "ACTIVE"},
            },
            "extract_offer": {"rate": 2.0, "confidence": 0.3},
        })
        assert result["result"] == "escalate"

        # Test: Rate acceptable
        result = await node_spec.fn({
            "get_context": {
                "load": {"winner_locked": False, "target_rate": 2.5},
                "negotiation": {"status": "ACTIVE"},
            },
            "extract_offer": {"rate": 2.0, "confidence": 0.9},
        })
        assert result["result"] == "accept"

        # Test: Rate too high - counter
        result = await node_spec.fn({
            "get_context": {
                "load": {"winner_locked": False, "target_rate": 2.0},
                "negotiation": {"status": "ACTIVE"},
            },
            "extract_offer": {"rate": 2.5, "confidence": 0.9},
        })
        assert result["result"] == "counter"

    @pytest.mark.asyncio
    async def test_switch_with_coalesce(self) -> None:
        """Test conditions using coalesce() for multiple fallbacks."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "coalesce(primary, secondary, 0) > 100", "action": "high"},
            ],
            else_action="low",
        )

        # Primary value present
        result = await node_spec.fn({"primary": 150, "secondary": 50})
        assert result["result"] == "high"

        # Only secondary value
        result = await node_spec.fn({"primary": None, "secondary": 150})
        assert result["result"] == "high"

        # Both null - falls back to 0
        result = await node_spec.fn({"primary": None, "secondary": None})
        assert result["result"] == "low"

    @pytest.mark.asyncio
    async def test_switch_with_isnone_isempty(self) -> None:
        """Test conditions using isnone() and isempty()."""
        node_spec = CompositeNode()(
            name="router",
            mode="switch",
            branches=[
                {"condition": "isnone(value)", "action": "null_value"},
                {"condition": "isempty(items)", "action": "empty_list"},
            ],
            else_action="has_data",
        )

        result = await node_spec.fn({"value": None, "items": [1, 2]})
        assert result["result"] == "null_value"

        result = await node_spec.fn({"value": 10, "items": []})
        assert result["result"] == "empty_list"

        result = await node_spec.fn({"value": 10, "items": [1]})
        assert result["result"] == "has_data"
