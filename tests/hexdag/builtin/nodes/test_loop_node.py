"""Tests for LoopNode and ConditionalNode with string expressions."""

import pytest

from hexdag.builtin.nodes.loop_node import ConditionalNode, LoopNode


class TestConditionalNodeWithExpressions:
    """Test ConditionalNode with string expression predicates."""

    def test_when_with_string_expression(self) -> None:
        """Test when() accepts string expressions."""
        node = ConditionalNode("test_cond")
        node.when("action == 'ACCEPT'", "approve")
        node.when("action == 'REJECT'", "reject")
        node.otherwise("review")

        spec = node.build()
        assert spec.name == "test_cond"

    def test_when_with_nested_expression(self) -> None:
        """Test when() with nested attribute access."""
        node = ConditionalNode("test_cond")
        node.when("result.action == 'ACCEPT'", "approve")
        node.when("result.action == 'REJECT'", "reject")

        spec = node.build()
        assert spec.name == "test_cond"

    def test_when_with_complex_expression(self) -> None:
        """Test when() with complex boolean expressions."""
        node = ConditionalNode("test_cond")
        node.when("action == 'ACCEPT' and confidence > 0.8", "approve")
        node.when("action == 'REJECT' or confidence < 0.2", "reject")

        spec = node.build()
        assert spec.name == "test_cond"

    def test_when_with_membership(self) -> None:
        """Test when() with in operator."""
        node = ConditionalNode("test_cond")
        node.when("status in ['active', 'pending']", "process")
        node.when("status not in ['active', 'pending']", "skip")

        spec = node.build()
        assert spec.name == "test_cond"

    def test_when_mixed_callable_and_string(self) -> None:
        """Test mixing callable and string predicates."""
        node = ConditionalNode("test_cond")
        node.when(lambda d, s: d.get("fast") is True, "quick_path")
        node.when("action == 'ACCEPT'", "approve")

        spec = node.build()
        assert spec.name == "test_cond"

    def test_when_invalid_string_raises_error(self) -> None:
        """Test that invalid string expression raises error at compile time."""
        from hexdag.core.expression_parser import ExpressionError

        node = ConditionalNode("test_cond")
        with pytest.raises(ExpressionError):
            node.when("action == ==", "invalid")

    def test_when_non_string_non_callable_raises(self) -> None:
        """Test that non-string non-callable raises ValueError."""
        node = ConditionalNode("test_cond")
        with pytest.raises(ValueError, match="callable or a string expression"):
            node.when(123, "invalid")  # type: ignore


class TestConditionalNodeExecution:
    """Test ConditionalNode execution with string expressions."""

    @pytest.mark.asyncio
    async def test_simple_expression_execution(self) -> None:
        """Test executing conditional with simple expression."""
        node = ConditionalNode("router")
        node.when("action == 'ACCEPT'", "approve")
        node.when("action == 'REJECT'", "reject")
        node.otherwise("review")

        spec = node.build()
        result = await spec.fn({"action": "ACCEPT"})

        assert result["result"] == "approve"
        assert result["metadata"]["matched_branch"] == 0

    @pytest.mark.asyncio
    async def test_nested_attribute_execution(self) -> None:
        """Test executing conditional with nested attribute access."""
        node = ConditionalNode("router")
        node.when("node.result.action == 'ACCEPT'", "approve")
        node.when("node.result.action == 'REJECT'", "reject")
        node.otherwise("review")

        spec = node.build()
        result = await spec.fn({"node": {"result": {"action": "REJECT"}}})

        assert result["result"] == "reject"
        assert result["metadata"]["matched_branch"] == 1

    @pytest.mark.asyncio
    async def test_complex_expression_execution(self) -> None:
        """Test executing conditional with complex expression."""
        node = ConditionalNode("router")
        node.when("action == 'ACCEPT' and confidence > 0.8", "high_confidence_approve")
        node.when("action == 'ACCEPT' and confidence <= 0.8", "low_confidence_approve")
        node.otherwise("review")

        spec = node.build()

        # High confidence
        result = await spec.fn({"action": "ACCEPT", "confidence": 0.95})
        assert result["result"] == "high_confidence_approve"

        # Low confidence
        result = await spec.fn({"action": "ACCEPT", "confidence": 0.5})
        assert result["result"] == "low_confidence_approve"

    @pytest.mark.asyncio
    async def test_else_action_execution(self) -> None:
        """Test fallback to else_action."""
        node = ConditionalNode("router")
        node.when("status == 'done'", "complete")
        node.otherwise("pending")

        spec = node.build()
        result = await spec.fn({"status": "processing"})

        assert result["result"] == "pending"
        assert result["metadata"]["matched_branch"] is None

    @pytest.mark.asyncio
    async def test_membership_execution(self) -> None:
        """Test executing conditional with membership test."""
        node = ConditionalNode("router")
        node.when("category in ['urgent', 'critical']", "priority")
        node.when("category in ['normal', 'low']", "standard")
        node.otherwise("review")

        spec = node.build()

        result = await spec.fn({"category": "urgent"})
        assert result["result"] == "priority"

        result = await spec.fn({"category": "normal"})
        assert result["result"] == "standard"

        result = await spec.fn({"category": "unknown"})
        assert result["result"] == "review"


class TestLoopNodeWithExpressions:
    """Test LoopNode with string expression conditions."""

    def test_condition_with_string_expression(self) -> None:
        """Test condition() accepts string expressions."""
        node = LoopNode("test_loop")
        node.condition("state.iteration < 10")
        node.do(lambda d, s: {"processed": True})

        spec = node.build()
        assert spec.name == "test_loop"

    def test_condition_with_complex_expression(self) -> None:
        """Test condition() with complex boolean expression."""
        node = LoopNode("test_loop")
        node.condition("state.iteration < 10 and not state.done")
        node.do(lambda d, s: {"processed": True})

        spec = node.build()
        assert spec.name == "test_loop"

    def test_condition_with_data_and_state(self) -> None:
        """Test condition referencing both data and state."""
        node = LoopNode("test_loop")
        node.condition("items_processed < total_items and state.iteration < 100")
        node.do(lambda d, s: {"processed": True})

        spec = node.build()
        assert spec.name == "test_loop"

    def test_condition_non_string_non_callable_raises(self) -> None:
        """Test that non-string non-callable raises ValueError."""
        node = LoopNode("test_loop")
        with pytest.raises(ValueError, match="callable or a string expression"):
            node.condition(42)  # type: ignore


class TestLoopNodeExecution:
    """Test LoopNode execution with string expressions."""

    @pytest.mark.asyncio
    async def test_simple_loop_execution(self) -> None:
        """Test loop with simple iteration condition.

        Note: The loop's iteration key is 'loop_iteration' by default,
        and we need to handle None safely (first iteration has no key set yet).
        """
        node = LoopNode("counter")
        # Use comparison that handles None: iteration counter starts undefined
        node.condition(lambda d, s: s.get("loop_iteration", 0) < 3)
        node.do(lambda d, s: s.get("loop_iteration", 0))
        node.max_iterations(10)
        node.collect_list()

        spec = node.build()
        result = await spec.fn({})

        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3

    @pytest.mark.asyncio
    async def test_string_expression_with_init_state(self) -> None:
        """Test loop with string expression and initialized state."""
        node = LoopNode("counter_expr")
        # Initialize count in state so expression can evaluate
        node.init_state({"count": 0})
        node.condition("state.count < 3")
        node.do(lambda d, s: s.get("count", 0))
        # Update count after each iteration
        node.on_iteration_end(lambda s, _: {**s, "count": s.get("count", 0) + 1})
        node.max_iterations(10)
        node.collect_list()

        spec = node.build()
        result = await spec.fn({})

        assert result["result"] == [0, 1, 2]
        assert result["metadata"]["iterations"] == 3

    @pytest.mark.asyncio
    async def test_loop_with_data_condition(self) -> None:
        """Test loop that checks data values with string expression."""
        node = LoopNode("processor")
        # Initialize state with iteration=0 so expression can evaluate properly
        node.init_state({"iteration": 0})
        node.condition("state.iteration < max_iterations")
        node.do(lambda d, s: d.get("max_iterations", 0) - s.get("iteration", 0))
        # Update state after each iteration
        node.on_iteration_end(lambda s, _: {**s, "iteration": s.get("iteration", 0) + 1})
        node.max_iterations(100)
        node.collect_list()

        spec = node.build()
        result = await spec.fn({"max_iterations": 3})

        assert result["result"] == [3, 2, 1]
        assert result["metadata"]["iterations"] == 3


class TestYamlIntegration:
    """Test scenarios that would come from YAML pipelines."""

    @pytest.mark.asyncio
    async def test_llm_output_routing(self) -> None:
        """Test routing based on simulated LLM output structure."""
        # Simulates output from an LLM node that returns action decisions
        node = ConditionalNode("llm_router")
        node.when("determine_action.result.action == 'ACCEPT'", "approve_path")
        node.when("determine_action.result.action == 'REJECT'", "reject_path")
        node.when("determine_action.result.action == 'NEEDS_REVIEW'", "review_path")
        node.otherwise("error_path")

        spec = node.build()

        # Test ACCEPT
        result = await spec.fn({
            "determine_action": {"result": {"action": "ACCEPT", "confidence": 0.95}}
        })
        assert result["result"] == "approve_path"

        # Test REJECT
        result = await spec.fn({
            "determine_action": {"result": {"action": "REJECT", "reason": "Missing docs"}}
        })
        assert result["result"] == "reject_path"

        # Test unknown action
        result = await spec.fn({"determine_action": {"result": {"action": "UNKNOWN"}}})
        assert result["result"] == "error_path"

    @pytest.mark.asyncio
    async def test_confidence_threshold_routing(self) -> None:
        """Test routing based on confidence scores."""
        node = ConditionalNode("confidence_router")
        node.when("score >= 0.9", "high_confidence")
        node.when("score >= 0.7 and score < 0.9", "medium_confidence")
        node.when("score >= 0.5 and score < 0.7", "low_confidence")
        node.otherwise("reject")

        spec = node.build()

        assert (await spec.fn({"score": 0.95}))["result"] == "high_confidence"
        assert (await spec.fn({"score": 0.85}))["result"] == "medium_confidence"
        assert (await spec.fn({"score": 0.6}))["result"] == "low_confidence"
        assert (await spec.fn({"score": 0.3}))["result"] == "reject"


class TestConditionalNodeYamlMode:
    """Test ConditionalNode with YAML-style direct parameter passing.

    These tests verify the YAML mode where branches and else_action are passed
    directly to __call__() instead of using the fluent API (.when().otherwise()).
    """

    def test_yaml_branches_basic(self) -> None:
        """Test __call__ with branches parameter (YAML mode)."""
        node_spec = ConditionalNode()(
            "router",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        assert node_spec.name == "router"

    def test_yaml_branches_without_else(self) -> None:
        """Test YAML mode works without else_action."""
        node_spec = ConditionalNode()(
            "router",
            branches=[
                {"condition": "status == 'done'", "action": "complete"},
            ],
        )
        assert node_spec.name == "router"

    @pytest.mark.asyncio
    async def test_yaml_branches_empty_uses_else(self) -> None:
        """Test that empty branches list falls back to else_action."""
        node_spec = ConditionalNode()(
            "router",
            branches=[],
            else_action="fallback",
        )
        assert node_spec.name == "router"

        # With empty branches, always use else_action
        result = await node_spec.fn({"action": "anything"})
        assert result["result"] == "fallback"
        assert result["metadata"]["matched_branch"] is None

    def test_yaml_branch_missing_condition_raises(self) -> None:
        """Test that branch without condition raises error."""
        with pytest.raises(ValueError, match="condition"):
            ConditionalNode()(
                "router",
                branches=[{"action": "approve"}],
            )

    def test_yaml_branch_missing_action_raises(self) -> None:
        """Test that branch without action raises error."""
        with pytest.raises(ValueError, match="action"):
            ConditionalNode()(
                "router",
                branches=[{"condition": "x == 1"}],
            )

    def test_yaml_invalid_expression_raises(self) -> None:
        """Test that invalid expression raises error at build time."""
        from hexdag.core.expression_parser import ExpressionError

        with pytest.raises(ExpressionError):
            ConditionalNode()(
                "router",
                branches=[{"condition": "invalid ==== syntax", "action": "fail"}],
            )

    @pytest.mark.asyncio
    async def test_yaml_execution_matches_first(self) -> None:
        """Test YAML mode execution matches first branch."""
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
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
        node_spec = ConditionalNode()(
            "router",
            branches=[
                {"condition": "llm.response.decision.action == 'ACCEPT'", "action": "approve"},
            ],
            else_action="manual_review",
        )
        result = await node_spec.fn({
            "llm": {"response": {"decision": {"action": "ACCEPT", "confidence": 0.95}}}
        })
        assert result["result"] == "approve"


class TestYamlPipelineConditionalNode:
    """Test ConditionalNode through YamlPipelineBuilder."""

    def test_yaml_pipeline_conditional_builds(self) -> None:
        """Test conditional_node builds correctly from YAML."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional
spec:
  nodes:
    - kind: conditional_node
      metadata:
        name: router
      spec:
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
    async def test_yaml_pipeline_conditional_executes(self) -> None:
        """Test conditional_node executes correctly from YAML pipeline."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional
spec:
  nodes:
    - kind: conditional_node
      metadata:
        name: router
      spec:
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

    def test_yaml_pipeline_conditional_without_else(self) -> None:
        """Test conditional_node in YAML without else_action."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional-no-else
spec:
  nodes:
    - kind: conditional_node
      metadata:
        name: router
      spec:
        branches:
          - condition: "status == 'done'"
            action: complete
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "router" in graph.nodes

    def test_yaml_pipeline_conditional_complex_expressions(self) -> None:
        """Test conditional_node in YAML with complex expressions."""
        from hexdag.core.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-conditional-complex
spec:
  nodes:
    - kind: conditional_node
      metadata:
        name: confidence_router
      spec:
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
    async def test_yaml_pipeline_conditional_with_dependencies(self) -> None:
        """Test conditional_node in YAML with dependencies."""
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
    - kind: conditional_node
      metadata:
        name: router
      spec:
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
