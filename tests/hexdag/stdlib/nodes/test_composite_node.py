"""Tests for CompositeNode - unified control flow node."""

import pytest

from hexdag.stdlib.nodes.composite_node import CompositeNode


class TestCompositeNodeBasics:
    """Test basic CompositeNode functionality."""

    def test_creates_with_while_mode(self) -> None:
        """Test creating CompositeNode with while mode."""
        node_spec = CompositeNode()(
            name="test_while",
            mode="while",
            condition="state.iteration < 3",
            initial_state={"iteration": 0},
        )
        assert node_spec.name == "test_while"

    def test_creates_with_foreach_mode(self) -> None:
        """Test creating CompositeNode with for-each mode."""
        node_spec = CompositeNode()(
            name="test_foreach",
            mode="for-each",
            items="$input.items",
        )
        assert node_spec.name == "test_foreach"

    def test_creates_with_times_mode(self) -> None:
        """Test creating CompositeNode with times mode."""
        node_spec = CompositeNode()(
            name="test_times",
            mode="times",
            count=5,
        )
        assert node_spec.name == "test_times"

    def test_creates_with_if_else_mode(self) -> None:
        """Test creating CompositeNode with if-else mode."""
        node_spec = CompositeNode()(
            name="test_if_else",
            mode="if-else",
            condition="status == 'active'",
        )
        assert node_spec.name == "test_if_else"

    def test_creates_with_switch_mode(self) -> None:
        """Test creating CompositeNode with switch mode."""
        node_spec = CompositeNode()(
            name="test_switch",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
                {"condition": "action == 'REJECT'", "action": "reject"},
            ],
            else_action="review",
        )
        assert node_spec.name == "test_switch"


class TestCompositeNodeValidation:
    """Test parameter validation for CompositeNode."""

    def test_while_requires_condition(self) -> None:
        """Test while mode requires condition parameter."""
        with pytest.raises(ValueError, match="condition"):
            CompositeNode()(name="test", mode="while")

    def test_foreach_requires_items(self) -> None:
        """Test for-each mode requires items parameter."""
        with pytest.raises(ValueError, match="items"):
            CompositeNode()(name="test", mode="for-each")

    def test_times_requires_count(self) -> None:
        """Test times mode requires positive count parameter."""
        with pytest.raises(ValueError, match="count"):
            CompositeNode()(name="test", mode="times")

    def test_if_else_requires_condition(self) -> None:
        """Test if-else mode requires condition parameter."""
        with pytest.raises(ValueError, match="condition"):
            CompositeNode()(name="test", mode="if-else")

    def test_switch_requires_branches(self) -> None:
        """Test switch mode requires branches parameter."""
        with pytest.raises(ValueError, match="branches"):
            CompositeNode()(name="test", mode="switch")

    def test_invalid_mode_raises(self) -> None:
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            CompositeNode()(name="test", mode="invalid")  # type: ignore

    def test_body_and_body_pipeline_exclusive(self) -> None:
        """Test body and body_pipeline are mutually exclusive."""
        with pytest.raises(ValueError, match="both"):
            CompositeNode()(
                name="test",
                mode="times",
                count=5,
                body="myapp.process",
                body_pipeline="./pipeline.yaml",
            )


class TestWhileModeExecution:
    """Test while mode execution."""

    @pytest.mark.asyncio
    async def test_while_basic_iteration(self) -> None:
        """Test basic while loop iteration."""
        counter = {"value": 0}

        def body_fn(data: dict, **ports: dict) -> dict:
            counter["value"] += 1
            return {"count": counter["value"]}

        node_spec = CompositeNode()(
            name="while_test",
            mode="while",
            condition="state.iteration < 3",
            initial_state={"iteration": 0},
            state_update={"iteration": "state.iteration + 1"},
            body=body_fn,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["metadata"]["mode"] == "while"
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["stopped_by"] == "condition"

    @pytest.mark.asyncio
    async def test_while_max_iterations_limit(self) -> None:
        """Test while loop stops at max_iterations."""
        node_spec = CompositeNode()(
            name="while_max",
            mode="while",
            condition="True",  # Always true
            initial_state={},
            max_iterations=5,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["metadata"]["iterations"] == 5
        assert result["metadata"]["stopped_by"] == "limit"

    @pytest.mark.asyncio
    async def test_while_yield_to_downstream(self) -> None:
        """Test while mode without body yields to downstream."""
        node_spec = CompositeNode()(
            name="while_yield",
            mode="while",
            condition="state.iteration < 2",
            initial_state={"iteration": 0},
            state_update={"iteration": "state.iteration + 1"},
            # No body - yields to downstream
        )

        result = await node_spec.fn({})

        assert result["metadata"]["mode"] == "while"
        assert result["metadata"]["iterations"] == 2
        # Results should be iteration contexts
        assert len(result["result"]) == 2
        assert result["result"][0]["$index"] == 0


class TestForEachModeExecution:
    """Test for-each mode execution."""

    @pytest.mark.asyncio
    async def test_foreach_basic_iteration(self) -> None:
        """Test basic for-each iteration."""
        node_spec = CompositeNode()(
            name="foreach_test",
            mode="for-each",
            items="items",
            collect="list",
        )

        result = await node_spec.fn({"items": [1, 2, 3]})

        assert result["metadata"]["mode"] == "for-each"
        assert result["metadata"]["iterations"] == 3
        assert result["metadata"]["stopped_by"] == "exhausted"

    @pytest.mark.asyncio
    async def test_foreach_with_body(self) -> None:
        """Test for-each with inline body function."""

        def double(data: dict, **ports: dict) -> int:
            return data.get("$item", 0) * 2

        node_spec = CompositeNode()(
            name="foreach_body",
            mode="for-each",
            items="items",
            body=double,
            collect="list",
        )

        result = await node_spec.fn({"items": [1, 2, 3]})

        assert result["result"] == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_foreach_concurrency(self) -> None:
        """Test for-each with concurrency."""
        import asyncio

        call_order: list[int] = []

        async def slow_process(data: dict, **ports: dict) -> int:
            idx = data.get("$index", 0)
            call_order.append(idx)
            await asyncio.sleep(0.01)
            return idx

        node_spec = CompositeNode()(
            name="foreach_concurrent",
            mode="for-each",
            items="items",
            body=slow_process,
            concurrency=3,  # Process 3 at a time
            collect="list",
        )

        result = await node_spec.fn({"items": [0, 1, 2, 3, 4]})

        assert result["metadata"]["iterations"] == 5
        assert set(result["result"]) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_foreach_empty_collection(self) -> None:
        """Test for-each with empty collection."""
        node_spec = CompositeNode()(
            name="foreach_empty",
            mode="for-each",
            items="items",
            collect="list",
        )

        result = await node_spec.fn({"items": []})

        assert result["metadata"]["iterations"] == 0
        assert result["result"] == []


class TestTimesModeExecution:
    """Test times mode execution."""

    @pytest.mark.asyncio
    async def test_times_basic_iteration(self) -> None:
        """Test basic times iteration."""
        node_spec = CompositeNode()(
            name="times_test",
            mode="times",
            count=5,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["metadata"]["mode"] == "times"
        assert result["metadata"]["iterations"] == 5

    @pytest.mark.asyncio
    async def test_times_with_body(self) -> None:
        """Test times with inline body function."""

        def process(data: dict, **ports: dict) -> int:
            return data.get("$index", 0) ** 2

        node_spec = CompositeNode()(
            name="times_body",
            mode="times",
            count=4,
            body=process,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["result"] == [0, 1, 4, 9]

    @pytest.mark.asyncio
    async def test_times_zero_count(self) -> None:
        """Test times with zero count."""
        node_spec = CompositeNode()(
            name="times_zero",
            mode="times",
            count=0,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["metadata"]["iterations"] == 0
        assert result["result"] == []


class TestIfElseModeExecution:
    """Test if-else mode execution."""

    @pytest.mark.asyncio
    async def test_if_else_true_branch(self) -> None:
        """Test if-else when condition is true."""

        def process_true(data: dict, **ports: dict) -> str:
            return "processed_true"

        node_spec = CompositeNode()(
            name="if_else_test",
            mode="if-else",
            condition="status == 'active'",
            body=process_true,
        )

        result = await node_spec.fn({"status": "active"})

        assert result["metadata"]["condition_met"] is True
        assert result["metadata"]["branch_taken"] == "if"
        assert result["result"] == "processed_true"

    @pytest.mark.asyncio
    async def test_if_else_false_branch(self) -> None:
        """Test if-else when condition is false with else_body."""

        def process_else(data: dict, **ports: dict) -> str:
            return "processed_else"

        node_spec = CompositeNode()(
            name="if_else_test",
            mode="if-else",
            condition="status == 'active'",
            body="dummy",  # Won't be called
            else_body=process_else,
        )

        result = await node_spec.fn({"status": "inactive"})

        assert result["metadata"]["condition_met"] is False
        assert result["metadata"]["branch_taken"] == "else"
        assert result["result"] == "processed_else"

    @pytest.mark.asyncio
    async def test_if_else_yield_to_downstream(self) -> None:
        """Test if-else yields condition result when no body."""
        node_spec = CompositeNode()(
            name="if_else_yield",
            mode="if-else",
            condition="status == 'active'",
            # No body - yields to downstream
        )

        result = await node_spec.fn({"status": "active"})

        assert result["metadata"]["condition_met"] is True
        assert result["result"]["condition_met"] is True


class TestSwitchModeExecution:
    """Test switch mode execution."""

    @pytest.mark.asyncio
    async def test_switch_routing_basic(self) -> None:
        """Test basic switch routing (no body, returns action)."""
        node_spec = CompositeNode()(
            name="switch_route",
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
    async def test_switch_routing_second_branch(self) -> None:
        """Test switch routes to second branch."""
        node_spec = CompositeNode()(
            name="switch_route",
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
    async def test_switch_routing_else(self) -> None:
        """Test switch falls back to else_action."""
        node_spec = CompositeNode()(
            name="switch_route",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "action": "approve"},
            ],
            else_action="review",
        )

        result = await node_spec.fn({"action": "UNKNOWN"})

        assert result["result"] == "review"
        assert result["metadata"]["matched_branch"] is None

    @pytest.mark.asyncio
    async def test_switch_with_body(self) -> None:
        """Test switch with inline body execution."""

        def approve_fn(data: dict, **ports: dict) -> dict:
            return {"status": "approved", "action": data.get("action")}

        node_spec = CompositeNode()(
            name="switch_body",
            mode="switch",
            branches=[
                {"condition": "action == 'ACCEPT'", "body": approve_fn},
            ],
            else_action="review",
        )

        result = await node_spec.fn({"action": "ACCEPT"})

        assert result["result"]["status"] == "approved"
        assert result["metadata"]["matched_branch"] == 0

    @pytest.mark.asyncio
    async def test_switch_complex_conditions(self) -> None:
        """Test switch with complex boolean conditions."""
        node_spec = CompositeNode()(
            name="switch_complex",
            mode="switch",
            branches=[
                {"condition": "score >= 90 and confidence > 0.8", "action": "high"},
                {"condition": "score >= 50", "action": "medium"},
            ],
            else_action="low",
        )

        result = await node_spec.fn({"score": 95, "confidence": 0.9})
        assert result["result"] == "high"

        result = await node_spec.fn({"score": 70, "confidence": 0.5})
        assert result["result"] == "medium"

        result = await node_spec.fn({"score": 30, "confidence": 0.2})
        assert result["result"] == "low"


class TestCollectModes:
    """Test different collect modes."""

    @pytest.mark.asyncio
    async def test_collect_list(self) -> None:
        """Test collect='list' returns all results."""
        node_spec = CompositeNode()(
            name="collect_list",
            mode="times",
            count=3,
            collect="list",
        )

        result = await node_spec.fn({})

        assert isinstance(result["result"], list)
        assert len(result["result"]) == 3

    @pytest.mark.asyncio
    async def test_collect_last(self) -> None:
        """Test collect='last' returns last result."""

        def counter(data: dict, **ports: dict) -> int:
            return data.get("$index", 0)

        node_spec = CompositeNode()(
            name="collect_last",
            mode="times",
            count=5,
            body=counter,
            collect="last",
        )

        result = await node_spec.fn({})

        assert result["result"] == 4

    @pytest.mark.asyncio
    async def test_collect_first(self) -> None:
        """Test collect='first' returns first result."""

        def counter(data: dict, **ports: dict) -> int:
            return data.get("$index", 0) + 10

        node_spec = CompositeNode()(
            name="collect_first",
            mode="times",
            count=5,
            body=counter,
            collect="first",
        )

        result = await node_spec.fn({})

        assert result["result"] == 10

    @pytest.mark.asyncio
    async def test_collect_dict(self) -> None:
        """Test collect='dict' returns keyed results."""

        def make_item(data: dict, **ports: dict) -> dict:
            idx = data.get("$index", 0)
            return {"id": f"item_{idx}", "value": idx * 10}

        node_spec = CompositeNode()(
            name="collect_dict",
            mode="times",
            count=3,
            body=make_item,
            collect="dict",
            key_field="id",
        )

        result = await node_spec.fn({})

        assert "item_0" in result["result"]
        assert "item_1" in result["result"]
        assert "item_2" in result["result"]


class TestYamlPipelineIntegration:
    """Test CompositeNode through YamlPipelineBuilder."""

    def test_yaml_composite_while_builds(self) -> None:
        """Test composite_node with while mode builds from YAML."""
        from hexdag.kernel.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-composite-while
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: retry_loop
      spec:
        mode: while
        condition: "state.attempts < 3"
        initial_state:
          attempts: 0
        state_update:
          attempts: "state.attempts + 1"
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "retry_loop" in graph.nodes

    def test_yaml_composite_foreach_builds(self) -> None:
        """Test composite_node with for-each mode builds from YAML."""
        from hexdag.kernel.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-composite-foreach
spec:
  nodes:
    - kind: composite_node
      metadata:
        name: process_items
      spec:
        mode: for-each
        items: "$input.items"
        concurrency: 5
"""
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)
        assert "process_items" in graph.nodes

    def test_yaml_composite_switch_routing(self) -> None:
        """Test composite_node with switch mode for routing."""
        from hexdag.kernel.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-composite-switch
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

    @pytest.mark.asyncio
    async def test_yaml_composite_switch_executes(self) -> None:
        """Test composite_node switch mode executes correctly."""
        from hexdag.kernel.pipeline_builder import YamlPipelineBuilder

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-composite-switch-exec
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


class TestErrorHandling:
    """Test error handling in CompositeNode."""

    @pytest.mark.asyncio
    async def test_error_handling_fail_fast(self) -> None:
        """Test fail_fast error handling stops on first error."""
        from hexdag.kernel.orchestration.body_executor import BodyExecutorError

        def failing_body(data: dict, **ports: dict) -> None:
            if data.get("$index", 0) == 2:
                raise ValueError("Intentional failure")

        node_spec = CompositeNode()(
            name="fail_fast_test",
            mode="times",
            count=5,
            body=failing_body,
            error_handling="fail_fast",
        )

        with pytest.raises(BodyExecutorError, match="Intentional failure"):
            await node_spec.fn({})

    @pytest.mark.asyncio
    async def test_error_handling_continue(self) -> None:
        """Test continue error handling continues on error."""
        call_count = {"value": 0}

        def failing_body(data: dict, **ports: dict) -> int:
            call_count["value"] += 1
            idx = data.get("$index", 0)
            if idx == 2:
                raise ValueError("Intentional failure")
            return idx

        node_spec = CompositeNode()(
            name="continue_test",
            mode="times",
            count=5,
            body=failing_body,
            error_handling="continue",
            collect="list",
        )

        result = await node_spec.fn({})

        # All iterations should be attempted
        assert call_count["value"] == 5
        assert result["metadata"]["failed"] == 1
        assert result["metadata"]["successful"] == 4

    @pytest.mark.asyncio
    async def test_error_handling_collect(self) -> None:
        """Test collect error handling collects errors."""

        def failing_body(data: dict, **ports: dict) -> int:
            idx = data.get("$index", 0)
            if idx % 2 == 1:
                raise ValueError(f"Failed at {idx}")
            return idx

        node_spec = CompositeNode()(
            name="collect_errors_test",
            mode="times",
            count=4,
            body=failing_body,
            error_handling="collect",
        )

        result = await node_spec.fn({})

        assert result["metadata"]["failed"] == 2
        assert result["metadata"]["errors"] is not None
        assert len(result["metadata"]["errors"]) == 2


class TestOrchestratorIntegration:
    """Test CompositeNode integration with orchestrator configuration."""

    @pytest.mark.asyncio
    async def test_body_executor_receives_orchestrator_config(self) -> None:
        """Test that BodyExecutor receives orchestrator configuration."""

        # Create a node with custom orchestrator configuration
        node_spec = CompositeNode()(
            name="config_test",
            mode="times",
            count=1,
            max_concurrent_nodes=5,
            strict_validation=True,
            default_node_timeout=30.0,
        )

        # The node spec should be created successfully
        assert node_spec is not None
        assert node_spec.name == "config_test"

    @pytest.mark.asyncio
    async def test_body_executor_can_be_imported_from_orchestration(self) -> None:
        """Test BodyExecutor is exported from orchestration module."""
        from hexdag.kernel.orchestration import BodyExecutor

        # Create executor with config
        executor = BodyExecutor(
            max_concurrent_nodes=5,
            strict_validation=True,
            default_node_timeout=30.0,
        )

        assert executor.max_concurrent_nodes == 5
        assert executor.strict_validation is True
        assert executor.default_node_timeout == 30.0

    @pytest.mark.asyncio
    async def test_composite_with_custom_config_executes(self) -> None:
        """Test CompositeNode with custom config executes correctly."""

        def simple_body(data: dict, **ports: dict) -> int:
            return data.get("$index", 0) * 2

        node_spec = CompositeNode()(
            name="custom_config_test",
            mode="times",
            count=3,
            body=simple_body,
            max_concurrent_nodes=2,
            strict_validation=False,
            collect="list",
        )

        result = await node_spec.fn({})

        assert result["result"] == [0, 2, 4]
        assert result["metadata"]["iterations"] == 3
