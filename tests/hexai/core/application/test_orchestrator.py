"""Tests for the Orchestrator DAG execution engine."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hexai.core.application.orchestrator import NodeExecutionError, Orchestrator, OrchestratorError
from hexai.core.bootstrap import ensure_bootstrapped
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.registry import registry


# Test data models
class ProcessorOutput(BaseModel):
    """Test output model for processor node."""

    text: str
    metadata: dict[str, str] = {}


class ValidatorOutput(BaseModel):
    """Test output model for validator node."""

    status: str
    score: float


# Test helper functions
async def async_add_one(x: int, **ports) -> int:
    """Async function that adds 1 to input."""
    return x + 1


def sync_multiply_two(x: int, **ports) -> int:
    """Sync function that multiplies by 2."""
    return x * 2


async def async_combine(inputs: dict, **ports) -> int:
    """Async function that sums multiple inputs."""
    return sum(inputs.values())


def failing_function(x: int, **ports) -> int:
    """Raise an exception."""
    raise ValueError("Intentional test failure")


async def async_with_memory(x: int, **ports) -> int:
    """Process value with ports."""
    # Observers are for observability only, not memory storage
    return x + 10


async def processor_function(input_data: str, **ports) -> ProcessorOutput:
    """Process text input."""
    return ProcessorOutput(text=f"processed_{input_data}", metadata={"lang": "en"})


async def validator_function(input_data: str, **ports) -> ValidatorOutput:
    """Validate input."""
    return ValidatorOutput(status="valid", score=0.95)


class MappedConsumerInput(BaseModel):
    """Input model for mapped consumer."""

    processor: ProcessorOutput
    validator: ValidatorOutput


async def mapper_consumer(input_data: MappedConsumerInput, **ports) -> str:
    """Consume mapped data using Pydantic model."""
    content = input_data.processor.text
    language = input_data.processor.metadata.get("lang", "unknown")
    validation_status = input_data.validator.status
    return f"{content} ({language}) - {validation_status}"


async def structured_consumer(input_data: dict, **ports) -> str:
    """Consume structured aggregated data."""
    processor_data = input_data.get("processor")
    validator_data = input_data.get("validator")

    # Handle Pydantic models
    text = processor_data.text if hasattr(processor_data, "text") else ""
    status = validator_data.status if hasattr(validator_data, "status") else "unknown"

    return f"{text} - {status}"


class TestOrchestrator:
    """Test cases for the Orchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing."""
        return Orchestrator()

    @pytest.fixture
    def observers(self):
        """Create mock observer manager for testing."""
        from hexai.core.application.events import ObserverManager

        mock = AsyncMock(spec=ObserverManager)
        return mock

    @pytest.mark.asyncio
    async def test_simple_sequential_execution(self, orchestrator, observers):
        """Test basic sequential DAG execution."""
        # Create simple DAG: input -> add_one -> multiply_two
        graph = DirectedGraph()
        graph.add(NodeSpec("add_one", async_add_one))
        graph.add(NodeSpec("multiply_two", sync_multiply_two).after("add_one"))

        results = await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert "add_one" in results
        assert "multiply_two" in results
        assert results["add_one"] == 6  # 5 + 1
        assert results["multiply_two"] == 12  # 6 * 2

        # Check events were notified
        assert (
            observers.notify.call_count >= 6
        )  # Pipeline started, 2x wave started, 2x wave completed, pipeline completed

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_execution(self, orchestrator, observers):
        """Test parallel DAG execution with fan-out via orchestrator."""
        # Create diamond DAG: input -> (branch_a, branch_b) -> combine
        graph = DirectedGraph()
        graph.add(NodeSpec("branch_a", async_add_one))
        graph.add(NodeSpec("branch_b", sync_multiply_two))
        graph.add(NodeSpec("combine", async_combine).after("branch_a", "branch_b"))

        results = await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert "branch_a" in results
        assert "branch_b" in results
        assert "combine" in results
        assert results["branch_a"] == 6  # 5 + 1
        assert results["branch_b"] == 10  # 5 * 2
        assert results["combine"] == 16  # 6 + 10

    @pytest.mark.asyncio
    async def test_complex_dag_execution(self, orchestrator, observers):
        """Test complex DAG with multiple waves."""
        # Complex DAG:
        #   start -> (process_a, process_b) -> (result_a, result_b) -> final
        graph = DirectedGraph()
        nodes = [
            NodeSpec("start", async_add_one),
            NodeSpec("process_a", sync_multiply_two).after("start"),
            NodeSpec("process_b", async_add_one).after("start"),
            NodeSpec("result_a", async_add_one).after("process_a"),
            NodeSpec("result_b", sync_multiply_two).after("process_b"),
            NodeSpec("final", async_combine).after("result_a", "result_b"),
        ]
        graph.add_many(*nodes)

        results = await orchestrator.run(graph, 3, additional_ports={"observer_manager": observers})

        # Verify execution flow:
        # start: 3 + 1 = 4
        # process_a: 4 * 2 = 8, process_b: 4 + 1 = 5
        # result_a: 8 + 1 = 9, result_b: 5 * 2 = 10
        # final: 9 + 10 = 19
        assert results["start"] == 4
        assert results["process_a"] == 8
        assert results["process_b"] == 5
        assert results["result_a"] == 9
        assert results["result_b"] == 10
        assert results["final"] == 19

    @pytest.mark.asyncio
    async def test_observers_memory_usage(self, orchestrator, observers):
        """Test that nodes can receive observers through ports."""
        graph = DirectedGraph()
        graph.add(NodeSpec("memory_node", async_with_memory))

        results = await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert results["memory_node"] == 15  # 5 + 10
        # Observers are for observability, not memory storage

    @pytest.mark.asyncio
    async def test_node_execution_error(self, orchestrator, observers):
        """Test handling of node execution errors."""
        graph = DirectedGraph()
        graph.add(NodeSpec("good_node", async_add_one))
        graph.add(NodeSpec("bad_node", failing_function).after("good_node"))

        with pytest.raises(NodeExecutionError) as exc_info:
            await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert exc_info.value.node_name == "bad_node"
        assert "Intentional test failure" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, ValueError)

    @pytest.mark.asyncio
    async def test_invalid_dag_error(self, orchestrator, observers):
        """Test handling of invalid DAG structures."""
        # Create DAG with cycle
        graph = DirectedGraph()
        graph.add(NodeSpec("a", async_add_one).after("b"))
        graph.add(NodeSpec("b", sync_multiply_two).after("a"))

        with pytest.raises(OrchestratorError) as exc_info:
            await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert "Invalid DAG" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_dag(self, orchestrator, observers):
        """Test execution of empty DAG."""
        graph = DirectedGraph()
        results = await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})
        assert results == {}

    @pytest.mark.asyncio
    async def test_single_node_dag(self, orchestrator, observers):
        """Test execution of single-node DAG."""
        graph = DirectedGraph()
        graph.add(NodeSpec("only_node", async_add_one))

        results = await orchestrator.run(
            graph, 10, additional_ports={"observer_manager": observers}
        )

        assert len(results) == 1
        assert results["only_node"] == 11

    @pytest.mark.asyncio
    async def test_kwargs_passing(self, orchestrator, observers):
        """Test that kwargs are passed to node functions."""

        async def node_with_kwargs(x: int, multiplier: int = 1, **ports) -> int:
            return x * multiplier

        graph = DirectedGraph()
        graph.add(NodeSpec("kwarg_node", node_with_kwargs))

        results = await orchestrator.run(
            graph, 5, additional_ports={"observer_manager": observers}, multiplier=3
        )

        assert results["kwarg_node"] == 15  # 5 * 3

    @pytest.mark.asyncio
    async def test_multiple_dependency_input_preparation(self, orchestrator, observers):
        """Test input preparation for nodes with multiple dependencies."""

        async def analyzer(inputs: dict, **ports) -> str:
            """Expect dict input from multiple dependencies."""
            values = list(inputs.values())
            return f"sum={sum(values)}"

        graph = DirectedGraph()
        graph.add(NodeSpec("a", async_add_one))
        graph.add(NodeSpec("b", sync_multiply_two))
        graph.add(NodeSpec("analyze", analyzer).after("a", "b"))

        results = await orchestrator.run(graph, 5, additional_ports={"observer_manager": observers})

        assert results["a"] == 6  # 5 + 1
        assert results["b"] == 10  # 5 * 2
        assert results["analyze"] == "sum=16"  # 6 + 10

    @pytest.mark.asyncio
    async def test_wave_based_execution_order(self, orchestrator, observers):
        """Test that execution follows proper wave-based ordering."""
        execution_order = []

        async def track_execution(name: str, expect_dict: bool = False):
            if expect_dict:

                async def wrapped_dict(x: dict, **ports) -> int:
                    execution_order.append(name)
                    await asyncio.sleep(0.01)
                    return sum(x.values()) + 1

                return wrapped_dict
            else:

                async def wrapped_int(x: int, **ports) -> int:
                    execution_order.append(name)
                    await asyncio.sleep(0.01)
                    return x + 1

                return wrapped_int

        # Create DAG where wave 1 = [a], wave 2 = [b, c], wave 3 = [d]
        graph = DirectedGraph()
        graph.add(NodeSpec("a", await track_execution("a")))
        graph.add(NodeSpec("b", await track_execution("b")).after("a"))
        graph.add(NodeSpec("c", await track_execution("c")).after("a"))
        graph.add(NodeSpec("d", await track_execution("d", expect_dict=True)).after("b", "c"))

        await orchestrator.run(graph, 1, additional_ports={"observer_manager": observers})

        # Check execution order
        assert execution_order[0] == "a"  # Wave 1 first
        assert execution_order[-1] == "d"  # Wave 3 last
        # b and c should be in middle (wave 2), order within wave doesn't matter
        assert "b" in execution_order[1:3]
        assert "c" in execution_order[1:3]

    @pytest.mark.asyncio
    async def test_ports_flag_injection(self, orchestrator, observers):
        """Test dummy node reads ports['flag'] == 42."""

        def dummy_node_with_flag_check(input_data, flag=None, **ports):
            """Check for flag port."""
            # Verify the flag is what we expect
            assert flag == 42
            return {"result": "success", "flag_value": flag, "input": input_data}

        # Create a simple DAG with one node
        graph = DirectedGraph()
        graph.add(NodeSpec("dummy_flag_check", dummy_node_with_flag_check))

        # Define ports with flag
        ports = {"flag": 42, "observers": observers}

        # Execute the DAG
        results = await orchestrator.run(graph, "test_input", additional_ports=ports)

        # Verify the result
        assert "dummy_flag_check" in results
        result = results["dummy_flag_check"]
        assert result["result"] == "success"
        assert result["flag_value"] == 42
        assert result["input"] == "test_input"

    @pytest.mark.asyncio
    async def test_ports_with_mocks(self, orchestrator, observers):
        """Test orchestrator with mock LLM and ToolRouter ports."""
        from hexai.adapters.mock.mock_llm import MockLLM
        from hexai.adapters.unified_tool_router import UnifiedToolRouter

        async def async_dummy_node_with_ports(input_data, llm=None, tool_router=None, **ports):
            """Async dummy node that uses ports."""
            result = {"input": input_data}

            if llm:
                llm_response = await llm.aresponse([{"role": "user", "content": "test"}])
                result["llm_response"] = llm_response

            if tool_router:
                tool_result = await tool_router.acall_tool(
                    "test_tool", {"input_data": "test_input"}
                )
                result["tool_result"] = tool_result

            return result

        # Create a DAG with an async node
        graph = DirectedGraph()
        graph.add(NodeSpec("async_node", async_dummy_node_with_ports))

        # Create mock ports - use UnifiedToolRouter with a simple mock function
        mock_llm = MockLLM(responses=["Test LLM response"])
        mock_tool_router = UnifiedToolRouter()

        # Add a simple mock tool by registering it with the global registry
        from hexai.core.registry import registry, tool
        from hexai.core.registry.models import ComponentType

        @tool(name="test_tool", namespace="test")
        def mock_test_tool(input_data):
            return f"Mock tool result for: {input_data}"

        # Register the tool with the global registry
        registry.register(
            name="test_tool",
            component=mock_test_tool,
            component_type=ComponentType.TOOL,
            namespace="test",
        )
        ports = {"llm": mock_llm, "tool_router": mock_tool_router, "observers": observers}

        # Execute the DAG
        results = await orchestrator.run(graph, "test_input", additional_ports=ports)

        # Verify the result
        result = results["async_node"]
        assert result["input"] == "test_input"
        assert result["llm_response"] == "Test LLM response"
        assert "test_input" in result["tool_result"]

    @pytest.mark.asyncio
    async def test_ports_none_defaults_to_empty(self, orchestrator, observers):
        """Test that ports=None defaults to empty dict."""

        def sync_dummy_node_with_memory(input_data, memory=None, **ports):
            """Sync dummy node that uses memory port."""
            if memory is None:
                return {"error": "No memory port provided"}

            return {"memory_available": True, "input": input_data}

        # Create a simple DAG
        graph = DirectedGraph()
        graph.add(NodeSpec("sync_node", sync_dummy_node_with_memory))

        # Execute without ports (should default to {})
        results = await orchestrator.run(
            graph, "test_input", additional_ports={"observer_manager": observers}
        )

        # Verify the node handled missing memory port gracefully
        result = results["sync_node"]
        assert result["error"] == "No memory port provided"

    @pytest.mark.asyncio
    async def test_ports_with_additional_kwargs(self, orchestrator, observers):
        """Test that ports work alongside additional kwargs."""

        def node_with_ports_and_kwargs(input_data, flag=None, extra_param=None, **kwargs):
            return {"flag": flag, "extra_param": extra_param, "input": input_data, "kwargs": kwargs}

        graph = DirectedGraph()
        graph.add(NodeSpec("mixed_node", node_with_ports_and_kwargs))

        # Execute with both ports and additional kwargs
        results = await orchestrator.run(
            graph,
            "test_input",
            additional_ports={"flag": 42, "observers": observers},
            extra_param="from_kwargs",
            another_kwarg="additional",
        )

        result = results["mixed_node"]
        assert result["flag"] == 42
        assert result["extra_param"] == "from_kwargs"
        assert result["input"] == "test_input"
        assert result["kwargs"]["another_kwarg"] == "additional"

    @pytest.mark.asyncio
    async def test_input_validation_with_pydantic_model(self, orchestrator, observers):
        """Test input validation using Pydantic models."""

        class ProcessingInput(BaseModel):
            text: str
            priority: int = 1

        # Node that expects specific input format
        async def strict_processor(input_data: ProcessingInput, **ports) -> dict:
            """Node that requires validated input (receives Pydantic model)."""
            return {
                "processed_text": input_data.text.upper(),
                "priority_level": input_data.priority * 10,
            }

        # Create DAG with input validation
        graph = DirectedGraph()
        node_spec = NodeSpec(
            "strict_processor",
            strict_processor,
            in_model=ProcessingInput,  # This will trigger validation
        )
        graph.add(node_spec)

        # Test with valid input
        valid_input = {"text": "hello world", "priority": 2}
        results = await orchestrator.run(
            graph, valid_input, additional_ports={"observer_manager": observers}
        )

        assert "strict_processor" in results
        result = results["strict_processor"]
        assert result["processed_text"] == "HELLO WORLD"
        assert result["priority_level"] == 20

    @pytest.mark.asyncio
    async def test_input_validation_failure(self, orchestrator, observers):
        """Test that input validation failures are properly handled."""
        # Use strict validation to ensure errors are raised
        orchestrator = Orchestrator(strict_validation=True)

        class StrictInput(BaseModel):
            required_field: str
            number_field: int

        async def validate_node(input_data: StrictInput, **ports) -> dict:
            return {"status": "processed"}

        graph = DirectedGraph()
        node_spec = NodeSpec("validate_node", validate_node, in_model=StrictInput)
        graph.add(node_spec)

        # Test with invalid input (missing required_field)
        invalid_input = {"number_field": 42}  # Missing required_field

        with pytest.raises(NodeExecutionError) as exc_info:
            await orchestrator.run(
                graph, invalid_input, additional_ports={"observer_manager": observers}
            )

        # Check that error mentions input validation
        error_str = str(exc_info.value)
        assert "Input validation failed" in error_str
        assert "validate_node" in error_str

    @pytest.mark.asyncio
    async def test_node_to_node_validation(self, orchestrator, observers):
        """Test validation of data flowing between nodes."""

        class NodeAOutput(BaseModel):
            result: str
            score: float

        class NodeBInput(BaseModel):
            result: str
            score: float
            threshold: float = 0.5

        async def producer_node(input_data: str, **ports) -> dict:
            """Produce data in expected format."""
            return {"result": input_data.upper(), "score": 0.8}

        async def consumer_node(input_data: NodeBInput, **ports) -> dict:
            """Consume validated data (receives Pydantic model)."""
            passed = input_data.score >= input_data.threshold
            return {"analysis": input_data.result, "passed_threshold": passed}

        # Create DAG with validation chain
        graph = DirectedGraph()
        graph.add(NodeSpec("producer", producer_node, out_model=NodeAOutput))
        graph.add(NodeSpec("consumer", consumer_node, in_model=NodeBInput).after("producer"))

        results = await orchestrator.run(
            graph, "test data", additional_ports={"observer_manager": observers}
        )

        # Check that data flowed correctly through validation
        producer_result = results["producer"]
        assert isinstance(producer_result, NodeAOutput)
        assert producer_result.result == "TEST DATA"
        assert producer_result.score == 0.8
        assert results["consumer"]["analysis"] == "TEST DATA"
        assert results["consumer"]["passed_threshold"] is True

    @pytest.mark.asyncio
    async def test_type_mismatch_validation(self, observers):
        """Test validation with basic type mismatches."""
        # Use strict validation for this test

        strict_orchestrator = Orchestrator(strict_validation=True)

        async def string_only_node(input_data: str, **ports) -> str:
            return input_data.upper()

        graph = DirectedGraph()
        node_spec = NodeSpec(
            "string_node", string_only_node
        )  # No type checking without Pydantic models
        graph.add(node_spec)

        # Pass dict when expecting string
        with pytest.raises(NodeExecutionError) as exc_info:
            await strict_orchestrator.run(
                graph, {"not": "a string"}, additional_ports={"observer_manager": observers}
            )

        error_str = str(exc_info.value)
        # With no Pydantic validation, the error comes from the function itself
        assert "'dict' object has no attribute 'upper'" in error_str

    @pytest.mark.asyncio
    async def test_node_to_node_validation_strict(self, observers):
        """Test validation of data flowing between nodes using strict Pydantic models."""
        # Use strict validation for this test

        strict_orchestrator = Orchestrator(strict_validation=True)

        class NodeAOutput(BaseModel):
            result: str
            score: float

        class NodeBInput(BaseModel):
            result: str
            score: float
            threshold: float = 0.5

        async def producer_node(input_data: str, **ports) -> dict:
            """Produce data in expected format."""
            return {"result": input_data.upper(), "score": 0.8}

        async def consumer_node(input_data: NodeBInput, **ports) -> dict:
            """Consume validated data (receives Pydantic model)."""
            passed = input_data.score >= input_data.threshold
            return {"analysis": input_data.result, "passed_threshold": passed}

        # Create DAG with validation chain
        graph = DirectedGraph()
        graph.add(NodeSpec("producer", producer_node, out_model=NodeAOutput))
        graph.add(NodeSpec("consumer", consumer_node, in_model=NodeBInput).after("producer"))

        results = await strict_orchestrator.run(
            graph, "test data", additional_ports={"observer_manager": observers}
        )

        # Check that data flowed correctly through validation - expect Pydantic models
        producer_result = results["producer"]
        assert isinstance(producer_result, NodeAOutput)
        assert producer_result.result == "TEST DATA"
        assert producer_result.score == 0.8
        assert results["consumer"]["analysis"] == "TEST DATA"
        assert results["consumer"]["passed_threshold"] is True

    @pytest.mark.asyncio
    async def test_graph_level_schema_validation(self, orchestrator, observers):
        """Test that graph validates schema compatibility at construction time."""

        class OutputSchemaA(BaseModel):
            data: str

        class InputSchemaB(BaseModel):
            different_field: int  # Incompatible with OutputSchemaA - no default!

        async def producer(input_data: str, **ports) -> dict:
            return {"data": input_data}

        async def consumer(input_data: InputSchemaB, **ports) -> dict:
            return {"processed": input_data.different_field}

        # Build DAG with incompatible schemas
        graph = DirectedGraph()
        graph.add(NodeSpec("producer", producer, out_model=OutputSchemaA))
        graph.add(NodeSpec("consumer", consumer, in_model=InputSchemaB).after("producer"))

        # Should fail at runtime when data flows between incompatible nodes
        with pytest.raises(NodeExecutionError) as exc_info:
            await orchestrator.run(graph, "test")

        error_str = str(exc_info.value)
        assert "consumer" in error_str
        # The error occurs when trying to access a field that doesn't exist
        assert "has no attribute" in error_str or "validation failed" in error_str.lower()

    @pytest.mark.asyncio
    async def test_graph_level_compatible_schemas(self, orchestrator, observers):
        """Test that compatible schemas pass graph validation."""

        class SharedSchema(BaseModel):
            text: str
            value: int = 10

        async def producer(input_data: str, **ports) -> dict:
            return {"text": input_data, "value": 42}

        async def consumer(input_data: SharedSchema, **ports) -> dict:
            return {"result": f"{input_data.text}:{input_data.value}"}

        # Build DAG with compatible schemas
        graph = DirectedGraph()
        graph.add(NodeSpec("producer", producer, out_model=SharedSchema))
        graph.add(NodeSpec("consumer", consumer, in_model=SharedSchema).after("producer"))

        # Should pass validation
        graph.validate()  # Should not raise

        # Should execute successfully
        results = await orchestrator.run(
            graph, "hello", additional_ports={"observer_manager": observers}
        )
        assert results["consumer"]["result"] == "hello:42"

    @pytest.mark.asyncio
    async def test_validation_disabled(self, orchestrator, observers):
        """Test that validation can be disabled."""
        graph = DirectedGraph()
        graph.add(NodeSpec("test_node", async_add_one))

        # Should work with validation disabled
        results = await orchestrator.run(
            graph, 5, additional_ports={"observer_manager": observers}, validate=False
        )
        assert results["test_node"] == 6

    @pytest.mark.asyncio
    async def test_orchestrator_with_shared_ports(self, observers):
        """Test orchestrator with shared ports in constructor."""
        # Create orchestrator with shared ports
        from hexai.adapters.local import LocalPolicyManager

        shared_ports = {
            "database": "mock_db",
            "cache": "mock_cache",
            "policy_manager": LocalPolicyManager(),
        }
        orchestrator = Orchestrator(ports=shared_ports)

        def node_with_shared_ports(input_data, database=None, cache=None, **ports):
            return {"database": database, "cache": cache, "input": input_data}

        graph = DirectedGraph()
        graph.add(NodeSpec("test_node", node_with_shared_ports))

        # Execute with additional ports
        results = await orchestrator.run(
            graph, "test_input", additional_ports={"observer_manager": observers}
        )

        # Should have access to both shared and additional ports
        result = results["test_node"]
        assert result["database"] == "mock_db"
        assert result["cache"] == "mock_cache"
        assert result["input"] == "test_input"

    @pytest.mark.asyncio
    async def test_observers_as_port(self, orchestrator):
        """Test that event manager is passed as a port to nodes."""

        async def node_with_observers(input_data, observer_manager=None, **ports):
            """Node that uses event manager from ports."""
            if observer_manager:
                await observer_manager.emit(MagicMock())  # Mock event
            return {"used_observers": observer_manager is not None}

        graph = DirectedGraph()
        graph.add(NodeSpec("test_node", node_with_observers))

        # Test with event manager
        observers = AsyncMock()
        results = await orchestrator.run(
            graph, "test_input", additional_ports={"observer_manager": observers}
        )

        assert results["test_node"]["used_observers"] is True

        # Test without providing observer_manager explicitly
        # The orchestrator fixture has defaults which includes observer_manager
        # So the node will still receive an observer_manager
        results = await orchestrator.run(graph, "test_input")
        assert (
            results["test_node"]["used_observers"] is True
        )  # True because orchestrator has default observer_manager

    @pytest.mark.asyncio
    async def test_data_mapping_functionality(self):
        """Test data mapping between nodes using Pydantic models."""
        orchestrator = Orchestrator()

        # Create nodes
        processor_node = NodeSpec(
            name="processor",
            fn=processor_function,
            in_model=str,
            out_model=ProcessorOutput,
            deps=set(),
        )

        validator_node = NodeSpec(
            name="validator",
            fn=validator_function,
            in_model=str,
            out_model=ValidatorOutput,
            deps=set(),
        )

        ensure_bootstrapped()
        function_node = registry.get("function_node", namespace="core")

        consumer_node = function_node(
            name="consumer",
            fn=mapper_consumer,
            input_schema=MappedConsumerInput,
            output_schema=str,
            deps=["processor", "validator"],
        )

        # Create DAG
        graph = DirectedGraph([processor_node, validator_node, consumer_node])

        # Execute pipeline
        result = await orchestrator.run(graph, "test_input")

        # Verify data mapping worked through Pydantic
        expected = "processed_test_input (en) - valid"
        assert result["consumer"] == expected

    @pytest.mark.asyncio
    async def test_structured_aggregation(self):
        """Test structured data aggregation preserving namespaces."""
        orchestrator = Orchestrator()

        # Create nodes that will be aggregated
        processor_node = NodeSpec(
            name="processor",
            fn=processor_function,
            in_model=str,
            out_model=ProcessorOutput,
            deps=set(),
        )

        validator_node = NodeSpec(
            name="validator",
            fn=validator_function,
            in_model=str,
            out_model=ValidatorOutput,
            deps=set(),
        )

        # Consumer without explicit mapping - uses structured aggregation
        from hexai.core.bootstrap import ensure_bootstrapped
        from hexai.core.registry import registry

        ensure_bootstrapped()
        function_node = registry.get("function_node", namespace="core")

        consumer_node = function_node(
            name="consumer",
            fn=structured_consumer,
            input_schema=dict,
            output_schema=str,
            deps=["processor", "validator"],
        )

        # Create DAG
        graph = DirectedGraph([processor_node, validator_node, consumer_node])

        # Execute pipeline
        result = await orchestrator.run(graph, "test_input")

        # Verify structured aggregation preserved namespaces
        expected = "processed_test_input - valid"
        assert result["consumer"] == expected

    async def test_custom_field_mappings_runtime(self):
        """Test data mapping using Pydantic models."""
        orchestrator = Orchestrator()

        # Source node that produces specific field names
        class SourceOutput(BaseModel):
            name: str
            content: str

        async def source_function(input_data: str, **ports) -> SourceOutput:
            return SourceOutput(name=f"processed_{input_data}", content="test content")

        # Consumer receives the source output directly due to single dependency
        async def custom_consumer(input_data: SourceOutput, **ports) -> str:
            title = input_data.name
            description = input_data.content
            return f"{title}: {description}"

        source_node = NodeSpec(
            name="source",
            fn=source_function,
            in_model=str,
            out_model=SourceOutput,
            deps=set(),
        )

        # Consumer using Pydantic model

        ensure_bootstrapped()
        function_node = registry.get("function_node", namespace="core")

        consumer_node = function_node(
            name="consumer",
            fn=custom_consumer,
            input_schema=SourceOutput,
            output_schema=str,
            deps=["source"],
        )

        graph = DirectedGraph([source_node, consumer_node])

        # Execute the graph
        result = await orchestrator.run(graph, "test")

        assert result["consumer"] == "processed_test: test content"

    async def test_single_dependency_passthrough(self):
        """Test that single dependencies pass through directly."""
        orchestrator = Orchestrator()

        # First node
        first_node = NodeSpec(
            name="first",
            fn=processor_function,
            in_model=str,
            out_model=ProcessorOutput,
            deps=set(),
        )

        # Second node depends only on first - should get direct passthrough
        async def single_dep_consumer(input_data: ProcessorOutput, **ports) -> str:
            return f"consumed: {input_data.text}"

        second_node = NodeSpec(
            name="second",
            fn=single_dep_consumer,
            in_model=ProcessorOutput,
            out_model=str,
            deps={"first"},
        )

        graph = DirectedGraph([first_node, second_node])

        result = await orchestrator.run(graph, "test")

        assert result["second"] == "consumed: processed_test"

    @pytest.mark.asyncio
    async def test_mixed_mapping_and_aggregation(self):
        """Test pipeline with data mapping and aggregation using Pydantic."""
        orchestrator = Orchestrator()

        # Source nodes
        processor_node = NodeSpec(
            name="processor",
            fn=processor_function,
            in_model=str,
            out_model=ProcessorOutput,
            deps=set(),
        )

        validator_node = NodeSpec(
            name="validator",
            fn=validator_function,
            in_model=str,
            out_model=ValidatorOutput,
            deps=set(),
        )

        # Node using Pydantic model for structured input

        ensure_bootstrapped()
        function_node = registry.get("function_node", namespace="core")

        mapped_node = function_node(
            name="mapped",
            fn=mapper_consumer,
            input_schema=MappedConsumerInput,
            output_schema=str,
            deps=["processor", "validator"],
        )

        # Node with structured aggregation (depends on mapped result)
        async def final_consumer(input_data: str, **ports) -> str:
            return f"final: {input_data}"

        final_node = function_node(
            name="final",
            fn=final_consumer,
            input_schema=str,
            output_schema=str,
            deps=["mapped"],
        )

        graph = DirectedGraph([processor_node, validator_node, mapped_node, final_node])

        result = await orchestrator.run(graph, "test")

        expected = "final: processed_test (en) - valid"
        assert result["final"] == expected
