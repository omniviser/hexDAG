"""Tests for PortCallNode."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hexdag.kernel.context import set_ports
from hexdag.stdlib.nodes.port_call_node import PortCallNode


class TestPortCallNodeCreation:
    """Test PortCallNode factory creation."""

    def test_creates_node_spec(self) -> None:
        """Test basic node spec creation."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="aexecute_query",
        )

        assert spec.name == "test_node"
        assert spec.fn is not None
        assert spec.deps == frozenset()

    def test_creates_node_spec_with_dependencies(self) -> None:
        """Test node spec with dependencies."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="aexecute_query",
            deps=["prev_node", "other_node"],
        )

        assert spec.name == "test_node"
        assert "prev_node" in spec.deps
        assert "other_node" in spec.deps

    def test_creates_node_spec_with_input_mapping(self) -> None:
        """Test node spec with input mapping stores mapping in params."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="aexecute_query",
            input_mapping={"query": "$input.sql", "params": "prev.data"},
            deps=["prev"],
        )

        assert spec.name == "test_node"
        assert spec.params.get("input_mapping") == {
            "query": "$input.sql",
            "params": "prev.data",
        }
        assert "prev" in spec.deps

    def test_function_name_set_correctly(self) -> None:
        """Test that the wrapped function has correct name."""
        factory = PortCallNode()
        spec = factory(
            name="my_call",
            port="database",
            method="aexecute_query",
        )

        assert spec.fn.__name__ == "port_call_my_call"


class TestPortCallNodeExecution:
    """Test PortCallNode execution."""

    @pytest.mark.asyncio
    async def test_executes_async_method(self) -> None:
        """Test execution of async port method."""
        # Create mock adapter with async method
        mock_adapter = AsyncMock()
        mock_adapter.aexecute_query = AsyncMock(return_value=[{"id": 1}, {"id": 2}])

        # Set up context with the mock port
        set_ports({"database": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="aexecute_query",
        )

        result = await spec.fn({"query": "SELECT * FROM users"})

        assert result["result"] == [{"id": 1}, {"id": 2}]
        assert result["port"] == "database"
        assert result["method"] == "aexecute_query"
        assert result["error"] is None
        mock_adapter.aexecute_query.assert_called_once_with(query="SELECT * FROM users")

        # Clear context
        set_ports(None)

    @pytest.mark.asyncio
    async def test_executes_sync_method(self) -> None:
        """Test execution of sync port method."""
        # Create mock adapter with sync method
        mock_adapter = MagicMock()
        mock_adapter.get_value = MagicMock(return_value="cached_value")

        # Set up context with the mock port
        set_ports({"cache": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="cache",
            method="get_value",
        )

        result = await spec.fn({"key": "my_key"})

        assert result["result"] == "cached_value"
        assert result["port"] == "cache"
        assert result["method"] == "get_value"
        assert result["error"] is None
        mock_adapter.get_value.assert_called_once_with(key="my_key")

        # Clear context
        set_ports(None)

    @pytest.mark.asyncio
    async def test_passes_multiple_kwargs(self) -> None:
        """Test that multiple kwargs are passed to method."""
        mock_adapter = AsyncMock()
        mock_adapter.record_data = AsyncMock(return_value={"success": True})

        set_ports({"database": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="record_data",
        )

        result = await spec.fn({
            "record_id": "123",
            "value": "test_value",
            "timestamp": "2024-01-01",
        })

        assert result["result"] == {"success": True}
        mock_adapter.record_data.assert_called_once_with(
            record_id="123",
            value="test_value",
            timestamp="2024-01-01",
        )

        set_ports(None)


class TestPortCallNodeFallback:
    """Test PortCallNode fallback behavior."""

    @pytest.mark.asyncio
    async def test_uses_fallback_when_port_missing(self) -> None:
        """Test fallback behavior when port is not available."""
        # Set up empty ports
        set_ports({})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="cache",
            method="aget",
            fallback={"cached": False},
            has_fallback=True,
        )

        result = await spec.fn({"key": "my_key"})

        assert result["result"] == {"cached": False}
        assert result["error"] is not None
        assert "not available" in result["error"]

        set_ports(None)

    @pytest.mark.asyncio
    async def test_uses_none_fallback(self) -> None:
        """Test that None can be used as fallback value."""
        set_ports({})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="cache",
            method="aget",
            fallback=None,
            has_fallback=True,
        )

        result = await spec.fn({"key": "my_key"})

        assert result["result"] is None
        assert result["error"] is not None

        set_ports(None)

    @pytest.mark.asyncio
    async def test_uses_fallback_on_method_error(self) -> None:
        """Test fallback when method raises exception."""
        mock_adapter = AsyncMock()
        mock_adapter.risky_operation = AsyncMock(side_effect=ValueError("Something went wrong"))

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="service",
            method="risky_operation",
            fallback="default_value",
            has_fallback=True,
        )

        result = await spec.fn({})

        assert result["result"] == "default_value"
        assert result["error"] == "Something went wrong"

        set_ports(None)


class TestPortCallNodeErrors:
    """Test PortCallNode error handling."""

    @pytest.mark.asyncio
    async def test_raises_when_port_missing_no_fallback(self) -> None:
        """Test error when port missing and no fallback."""
        set_ports({})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="missing_port",
            method="some_method",
        )

        with pytest.raises(RuntimeError, match="not available"):
            await spec.fn({})

        set_ports(None)

    @pytest.mark.asyncio
    async def test_raises_when_method_missing(self) -> None:
        """Test error when method doesn't exist on port."""
        mock_adapter = MagicMock(spec=["existing_method"])

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="service",
            method="nonexistent_method",
        )

        with pytest.raises(AttributeError, match="no method"):
            await spec.fn({})

        set_ports(None)

    @pytest.mark.asyncio
    async def test_raises_method_error_without_fallback(self) -> None:
        """Test that method errors are re-raised without fallback."""
        mock_adapter = AsyncMock()
        mock_adapter.failing_method = AsyncMock(side_effect=ValueError("Method failed"))

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="service",
            method="failing_method",
        )

        with pytest.raises(ValueError, match="Method failed"):
            await spec.fn({})

        set_ports(None)


class TestPortCallNodeWithOrchestrator:
    """Integration tests with actual orchestrator execution."""

    @pytest.mark.asyncio
    async def test_in_dag_execution(self) -> None:
        """Test port_call_node works in a full DAG execution."""
        from hexdag.kernel.domain.dag import DirectedGraph
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        # Create mock adapter
        mock_db = AsyncMock()
        mock_db.save_record = AsyncMock(return_value={"saved": True, "id": "abc123"})

        # Create the node
        factory = PortCallNode()
        node = factory(
            name="save_data",
            port="database",
            method="save_record",
        )

        # Build DAG
        graph = DirectedGraph()
        graph.add(node)

        # Create orchestrator and run
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {"name": "test", "value": 42},
            additional_ports={"database": mock_db},
        )

        # Verify
        assert "save_data" in results
        assert results["save_data"]["result"] == {"saved": True, "id": "abc123"}
        mock_db.save_record.assert_called_once_with(name="test", value=42)

    @pytest.mark.asyncio
    async def test_with_expression_node_dependency(self) -> None:
        """Test port_call_node with ExpressionNode dependency."""
        from hexdag.kernel.domain.dag import DirectedGraph
        from hexdag.kernel.orchestration.orchestrator import Orchestrator
        from hexdag.stdlib.nodes.expression_node import ExpressionNode

        # Create mock adapter
        mock_db = AsyncMock()
        mock_db.save_record = AsyncMock(return_value={"saved": True})

        # Create nodes
        expr_factory = ExpressionNode()
        expr_node = expr_factory(
            name="static_config",
            expressions={"action": "'ACCEPT'", "reason": "'Approved'"},
            output_fields=["action", "reason"],
        )

        port_factory = PortCallNode()
        port_node = port_factory(
            name="save_action",
            port="database",
            method="save_record",
            deps=["static_config"],
        )

        # Build DAG
        graph = DirectedGraph()
        graph.add(expr_node)
        graph.add(port_node)

        # Create orchestrator and run
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {},
            additional_ports={"database": mock_db},
        )

        # Verify execution order and results
        assert "static_config" in results
        assert "save_action" in results
        assert results["static_config"]["action"] == "ACCEPT"


class TestStaticNodeAlias:
    """Test that static_node is a valid alias for DataNode."""

    def test_static_node_resolves_to_data_node(self) -> None:
        """Test that static_node alias resolves correctly."""
        from hexdag.kernel.resolver import resolve

        StaticNode = resolve("static_node")
        DataNode = resolve("data_node")

        assert StaticNode is DataNode

    def test_static_node_alias_resolves_to_data_node(self) -> None:
        """Test that static_node alias resolves correctly."""
        from hexdag.kernel.resolver import resolve

        StaticNode = resolve("static_node")
        DataNodeCls = resolve("data_node")

        assert StaticNode is DataNodeCls

    def test_static_node_creates_working_node(self) -> None:
        """Test that static_node creates a functional node."""
        from hexdag.kernel.resolver import resolve

        StaticNode = resolve("static_node")
        factory = StaticNode()
        node = factory(
            name="reject_locked",
            output={"action": "REJECTED", "reason": "Load already has winner locked"},
        )

        assert node.name == "reject_locked"
        # Note: DataNode now delegates to ExpressionNode internally,
        # so we test behavior (execution) rather than internal params structure

    @pytest.mark.asyncio
    async def test_static_node_execution(self) -> None:
        """Test that static_node executes and returns static data."""
        from hexdag.kernel.resolver import resolve

        StaticNode = resolve("static_node")
        factory = StaticNode()
        node = factory(
            name="no_action",
            output={"action": "NO_ACTION", "reason": "No action required"},
        )

        result = await node.fn({"ignored": "input"})

        assert result["action"] == "NO_ACTION"
        assert result["reason"] == "No action required"


class TestPortCallNodeParameters:
    """Test PortCallNode with various parameter configurations."""

    def test_creates_node_with_timeout(self) -> None:
        """Test node spec with timeout parameter."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="slow_query",
            timeout=30.0,
        )

        assert spec.timeout == 30.0

    def test_creates_node_with_max_retries(self) -> None:
        """Test node spec with max_retries parameter."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="flaky_operation",
            max_retries=3,
        )

        assert spec.max_retries == 3

    def test_creates_node_with_extra_kwargs(self) -> None:
        """Test that extra kwargs are stored in params."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="query",
            custom_param="custom_value",
            another_param=123,
        )

        assert spec.params.get("custom_param") == "custom_value"
        assert spec.params.get("another_param") == 123

    def test_function_doc_set_correctly(self) -> None:
        """Test that the wrapped function has correct docstring."""
        factory = PortCallNode()
        spec = factory(
            name="my_call",
            port="api",
            method="fetch_data",
        )

        assert spec.fn.__doc__ == "Port call: api.fetch_data"

    def test_input_schema_created_from_mapping(self) -> None:
        """Test that input schema is created from input_mapping keys."""
        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="database",
            method="query",
            input_mapping={"query": "$input.sql", "limit": "$input.max_rows"},
        )

        # Input model should have fields matching input_mapping keys
        assert spec.in_model is not None
        assert "query" in spec.in_model.model_fields
        assert "limit" in spec.in_model.model_fields


class TestPortCallNodeEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_input_data(self) -> None:
        """Test calling method with empty input data."""
        mock_adapter = AsyncMock()
        mock_adapter.get_all = AsyncMock(return_value=["item1", "item2"])

        set_ports({"store": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="store",
            method="get_all",
        )

        result = await spec.fn({})

        assert result["result"] == ["item1", "item2"]
        mock_adapter.get_all.assert_called_once_with()

        set_ports(None)

    @pytest.mark.asyncio
    async def test_method_returns_none(self) -> None:
        """Test method that returns None."""
        mock_adapter = AsyncMock()
        mock_adapter.delete = AsyncMock(return_value=None)

        set_ports({"store": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="store",
            method="delete",
        )

        result = await spec.fn({"key": "item_to_delete"})

        assert result["result"] is None
        assert result["error"] is None

        set_ports(None)

    @pytest.mark.asyncio
    async def test_method_returns_primitive(self) -> None:
        """Test method that returns a primitive value."""
        mock_adapter = AsyncMock()
        mock_adapter.count = AsyncMock(return_value=42)

        set_ports({"store": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="store",
            method="count",
        )

        result = await spec.fn({})

        assert result["result"] == 42

        set_ports(None)

    @pytest.mark.asyncio
    async def test_error_message_lists_available_methods(self) -> None:
        """Test that error message includes available methods when method not found."""

        # Create a class that strictly defines its methods
        class StrictAdapter:
            def method_a(self) -> None:
                pass

            def method_b(self) -> None:
                pass

        mock_adapter = StrictAdapter()

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="test_node",
            port="service",
            method="nonexistent",
        )

        with pytest.raises(AttributeError) as exc_info:
            await spec.fn({})

        error_message = str(exc_info.value)
        assert "method_a" in error_message or "method_b" in error_message

        set_ports(None)


class TestPortCallNodeYAMLIntegration:
    """Test PortCallNode with YAML pipeline builder."""

    @pytest.mark.asyncio
    async def test_port_call_node_from_yaml(self) -> None:
        """Test creating port_call_node from YAML configuration."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        yaml_config = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-port-call
spec:
  nodes:
    - kind: port_call_node
      metadata:
        name: fetch_data
      spec:
        port: api
        method: aget_data
"""
        # Build the pipeline
        builder = YamlPipelineBuilder()
        graph, _ = builder.build_from_yaml_string(yaml_config)

        # Create mock adapter
        mock_api = AsyncMock()
        mock_api.aget_data = AsyncMock(return_value={"data": "fetched"})

        # Execute
        orchestrator = Orchestrator()
        results = await orchestrator.run(
            graph,
            {},
            additional_ports={"api": mock_api},
        )

        assert "fetch_data" in results
        assert results["fetch_data"]["result"] == {"data": "fetched"}

    @pytest.mark.asyncio
    async def test_static_node_from_yaml(self) -> None:
        """Test creating static_node from YAML configuration."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        yaml_config = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-static
spec:
  nodes:
    - kind: static_node
      metadata:
        name: reject_response
      spec:
        output:
          action: REJECTED
          reason: Not eligible
"""
        # Build the pipeline
        builder = YamlPipelineBuilder()
        graph, _ = builder.build_from_yaml_string(yaml_config)

        # Execute
        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})

        assert "reject_response" in results
        assert results["reject_response"]["action"] == "REJECTED"
        assert results["reject_response"]["reason"] == "Not eligible"

    @pytest.mark.asyncio
    async def test_port_call_with_fallback_from_yaml(self) -> None:
        """Test port_call_node with fallback from YAML configuration."""
        from hexdag.compiler.yaml_builder import YamlPipelineBuilder
        from hexdag.kernel.orchestration.orchestrator import Orchestrator

        yaml_config = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-fallback
spec:
  nodes:
    - kind: port_call_node
      metadata:
        name: cache_lookup
      spec:
        port: cache
        method: aget
        fallback: null
        has_fallback: true
"""
        # Build the pipeline
        builder = YamlPipelineBuilder()
        graph, _ = builder.build_from_yaml_string(yaml_config)

        # Execute without the cache port configured (should use fallback)
        orchestrator = Orchestrator()
        results = await orchestrator.run(graph, {})

        assert "cache_lookup" in results
        assert results["cache_lookup"]["result"] is None
        assert results["cache_lookup"]["error"] is not None


class TestPortCallNodeLogging:
    """Test PortCallNode logging behavior.

    Note: hexDAG uses loguru for logging. We use a custom sink to capture logs.
    """

    @pytest.fixture
    def log_capture(self):
        """Fixture to capture loguru logs."""
        from loguru import logger

        captured_logs: list[dict] = []

        def sink(message):
            record = message.record
            captured_logs.append({
                "level": record["level"].name,
                "message": record["message"],
                "extra": dict(record["extra"]),
            })

        handler_id = logger.add(sink, level="DEBUG", format="{message}")
        yield captured_logs
        logger.remove(handler_id)

    @pytest.mark.asyncio
    async def test_logs_info_on_method_call(self, log_capture: list[dict]) -> None:
        """Test that INFO log is emitted when calling port method."""
        mock_adapter = AsyncMock()
        mock_adapter.do_work = AsyncMock(return_value="done")

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="log_test",
            port="service",
            method="do_work",
        )

        await spec.fn({"param1": "value1", "param2": "value2"})

        # Verify INFO log was emitted
        info_logs = [log for log in log_capture if log["level"] == "INFO"]
        assert len(info_logs) >= 1
        assert any("Calling port method" in log["message"] for log in info_logs)

        set_ports(None)

    @pytest.mark.asyncio
    async def test_logs_debug_on_completion(self, log_capture: list[dict]) -> None:
        """Test that DEBUG log is emitted on successful completion."""
        mock_adapter = AsyncMock()
        mock_adapter.do_work = AsyncMock(return_value={"status": "ok"})

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="log_test",
            port="service",
            method="do_work",
        )

        await spec.fn({})

        # Verify DEBUG log was emitted with completion info
        debug_logs = [log for log in log_capture if log["level"] == "DEBUG"]
        assert len(debug_logs) >= 1
        assert any("Port method completed" in log["message"] for log in debug_logs)

        set_ports(None)

    @pytest.mark.asyncio
    async def test_logs_error_on_failure(self, log_capture: list[dict]) -> None:
        """Test that ERROR log is emitted when method fails."""
        mock_adapter = AsyncMock()
        mock_adapter.failing_method = AsyncMock(side_effect=ValueError("Test error"))

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="log_test",
            port="service",
            method="failing_method",
        )

        with pytest.raises(ValueError):
            await spec.fn({})

        # Verify ERROR log was emitted
        error_logs = [log for log in log_capture if log["level"] == "ERROR"]
        assert len(error_logs) >= 1
        assert any("Port method failed" in log["message"] for log in error_logs)

        set_ports(None)

    @pytest.mark.asyncio
    async def test_logs_warning_on_fallback_used(self, log_capture: list[dict]) -> None:
        """Test that WARNING log is emitted when fallback is used."""
        set_ports({})  # No port configured

        factory = PortCallNode()
        spec = factory(
            name="log_test",
            port="missing_port",
            method="some_method",
            fallback="default",
            has_fallback=True,
        )

        await spec.fn({})

        # Verify WARNING log was emitted
        warning_logs = [log for log in log_capture if log["level"] == "WARNING"]
        assert len(warning_logs) >= 1
        assert any(
            "not available" in log["message"] and "fallback" in log["message"]
            for log in warning_logs
        )

        set_ports(None)

    @pytest.mark.asyncio
    async def test_log_includes_node_context(self, log_capture: list[dict]) -> None:
        """Test that log records include node context (node name, port, method)."""
        mock_adapter = AsyncMock()
        mock_adapter.test_method = AsyncMock(return_value="result")

        set_ports({"test_port": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="context_test_node",
            port="test_port",
            method="test_method",
        )

        await spec.fn({})

        # Check that log records have node context in extra
        assert len(log_capture) > 0
        # The logger is bound with node, node_type, port, and method
        info_log = next((log for log in log_capture if log["level"] == "INFO"), None)
        assert info_log is not None
        assert info_log["extra"].get("node") == "context_test_node"
        assert info_log["extra"].get("port") == "test_port"
        assert info_log["extra"].get("method") == "test_method"

        set_ports(None)

    @pytest.mark.asyncio
    async def test_log_includes_node_type(self, log_capture: list[dict]) -> None:
        """Test that log records include node_type context."""
        mock_adapter = AsyncMock()
        mock_adapter.some_method = AsyncMock(return_value="done")

        set_ports({"svc": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="type_test",
            port="svc",
            method="some_method",
        )

        await spec.fn({})

        # Check node_type is set correctly
        info_log = next((log for log in log_capture if log["level"] == "INFO"), None)
        assert info_log is not None
        assert info_log["extra"].get("node_type") == "port_call_node"

        set_ports(None)

    @pytest.mark.asyncio
    async def test_log_includes_args_list(self, log_capture: list[dict]) -> None:
        """Test that the INFO log includes the list of argument names."""
        mock_adapter = AsyncMock()
        mock_adapter.multi_arg_method = AsyncMock(return_value="ok")

        set_ports({"service": mock_adapter})

        factory = PortCallNode()
        spec = factory(
            name="args_test",
            port="service",
            method="multi_arg_method",
        )

        await spec.fn({"arg_a": 1, "arg_b": 2, "arg_c": 3})

        # Verify the log contains args info
        info_log = next((log for log in log_capture if log["level"] == "INFO"), None)
        assert info_log is not None
        # The args are logged in extra
        assert "args" in info_log["extra"]
        assert set(info_log["extra"]["args"]) == {"arg_a", "arg_b", "arg_c"}

        set_ports(None)


class TestPortCallNodeMultiplePortTypes:
    """Test PortCallNode with various port types."""

    @pytest.mark.asyncio
    async def test_with_llm_port(self) -> None:
        """Test port_call_node with LLM port."""
        mock_llm = AsyncMock()
        mock_llm.aresponse = AsyncMock(return_value="LLM response text")

        set_ports({"llm": mock_llm})

        factory = PortCallNode()
        spec = factory(
            name="llm_call",
            port="llm",
            method="aresponse",
        )

        result = await spec.fn({"messages": [{"role": "user", "content": "Hello"}]})

        assert result["result"] == "LLM response text"
        assert result["port"] == "llm"

        set_ports(None)

    @pytest.mark.asyncio
    async def test_with_memory_port(self) -> None:
        """Test port_call_node with memory port."""
        mock_memory = AsyncMock()
        mock_memory.aget = AsyncMock(return_value="stored_value")

        set_ports({"memory": mock_memory})

        factory = PortCallNode()
        spec = factory(
            name="memory_get",
            port="memory",
            method="aget",
        )

        result = await spec.fn({"key": "my_key"})

        assert result["result"] == "stored_value"
        mock_memory.aget.assert_called_once_with(key="my_key")

        set_ports(None)

    @pytest.mark.asyncio
    async def test_with_tool_router_port(self) -> None:
        """Test port_call_node with tool_router port."""
        mock_router = AsyncMock()
        mock_router.acall_tool = AsyncMock(return_value={"tool_result": "success"})

        set_ports({"tool_router": mock_router})

        factory = PortCallNode()
        spec = factory(
            name="tool_call",
            port="tool_router",
            method="acall_tool",
        )

        result = await spec.fn({"tool_name": "search", "params": {"query": "test"}})

        assert result["result"] == {"tool_result": "success"}

        set_ports(None)
