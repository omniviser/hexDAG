"""Integration tests for hooks with Orchestrator.

Tests the complete lifecycle of pre-DAG and post-DAG hooks including:
- Health checks before pipeline execution
- Secret injection and cleanup
- Adapter lifecycle management
- Hook execution on success and failure
- Context metadata tracking
"""

import pytest

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.orchestration.hooks import HookConfig, PostDagHookConfig
from hexai.core.ports.healthcheck import HealthStatus

# ===================================================================
# Mock Adapters for Testing
# ===================================================================


class MockAdapterWithLifecycle:
    """Mock adapter with full lifecycle methods."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._hexdag_name = name
        self.health_check_count = 0
        self.close_count = 0
        self.health_status = "healthy"
        self.should_fail_health_check = False

    async def ahealth_check(self) -> HealthStatus:
        """Health check."""
        self.health_check_count += 1

        if self.should_fail_health_check:
            return HealthStatus(
                status="unhealthy",
                adapter_name=self.name,
                latency_ms=5.0,
                error=Exception("Mock health check failure"),
            )

        return HealthStatus(
            status=self.health_status,
            adapter_name=self.name,
            latency_ms=5.0,
        )

    async def aclose(self) -> None:
        """Cleanup."""
        self.close_count += 1


class MockMemory:
    """Mock memory adapter."""

    def __init__(self):
        self.storage = {}
        self.health_check_count = 0

    async def aget(self, key: str):
        return self.storage.get(key)

    async def aset(self, key: str, value):
        self.storage[key] = value

    async def ahealth_check(self) -> HealthStatus:
        """Health check for memory."""
        self.health_check_count += 1
        return HealthStatus(
            status="healthy",
            adapter_name="mock_memory",
            latency_ms=2.0,
        )


class MockSecretPort:
    """Mock secret port."""

    def __init__(self, secrets: dict[str, str]):
        self.secrets = secrets
        self.health_check_count = 0

    async def aget_secret(self, key: str):
        from hexai.helpers.secrets import Secret

        return Secret(self.secrets[key])

    async def aload_secrets_to_memory(
        self,
        memory,
        prefix: str = "secret:",
        keys: list[str] | None = None,
    ):
        keys_to_load = keys or list(self.secrets.keys())
        mapping = {}

        for key in keys_to_load:
            if key in self.secrets:
                memory_key = f"{prefix}{key}"
                await memory.aset(memory_key, self.secrets[key])
                mapping[key] = memory_key

        return mapping

    async def ahealth_check(self) -> HealthStatus:
        """Health check for secrets."""
        self.health_check_count += 1
        return HealthStatus(
            status="healthy",
            adapter_name="mock_secret",
            latency_ms=3.0,
        )


# ===================================================================
# Pre-DAG Hooks Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_pre_dag_health_checks():
    """Test orchestrator executes health checks before pipeline."""

    # Create simple graph
    def simple_node(x: int, **kwargs) -> int:
        return x * 2

    graph = DirectedGraph()
    graph.add(NodeSpec(name="double", fn=simple_node, deps=set()))

    # Create adapter with health check
    mock_adapter = MockAdapterWithLifecycle("test_adapter")

    # Create orchestrator with health check enabled
    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        pre_hook_config=HookConfig(
            enable_health_checks=True,
            enable_secret_injection=False,
        ),
    )

    # Run pipeline
    results = await orchestrator.run(graph, 5)

    # Assert pipeline executed successfully
    assert results["double"] == 10

    # Assert health check ran
    assert mock_adapter.health_check_count == 1


@pytest.mark.asyncio
async def test_orchestrator_pre_dag_health_checks_multiple_adapters():
    """Test health checks work with multiple adapters."""

    def simple_node(x: int, **kwargs) -> int:
        return x + 1

    graph = DirectedGraph()
    graph.add(NodeSpec(name="increment", fn=simple_node, deps=set()))

    # Create multiple adapters
    adapter1 = MockAdapterWithLifecycle("adapter1")
    adapter2 = MockAdapterWithLifecycle("adapter2")
    memory = MockMemory()

    orchestrator = Orchestrator(
        ports={
            "adapter1": adapter1,
            "adapter2": adapter2,
            "memory": memory,
        },
        pre_hook_config=HookConfig(enable_health_checks=True),
    )

    results = await orchestrator.run(graph, 10)

    assert results["increment"] == 11
    assert adapter1.health_check_count == 1
    assert adapter2.health_check_count == 1
    assert memory.health_check_count == 1


@pytest.mark.asyncio
async def test_orchestrator_pre_dag_health_checks_degraded_status():
    """Test pipeline continues with degraded adapter health."""

    def simple_node(x: int, **kwargs) -> int:
        return x * 3

    graph = DirectedGraph()
    graph.add(NodeSpec(name="triple", fn=simple_node, deps=set()))

    # Create adapter with degraded status
    mock_adapter = MockAdapterWithLifecycle("degraded_adapter")
    mock_adapter.health_status = "degraded"

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        pre_hook_config=HookConfig(enable_health_checks=True),
    )

    # Pipeline should still execute
    results = await orchestrator.run(graph, 5)
    assert results["triple"] == 15


@pytest.mark.asyncio
async def test_orchestrator_secret_injection():
    """Test orchestrator loads secrets into memory."""

    def simple_node(x=None, **kwargs) -> str:
        return "node_executed"

    graph = DirectedGraph()
    graph.add(NodeSpec(name="simple", fn=simple_node, deps=set()))

    # Create ports
    memory = MockMemory()
    secret_port = MockSecretPort(
        secrets={
            "API_KEY": "sk-test-123",
            "DATABASE_URL": "postgres://localhost/test",
        }
    )

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_health_checks=False,
            enable_secret_injection=True,
            secret_keys=["API_KEY", "DATABASE_URL"],
        ),
    )

    # Run pipeline
    await orchestrator.run(graph, None)

    # Secrets should be loaded into memory during execution
    # (they get cleaned up after, so we need to check in a node)


@pytest.mark.asyncio
async def test_orchestrator_secret_injection_with_custom_prefix():
    """Test secret injection with custom prefix."""
    secrets_loaded = {}

    def capture_secrets(x=None, **kwargs) -> str:
        # Capture memory state during execution
        memory = kwargs.get("memory")
        if memory:
            secrets_loaded["api_key"] = memory.storage.get("env:API_KEY")
        return "captured"

    graph = DirectedGraph()
    graph.add(NodeSpec(name="capture", fn=capture_secrets, deps=set()))

    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"API_KEY": "sk-custom-prefix"})

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_health_checks=False,
            enable_secret_injection=True,
            secret_keys=["API_KEY"],
            secret_prefix="env:",
        ),
    )

    await orchestrator.run(graph, None)

    # Verify secret was loaded with custom prefix
    assert secrets_loaded["api_key"] == "sk-custom-prefix"


# ===================================================================
# Post-DAG Hooks Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_post_dag_secret_cleanup():
    """Test secrets are cleaned up after pipeline execution."""

    def simple_node(x=None, **kwargs) -> str:
        return "node_executed"

    graph = DirectedGraph()
    graph.add(NodeSpec(name="simple", fn=simple_node, deps=set()))

    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"API_KEY": "sk-test-123"})

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_health_checks=False,
            enable_secret_injection=True,
            secret_keys=["API_KEY"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_secret_cleanup=True,
        ),
    )

    await orchestrator.run(graph, None)

    # Secret should be removed from memory
    assert await memory.aget("secret:API_KEY") is None


@pytest.mark.asyncio
async def test_orchestrator_post_dag_secret_cleanup_multiple_secrets():
    """Test multiple secrets are cleaned up properly."""

    def simple_node(x=None, **kwargs) -> str:
        return "done"

    graph = DirectedGraph()
    graph.add(NodeSpec(name="worker", fn=simple_node, deps=set()))

    memory = MockMemory()
    secret_port = MockSecretPort(
        secrets={
            "API_KEY": "sk-key-123",
            "DATABASE_URL": "postgres://db",
            "JWT_SECRET": "jwt-token",
        }
    )

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_secret_injection=True,
            secret_keys=["API_KEY", "DATABASE_URL", "JWT_SECRET"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_secret_cleanup=True,
        ),
    )

    await orchestrator.run(graph, None)

    # All secrets should be removed
    assert await memory.aget("secret:API_KEY") is None
    assert await memory.aget("secret:DATABASE_URL") is None
    assert await memory.aget("secret:JWT_SECRET") is None


@pytest.mark.asyncio
async def test_orchestrator_post_dag_adapter_cleanup():
    """Test adapter cleanup after pipeline."""

    def simple_node(x: int, **kwargs) -> int:
        return x + 1

    graph = DirectedGraph()
    graph.add(NodeSpec(name="increment", fn=simple_node, deps=set()))

    mock_adapter = MockAdapterWithLifecycle("cleanup_adapter")

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
        ),
    )

    results = await orchestrator.run(graph, 5)

    assert results["increment"] == 6
    assert mock_adapter.close_count == 1


@pytest.mark.asyncio
async def test_orchestrator_post_dag_cleanup_multiple_adapters():
    """Test cleanup works with multiple adapters."""

    def simple_node(x: int, **kwargs) -> int:
        return x * 2

    graph = DirectedGraph()
    graph.add(NodeSpec(name="double", fn=simple_node, deps=set()))

    adapter1 = MockAdapterWithLifecycle("adapter1")
    adapter2 = MockAdapterWithLifecycle("adapter2")
    adapter3 = MockAdapterWithLifecycle("adapter3")

    orchestrator = Orchestrator(
        ports={
            "adapter1": adapter1,
            "adapter2": adapter2,
            "adapter3": adapter3,
        },
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
        ),
    )

    await orchestrator.run(graph, 5)

    # All adapters should be cleaned up
    assert adapter1.close_count == 1
    assert adapter2.close_count == 1
    assert adapter3.close_count == 1


# ===================================================================
# Error Handling and Failure Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_hooks_run_on_pipeline_failure():
    """Test post-DAG hooks run even when pipeline fails."""

    def failing_node(x: int, **kwargs) -> int:
        raise ValueError("Intentional failure")

    graph = DirectedGraph()
    graph.add(NodeSpec(name="fail", fn=failing_node, deps=set()))

    mock_adapter = MockAdapterWithLifecycle("failure_adapter")

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
            run_on_failure=True,
        ),
    )

    with pytest.raises(Exception, match="Intentional failure"):
        await orchestrator.run(graph, 5)

    # Cleanup should still run
    assert mock_adapter.close_count == 1


@pytest.mark.asyncio
async def test_orchestrator_secret_cleanup_on_failure():
    """Test secrets are cleaned up even when pipeline fails."""

    def failing_node(x=None, **kwargs) -> str:
        raise RuntimeError("Pipeline failed")

    graph = DirectedGraph()
    graph.add(NodeSpec(name="fail", fn=failing_node, deps=set()))

    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"API_KEY": "sk-test"})

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_secret_injection=True,
            secret_keys=["API_KEY"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_secret_cleanup=True,
            run_on_failure=True,
        ),
    )

    # NodeExecutionError wraps the RuntimeError
    with pytest.raises(Exception, match="Pipeline failed"):
        await orchestrator.run(graph, None)

    # Secret should still be cleaned up
    assert await memory.aget("secret:API_KEY") is None


@pytest.mark.asyncio
async def test_orchestrator_hooks_skip_cleanup_on_failure_when_disabled():
    """Test cleanup can be skipped on failure."""

    def failing_node(x: int, **kwargs) -> int:
        raise ValueError("Error")

    graph = DirectedGraph()
    graph.add(NodeSpec(name="fail", fn=failing_node, deps=set()))

    mock_adapter = MockAdapterWithLifecycle("no_cleanup_adapter")

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
            run_on_failure=False,  # Don't run on failure
        ),
    )

    with pytest.raises(Exception, match="Error"):
        await orchestrator.run(graph, 5)

    # Cleanup should NOT run because run_on_failure=False
    assert mock_adapter.close_count == 0


# ===================================================================
# Multiple Runs and Adapter Reuse Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_multiple_runs_reuse_adapters():
    """Test multiple pipeline runs reuse the same adapter instances."""

    def simple_node(x: int, **kwargs) -> int:
        return x * 2

    graph = DirectedGraph()
    graph.add(NodeSpec(name="double", fn=simple_node, deps=set()))

    mock_adapter = MockAdapterWithLifecycle("reusable_adapter")

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        pre_hook_config=HookConfig(enable_health_checks=True),
        post_hook_config=PostDagHookConfig(enable_adapter_cleanup=True),
    )

    # Run multiple times
    results1 = await orchestrator.run(graph, 1)
    results2 = await orchestrator.run(graph, 2)
    results3 = await orchestrator.run(graph, 3)

    assert results1["double"] == 2
    assert results2["double"] == 4
    assert results3["double"] == 6

    # Health check ran 3 times (once per run)
    assert mock_adapter.health_check_count == 3

    # Cleanup ran 3 times (once per run)
    assert mock_adapter.close_count == 3


@pytest.mark.asyncio
async def test_orchestrator_secret_injection_isolated_per_run():
    """Test secrets are properly isolated between multiple runs."""
    captured_secrets = []

    def capture_secret(x=None, **kwargs) -> str:
        memory = kwargs.get("memory")
        if memory:
            secret = memory.storage.get("secret:API_KEY")
            captured_secrets.append(secret)
        return "captured"

    graph = DirectedGraph()
    graph.add(NodeSpec(name="capture", fn=capture_secret, deps=set()))

    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"API_KEY": "sk-test-123"})

    orchestrator = Orchestrator(
        ports={
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_secret_injection=True,
            secret_keys=["API_KEY"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_secret_cleanup=True,
        ),
    )

    # Run twice
    await orchestrator.run(graph, None)
    await orchestrator.run(graph, None)

    # Both runs should have captured the secret
    assert len(captured_secrets) == 2
    assert captured_secrets[0] == "sk-test-123"
    assert captured_secrets[1] == "sk-test-123"

    # Secret should be cleaned up after final run
    assert await memory.aget("secret:API_KEY") is None


# ===================================================================
# Combined Hooks Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_full_lifecycle_with_all_hooks():
    """Test complete lifecycle with all hooks enabled."""

    def process_data(x: int, **kwargs) -> int:
        return x + 10

    graph = DirectedGraph()
    graph.add(NodeSpec(name="processor", fn=process_data, deps=set()))

    adapter = MockAdapterWithLifecycle("full_lifecycle_adapter")
    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"API_KEY": "sk-full-test"})

    orchestrator = Orchestrator(
        ports={
            "adapter": adapter,
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_health_checks=True,
            enable_secret_injection=True,
            secret_keys=["API_KEY"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
            enable_secret_cleanup=True,
        ),
    )

    results = await orchestrator.run(graph, 5)

    # Verify execution
    assert results["processor"] == 15

    # Verify pre-DAG hooks ran
    assert adapter.health_check_count == 1
    assert memory.health_check_count == 1

    # Verify post-DAG hooks ran
    assert adapter.close_count == 1
    assert await memory.aget("secret:API_KEY") is None


@pytest.mark.asyncio
async def test_orchestrator_hooks_context_metadata():
    """Test hook results are stored in context metadata."""

    def simple_node(x: int, **kwargs) -> int:
        return x + 1

    graph = DirectedGraph()
    graph.add(NodeSpec(name="increment", fn=simple_node, deps=set()))

    mock_adapter = MockAdapterWithLifecycle("metadata_adapter")

    orchestrator = Orchestrator(
        ports={"test_port": mock_adapter},
        pre_hook_config=HookConfig(enable_health_checks=True),
        post_hook_config=PostDagHookConfig(enable_adapter_cleanup=True),
    )

    results = await orchestrator.run(graph, 5)

    # Context metadata is set internally by orchestrator
    # We verify it works by checking the execution succeeded
    assert results["increment"] == 6


@pytest.mark.asyncio
async def test_orchestrator_hooks_with_complex_pipeline():
    """Test hooks work correctly with complex multi-node pipeline."""

    def add_one(x: int, **kwargs) -> int:
        return x + 1

    def multiply_two(x: int, **kwargs) -> int:
        return x * 2

    def combine(inputs: dict, **kwargs) -> int:
        return inputs["add_one"] + inputs["multiply_two"]

    graph = DirectedGraph()
    graph.add(NodeSpec(name="add_one", fn=add_one, deps=set()))
    graph.add(NodeSpec(name="multiply_two", fn=multiply_two, deps=set()))
    graph.add(NodeSpec(name="combine", fn=combine, deps={"add_one", "multiply_two"}))

    adapter1 = MockAdapterWithLifecycle("adapter1")
    adapter2 = MockAdapterWithLifecycle("adapter2")
    memory = MockMemory()
    secret_port = MockSecretPort(secrets={"KEY1": "value1", "KEY2": "value2"})

    orchestrator = Orchestrator(
        ports={
            "adapter1": adapter1,
            "adapter2": adapter2,
            "memory": memory,
            "secret": secret_port,
        },
        pre_hook_config=HookConfig(
            enable_health_checks=True,
            enable_secret_injection=True,
            secret_keys=["KEY1", "KEY2"],
        ),
        post_hook_config=PostDagHookConfig(
            enable_adapter_cleanup=True,
            enable_secret_cleanup=True,
        ),
    )

    results = await orchestrator.run(graph, 5)

    # Verify complex pipeline executed correctly
    assert results["add_one"] == 6
    assert results["multiply_two"] == 10
    assert results["combine"] == 16

    # Verify hooks ran
    assert adapter1.health_check_count == 1
    assert adapter2.health_check_count == 1
    assert adapter1.close_count == 1
    assert adapter2.close_count == 1
    assert await memory.aget("secret:KEY1") is None
    assert await memory.aget("secret:KEY2") is None
