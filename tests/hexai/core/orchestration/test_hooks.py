"""Tests for pre-DAG and post-DAG hook system."""

import pytest

from hexai.core.context import ExecutionContext
from hexai.core.orchestration.hooks import (
    HookConfig,
    PostDagHookConfig,
    PostDagHookManager,
    PreDagHookManager,
)
from hexai.core.orchestration.models import NodeExecutionContext
from hexai.core.ports.healthcheck import HealthStatus


class MockAdapterWithHealth:
    """Mock adapter that implements health check."""

    def __init__(self, health_status: str = "healthy", should_fail: bool = False):
        self.health_status = health_status
        self.should_fail = should_fail
        self._hexdag_name = "mock_adapter"
        self.health_check_called = False
        self.close_called = False

    async def ahealth_check(self) -> HealthStatus:
        """Mock health check."""
        self.health_check_called = True

        if self.should_fail:
            raise Exception("Health check failed!")

        return HealthStatus(
            status=self.health_status,
            adapter_name=self._hexdag_name,
            latency_ms=10.5,
        )

    async def aclose(self) -> None:
        """Mock cleanup."""
        self.close_called = True


class MockSecretPort:
    """Mock secret port for testing."""

    def __init__(self, secrets: dict[str, str] | None = None):
        self.secrets = secrets or {}
        self.load_called = False

    async def aget_secret(self, key: str):
        """Get single secret."""
        from hexai.core.types import Secret

        if key not in self.secrets:
            raise KeyError(f"Secret '{key}' not found")
        return Secret(self.secrets[key])

    async def aload_secrets_to_memory(
        self,
        memory,
        prefix: str = "secret:",
        keys: list[str] | None = None,
    ) -> dict[str, str]:
        """Load secrets into memory."""
        self.load_called = True

        keys_to_load = keys or list(self.secrets.keys())
        mapping = {}

        for key in keys_to_load:
            if key in self.secrets:
                memory_key = f"{prefix}{key}"
                await memory.aset(memory_key, self.secrets[key])
                mapping[key] = memory_key

        return mapping


class MockMemory:
    """Mock memory port for testing."""

    def __init__(self):
        self.storage = {}

    async def aget(self, key: str, dag_id: str | None = None):
        """Get value from memory."""
        return self.storage.get(key)

    async def aset(self, key: str, value, dag_id: str | None = None):
        """Set value in memory."""
        self.storage[key] = value


@pytest.fixture
def mock_context():
    """Create mock execution context."""
    return NodeExecutionContext(dag_id="test_pipeline")


class TestPreDagHookManager:
    """Test PreDagHookManager functionality."""

    @pytest.mark.asyncio
    async def test_health_checks_healthy_adapters(self, mock_context):
        """Test health checks with healthy adapters."""
        # Setup
        adapter1 = MockAdapterWithHealth(health_status="healthy")
        adapter2 = MockAdapterWithHealth(health_status="healthy")

        ports = {
            "llm": adapter1,
            "database": adapter2,
        }

        config = HookConfig(
            enable_health_checks=True,
            enable_secret_injection=False,
        )

        manager = PreDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            results = await manager.execute_hooks(
                context=mock_context, pipeline_name="test_pipeline"
            )

        # Assert
        assert "health_checks" in results
        assert len(results["health_checks"]) == 2
        assert all(h.status == "healthy" for h in results["health_checks"])
        assert adapter1.health_check_called
        assert adapter2.health_check_called

    @pytest.mark.asyncio
    async def test_health_checks_unhealthy_with_warning(self, mock_context):
        """Test health checks with unhealthy adapter (warn only)."""
        # Setup
        healthy_adapter = MockAdapterWithHealth(health_status="healthy")
        unhealthy_adapter = MockAdapterWithHealth(health_status="unhealthy")

        ports = {
            "llm": healthy_adapter,
            "database": unhealthy_adapter,
        }

        config = HookConfig(
            enable_health_checks=True,
            health_check_fail_fast=False,
            health_check_warn_only=True,
        )

        manager = PreDagHookManager(config)

        # Execute - should not raise
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            results = await manager.execute_hooks(
                context=mock_context, pipeline_name="test_pipeline"
            )

        # Assert
        assert "health_checks" in results
        health_statuses = {h.status for h in results["health_checks"]}
        assert "unhealthy" in health_statuses

    @pytest.mark.asyncio
    async def test_health_checks_fail_fast(self, mock_context):
        """Test health checks with fail-fast enabled."""
        from hexai.core.orchestration.components import OrchestratorError

        # Setup
        unhealthy_adapter = MockAdapterWithHealth(health_status="unhealthy")

        ports = {"llm": unhealthy_adapter}

        config = HookConfig(
            enable_health_checks=True,
            health_check_fail_fast=True,
        )

        manager = PreDagHookManager(config)

        # Execute and assert raises
        with pytest.raises(OrchestratorError, match="Health check failed"):
            async with ExecutionContext(
                observer_manager=None,
                policy_manager=None,
                run_id="test-run",
                ports=ports,
            ):
                await manager.execute_hooks(context=mock_context, pipeline_name="test_pipeline")

    @pytest.mark.asyncio
    async def test_secret_injection(self, mock_context):
        """Test secret injection from SecretPort to Memory."""
        # Setup
        secret_port = MockSecretPort(
            secrets={
                "OPENAI_API_KEY": "sk-test-123",
                "DATABASE_PASSWORD": "dbpass456",
            }
        )

        memory = MockMemory()

        ports = {
            "secret": secret_port,
            "memory": memory,
        }

        config = HookConfig(
            enable_health_checks=False,
            enable_secret_injection=True,
            secret_keys=["OPENAI_API_KEY", "DATABASE_PASSWORD"],
            secret_prefix="secret:",
        )

        manager = PreDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            results = await manager.execute_hooks(
                context=mock_context, pipeline_name="test_pipeline"
            )

        # Assert
        assert "secrets_loaded" in results
        assert results["secrets_loaded"]["OPENAI_API_KEY"] == "secret:OPENAI_API_KEY"
        assert results["secrets_loaded"]["DATABASE_PASSWORD"] == "secret:DATABASE_PASSWORD"

        # Verify secrets in memory
        assert await memory.aget("secret:OPENAI_API_KEY") == "sk-test-123"
        assert await memory.aget("secret:DATABASE_PASSWORD") == "dbpass456"
        assert secret_port.load_called

    @pytest.mark.asyncio
    async def test_custom_hooks(self, mock_context):
        """Test custom user-defined hooks."""
        # Setup
        hook_called = False

        async def custom_hook(ports, context):
            nonlocal hook_called
            hook_called = True
            return {"custom_data": "test"}

        config = HookConfig(
            enable_health_checks=False,
            enable_secret_injection=False,
            custom_hooks=[custom_hook],
        )

        manager = PreDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={},
        ):
            results = await manager.execute_hooks(
                context=mock_context, pipeline_name="test_pipeline"
            )

        # Assert
        assert hook_called
        assert "custom_hook" in results
        assert results["custom_hook"]["custom_data"] == "test"

    @pytest.mark.asyncio
    async def test_skips_manager_ports(self, mock_context):
        """Test that manager ports are skipped during health checks."""
        # Setup
        adapter = MockAdapterWithHealth(health_status="healthy")

        # Include manager ports that should be skipped
        ports = {
            "llm": adapter,
            "observer_manager": "should_be_skipped",
            "policy_manager": "should_be_skipped",
        }

        config = HookConfig(enable_health_checks=True)
        manager = PreDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            results = await manager.execute_hooks(
                context=mock_context, pipeline_name="test_pipeline"
            )

        # Assert - only one health check (for llm adapter)
        assert len(results["health_checks"]) == 1


class TestPostDagHookManager:
    """Test PostDagHookManager functionality."""

    @pytest.mark.asyncio
    async def test_adapter_cleanup(self, mock_context):
        """Test adapter cleanup calls aclose()."""
        # Setup
        adapter1 = MockAdapterWithHealth()
        adapter2 = MockAdapterWithHealth()

        ports = {
            "llm": adapter1,
            "database": adapter2,
        }

        config = PostDagHookConfig(
            enable_adapter_cleanup=True,
            enable_secret_cleanup=False,
        )

        manager = PostDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports=ports,
        ):
            results = await manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="success",
                node_results={},
            )

        # Assert
        assert "adapter_cleanup" in results
        assert results["adapter_cleanup"]["count"] == 2
        assert adapter1.close_called
        assert adapter2.close_called

    @pytest.mark.asyncio
    async def test_runs_on_failure(self, mock_context):
        """Test hooks run when pipeline fails."""
        adapter = MockAdapterWithHealth()

        config = PostDagHookConfig(
            run_on_failure=True,
            enable_adapter_cleanup=True,
        )

        manager = PostDagHookManager(config)

        # Execute with failed status
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={"llm": adapter},
        ):
            results = await manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="failed",
                node_results={},
                error=Exception("Test error"),
            )

        # Assert cleanup still ran
        assert adapter.close_called
        assert results["adapter_cleanup"]["count"] == 1

    @pytest.mark.asyncio
    async def test_skips_on_success_when_configured(self, mock_context):
        """Test hooks skip when configured not to run on success."""
        adapter = MockAdapterWithHealth()

        config = PostDagHookConfig(
            run_on_success=False,  # Don't run on success
            run_on_failure=True,
        )

        manager = PostDagHookManager(config)

        # Execute with success status
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={"llm": adapter},
        ):
            results = await manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="success",
                node_results={},
            )

        # Assert hooks were skipped
        assert results.get("skipped") is True
        assert not adapter.close_called

    @pytest.mark.asyncio
    async def test_custom_post_hooks(self, mock_context):
        """Test custom post-DAG hooks."""
        hook_called = False
        received_status = None

        async def custom_cleanup(ports, context, node_results, pipeline_status, error):
            nonlocal hook_called, received_status
            hook_called = True
            received_status = pipeline_status
            return {"cleaned": True}

        config = PostDagHookConfig(
            custom_hooks=[custom_cleanup],
            enable_adapter_cleanup=False,
        )

        manager = PostDagHookManager(config)

        # Execute
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={},
        ):
            results = await manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="success",
                node_results={"node1": "result1"},
            )

        # Assert
        assert hook_called
        assert received_status == "success"
        assert "custom_cleanup" in results
        assert results["custom_cleanup"]["cleaned"] is True


class TestPostHookCleanupRobustness:
    """Test that critical cleanup happens even when other hooks fail."""

    @pytest.mark.asyncio
    async def test_secret_cleanup_runs_even_when_custom_hook_fails(self):
        """Test that secret cleanup runs even if custom hook raises unexpected exception."""
        # Setup
        mock_memory = MockMemory()
        mock_secret = MockSecretPort({"API_KEY": "secret123"})

        # Pre-hook manager to inject secrets
        pre_config = HookConfig(enable_secret_injection=True)
        pre_manager = PreDagHookManager(pre_config)

        # Create a custom hook that fails with unexpected exception
        def failing_hook(ports, context, node_results, status, error):
            raise AttributeError("Unexpected attribute error in custom hook!")

        # Post-hook config with failing custom hook
        post_config = PostDagHookConfig(
            custom_hooks=[failing_hook],
            enable_secret_cleanup=True,
            enable_adapter_cleanup=False,
        )
        post_manager = PostDagHookManager(post_config, pre_manager)

        mock_context = NodeExecutionContext(dag_id="test-dag")

        # Execute: First inject secrets, then try cleanup with failing hook
        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={"memory": mock_memory, "secret": mock_secret},
        ):
            # Inject secrets
            await pre_manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
            )

            # Verify secret was injected
            secret_key = "secret:API_KEY"
            stored_secret = await mock_memory.aget(secret_key, dag_id="test-dag")
            assert stored_secret == "secret123"

            # Run post-hooks with failing custom hook
            results = await post_manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="success",
                node_results={},
            )

            # Assert: Custom hook failed but secret cleanup still ran
            assert "failing_hook" in results
            assert "error" in results["failing_hook"]
            assert "Unexpected attribute error" in results["failing_hook"]["error"]

            # Critical: Secret cleanup should have run despite custom hook failure
            assert "secret_cleanup" in results
            assert results["secret_cleanup"]["keys_removed"] == 1

            # Verify secret was actually removed
            cleaned_secret = await mock_memory.aget(secret_key, dag_id="test-dag")
            assert cleaned_secret is None

    @pytest.mark.asyncio
    async def test_adapter_cleanup_runs_even_when_checkpoint_fails(self):
        """Test that adapter cleanup runs even if checkpoint save fails."""
        # Setup
        mock_adapter = MockAdapterWithHealth()
        MockMemory()

        # Post-hook config with checkpoint enabled but it will fail
        post_config = PostDagHookConfig(
            enable_checkpoint_save=True,
            enable_adapter_cleanup=True,
            enable_secret_cleanup=False,
        )
        post_manager = PostDagHookManager(post_config, None)

        mock_context = NodeExecutionContext(dag_id="test-dag")

        # Make memory fail during checkpoint save by not having required methods
        class BrokenMemory:
            async def aget(self, key, dag_id=None):
                raise RuntimeError("Memory is broken!")

            async def aset(self, key, value, dag_id=None):
                raise RuntimeError("Memory is broken!")

        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={"memory": BrokenMemory(), "test_adapter": mock_adapter},
        ):
            # Run post-hooks
            results = await post_manager.execute_hooks(
                context=mock_context,
                pipeline_name="test_pipeline",
                pipeline_status="success",
                node_results={"node1": "result1"},
            )

            # Checkpoint should have failed
            assert "checkpoint" in results
            # Note: checkpoint might be skipped if memory port check fails early
            # The important thing is adapter cleanup still runs

            # Critical: Adapter cleanup should have run despite checkpoint issues
            assert "adapter_cleanup" in results
            assert mock_adapter.close_called is True

    @pytest.mark.asyncio
    async def test_all_cleanup_runs_with_multiple_failures(self):
        """Test that all cleanup operations run even with multiple failures."""
        # Setup
        mock_memory = MockMemory()
        mock_secret = MockSecretPort({"KEY": "value"})
        mock_adapter = MockAdapterWithHealth()

        pre_config = HookConfig(enable_secret_injection=True)
        pre_manager = PreDagHookManager(pre_config)

        # Multiple failing custom hooks
        def hook1(ports, context, node_results, status, error):
            raise ValueError("Hook 1 failed")

        def hook2(ports, context, node_results, status, error):
            raise TypeError("Hook 2 failed")

        post_config = PostDagHookConfig(
            custom_hooks=[hook1, hook2],
            enable_secret_cleanup=True,
            enable_adapter_cleanup=True,
        )
        post_manager = PostDagHookManager(post_config, pre_manager)

        mock_context = NodeExecutionContext(dag_id="test-dag")

        async with ExecutionContext(
            observer_manager=None,
            policy_manager=None,
            run_id="test-run",
            ports={"memory": mock_memory, "secret": mock_secret, "adapter": mock_adapter},
        ):
            # Inject secret
            await pre_manager.execute_hooks(context=mock_context, pipeline_name="test")

            # Run post-hooks with multiple failures
            results = await post_manager.execute_hooks(
                context=mock_context,
                pipeline_name="test",
                pipeline_status="success",
                node_results={},
            )

            # Both custom hooks failed
            assert "hook1" in results and "error" in results["hook1"]
            assert "hook2" in results and "error" in results["hook2"]

            # Critical: Both cleanup operations still ran
            assert "secret_cleanup" in results
            assert results["secret_cleanup"]["keys_removed"] == 1

            assert "adapter_cleanup" in results
            assert mock_adapter.close_called is True
