"""Integration tests for the complete secret management system.

Tests the end-to-end secret management flow including:
- SecretField() declarative helper
- Auto-resolution from environment variables
- LocalSecretAdapter with Orchestrator
- Secret injection and cleanup hooks
- LLM adapters using SecretField()
"""

import os

import pytest
from pydantic import SecretStr

from hexai.adapters.llm.anthropic_adapter import AnthropicAdapter
from hexai.adapters.llm.openai_adapter import OpenAIAdapter
from hexai.adapters.memory.in_memory_memory import InMemoryMemory
from hexai.adapters.secret.local_secret_adapter import LocalSecretAdapter
from hexai.core.application.orchestrator import Orchestrator
from hexai.core.configurable import AdapterConfig, ConfigurableAdapter, SecretField
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.orchestration.hooks import HookConfig, PostDagHookConfig
from hexai.core.types import Secret

# ===================================================================
# Test Adapters Using SecretField()
# ===================================================================


class SecretTestAdapter(ConfigurableAdapter):
    """Test adapter demonstrating SecretField() usage."""

    class Config(AdapterConfig):
        api_key: SecretStr | None = SecretField(env_var="TEST_API_KEY", description="Test API key")
        api_secret: SecretStr | None = SecretField(
            env_var="TEST_API_SECRET", description="Test API secret"
        )
        service_url: str = "https://api.example.com"

    config: Config

    def get_api_key_value(self) -> str | None:
        """Get the raw API key value (for testing)."""
        return self.config.api_key.get_secret_value() if self.config.api_key else None

    def get_api_secret_value(self) -> str | None:
        """Get the raw API secret value (for testing)."""
        return self.config.api_secret.get_secret_value() if self.config.api_secret else None


# ===================================================================
# SecretField() Auto-Resolution Tests
# ===================================================================


@pytest.mark.asyncio
async def test_secret_field_resolves_from_env():
    """Test SecretField automatically resolves from environment variables."""
    # Set environment variables
    os.environ["TEST_API_KEY"] = "sk-test-key-123"
    os.environ["TEST_API_SECRET"] = "secret-value-456"

    try:
        # Create adapter - secrets should auto-resolve
        adapter = SecretTestAdapter()

        # Verify secrets were resolved
        assert adapter.get_api_key_value() == "sk-test-key-123"
        assert adapter.get_api_secret_value() == "secret-value-456"

        # Verify secrets are hidden in string representation
        assert str(adapter.config.api_key) == "**********"
        assert str(adapter.config.api_secret) == "**********"

    finally:
        # Cleanup
        del os.environ["TEST_API_KEY"]
        del os.environ["TEST_API_SECRET"]


@pytest.mark.asyncio
async def test_secret_field_explicit_value_override():
    """Test explicit values override environment variables."""
    os.environ["TEST_API_KEY"] = "env-key-123"

    try:
        # Provide explicit value
        adapter = SecretTestAdapter(api_key="explicit-key-456")

        # Explicit value should win
        assert adapter.get_api_key_value() == "explicit-key-456"

    finally:
        del os.environ["TEST_API_KEY"]


@pytest.mark.asyncio
async def test_secret_field_missing_optional():
    """Test missing optional secret fields don't cause errors."""
    # Don't set any environment variables
    adapter = SecretTestAdapter()

    # Optional secrets should be None
    assert adapter.config.api_key is None
    assert adapter.config.api_secret is None


@pytest.mark.asyncio
async def test_secret_field_with_pydantic_secret_str():
    """Test SecretField works with Pydantic SecretStr."""
    os.environ["TEST_API_KEY"] = "sk-hidden-key"

    try:
        adapter = SecretTestAdapter()

        # Verify it's a SecretStr
        assert isinstance(adapter.config.api_key, SecretStr)

        # Verify repr is hidden
        assert repr(adapter.config.api_key) == "SecretStr('**********')"

        # Verify actual value accessible
        assert adapter.config.api_key.get_secret_value() == "sk-hidden-key"

    finally:
        del os.environ["TEST_API_KEY"]


# ===================================================================
# LocalSecretAdapter Integration Tests
# ===================================================================


@pytest.mark.asyncio
async def test_local_secret_adapter_get_secret():
    """Test LocalSecretAdapter retrieves secrets from environment."""
    os.environ["OPENAI_API_KEY"] = "sk-local-test-123"

    try:
        adapter = LocalSecretAdapter()
        secret = await adapter.aget_secret("OPENAI_API_KEY")

        # Verify Secret wrapper
        assert isinstance(secret, Secret)
        assert secret.get() == "sk-local-test-123"

        # Verify string representation is hidden
        assert str(secret) == "<SECRET>"
        assert repr(secret) == "<SECRET>"

    finally:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.asyncio
async def test_local_secret_adapter_with_prefix():
    """Test LocalSecretAdapter respects env_prefix."""
    os.environ["MYAPP_API_KEY"] = "sk-prefixed-key"

    try:
        adapter = LocalSecretAdapter(env_prefix="MYAPP_")
        secret = await adapter.aget_secret("API_KEY")

        assert secret.get() == "sk-prefixed-key"

    finally:
        del os.environ["MYAPP_API_KEY"]


@pytest.mark.asyncio
async def test_local_secret_adapter_missing_secret_raises():
    """Test LocalSecretAdapter raises KeyError for missing secrets."""
    adapter = LocalSecretAdapter()

    with pytest.raises(KeyError, match="Secret 'NONEXISTENT_KEY' not found"):
        await adapter.aget_secret("NONEXISTENT_KEY")


@pytest.mark.asyncio
async def test_local_secret_adapter_empty_secret_raises():
    """Test LocalSecretAdapter raises ValueError for empty secrets."""
    os.environ["EMPTY_KEY"] = ""

    try:
        adapter = LocalSecretAdapter()

        with pytest.raises(ValueError, match="cannot be empty"):
            await adapter.aget_secret("EMPTY_KEY")

    finally:
        del os.environ["EMPTY_KEY"]


@pytest.mark.asyncio
async def test_local_secret_adapter_allow_empty():
    """Test LocalSecretAdapter allows empty secrets when configured."""
    os.environ["EMPTY_KEY"] = ""

    try:
        adapter = LocalSecretAdapter(allow_empty=True)
        secret = await adapter.aget_secret("EMPTY_KEY")

        assert secret.get() == ""

    finally:
        del os.environ["EMPTY_KEY"]


@pytest.mark.asyncio
async def test_local_secret_adapter_load_to_memory():
    """Test LocalSecretAdapter loads secrets into Memory."""
    os.environ["KEY1"] = "value1"
    os.environ["KEY2"] = "value2"

    try:
        adapter = LocalSecretAdapter()
        memory = InMemoryMemory()

        # Load specific secrets
        mapping = await adapter.aload_secrets_to_memory(memory=memory, keys=["KEY1", "KEY2"])

        # Verify mapping
        assert mapping == {"KEY1": "secret:KEY1", "KEY2": "secret:KEY2"}

        # Verify values in memory
        assert await memory.aget("secret:KEY1") == "value1"
        assert await memory.aget("secret:KEY2") == "value2"

    finally:
        del os.environ["KEY1"]
        del os.environ["KEY2"]


@pytest.mark.asyncio
async def test_local_secret_adapter_custom_prefix():
    """Test LocalSecretAdapter with custom memory prefix."""
    os.environ["API_KEY"] = "sk-custom-prefix"

    try:
        adapter = LocalSecretAdapter()
        memory = InMemoryMemory()

        mapping = await adapter.aload_secrets_to_memory(
            memory=memory, prefix="env:", keys=["API_KEY"]
        )

        assert mapping == {"API_KEY": "env:API_KEY"}
        assert await memory.aget("env:API_KEY") == "sk-custom-prefix"

    finally:
        del os.environ["API_KEY"]


@pytest.mark.asyncio
async def test_local_secret_adapter_health_check():
    """Test LocalSecretAdapter health check."""
    os.environ["TEST_SECRET"] = "value"

    try:
        adapter = LocalSecretAdapter()
        status = await adapter.ahealth_check()

        assert status.status == "healthy"
        assert status.adapter_name == "local_env"
        assert status.port_name == "secret"
        assert "env_vars_count" in status.details

    finally:
        del os.environ["TEST_SECRET"]


# ===================================================================
# Orchestrator + LocalSecretAdapter Integration Tests
# ===================================================================


@pytest.mark.asyncio
async def test_orchestrator_with_local_secret_adapter():
    """Test Orchestrator uses LocalSecretAdapter for secret injection."""
    os.environ["PIPELINE_API_KEY"] = "sk-pipeline-test"
    os.environ["PIPELINE_DB_PASSWORD"] = "db-pass-123"

    try:
        secrets_captured = {}

        def capture_secrets(x=None, **kwargs):
            """Capture secrets from Memory during execution."""
            from hexai.core.context import get_port

            memory = get_port("memory")
            if memory:
                secrets_captured["api_key"] = memory.storage.get("secret:PIPELINE_API_KEY")
                secrets_captured["db_password"] = memory.storage.get("secret:PIPELINE_DB_PASSWORD")
            return "captured"

        graph = DirectedGraph()
        graph.add(NodeSpec(name="capture", fn=capture_secrets, deps=set()))

        # Create orchestrator with LocalSecretAdapter
        orchestrator = Orchestrator(
            ports={
                "memory": InMemoryMemory(),
                "secret": LocalSecretAdapter(),
            },
            pre_hook_config=HookConfig(
                enable_health_checks=True,
                enable_secret_injection=True,
                secret_keys=["PIPELINE_API_KEY", "PIPELINE_DB_PASSWORD"],
            ),
            post_hook_config=PostDagHookConfig(
                enable_secret_cleanup=True,
            ),
        )

        await orchestrator.run(graph, None)

        # Verify secrets were loaded
        assert secrets_captured["api_key"] == "sk-pipeline-test"
        assert secrets_captured["db_password"] == "db-pass-123"

    finally:
        del os.environ["PIPELINE_API_KEY"]
        del os.environ["PIPELINE_DB_PASSWORD"]


# ===================================================================
# Real Adapter Integration Tests
# ===================================================================


@pytest.mark.asyncio
async def test_anthropic_adapter_secret_auto_resolution():
    """Test AnthropicAdapter auto-resolves API key from environment."""
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key-123"

    try:
        # Create adapter - API key should auto-resolve
        adapter = AnthropicAdapter()

        # Verify API key was resolved (client should be initialized)
        assert adapter.client is not None

        # Verify API key is hidden
        assert str(adapter.config.api_key) == "**********"

    finally:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.asyncio
async def test_anthropic_adapter_explicit_api_key():
    """Test AnthropicAdapter accepts explicit API key."""
    adapter = AnthropicAdapter(api_key="sk-ant-explicit-key")

    assert adapter.client is not None
    assert str(adapter.config.api_key) == "**********"


@pytest.mark.asyncio
async def test_anthropic_adapter_missing_api_key_raises():
    """Test AnthropicAdapter raises when API key is missing."""
    # Make sure env var is not set
    os.environ.pop("ANTHROPIC_API_KEY", None)

    with pytest.raises(ValueError, match="API key required"):
        AnthropicAdapter()


@pytest.mark.asyncio
async def test_openai_adapter_secret_auto_resolution():
    """Test OpenAIAdapter auto-resolves API key from environment."""
    os.environ["OPENAI_API_KEY"] = "sk-openai-test-key-123"

    try:
        adapter = OpenAIAdapter()

        assert adapter.client is not None
        assert str(adapter.config.api_key) == "**********"

    finally:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.asyncio
async def test_openai_adapter_explicit_api_key():
    """Test OpenAIAdapter accepts explicit API key."""
    adapter = OpenAIAdapter(api_key="sk-openai-explicit-key")

    assert adapter.client is not None
    assert str(adapter.config.api_key) == "**********"


@pytest.mark.asyncio
async def test_openai_adapter_missing_api_key_raises():
    """Test OpenAIAdapter raises when API key is missing."""
    os.environ.pop("OPENAI_API_KEY", None)

    with pytest.raises(ValueError, match="API key required"):
        OpenAIAdapter()


# ===================================================================
# End-to-End Secret Lifecycle Tests
# ===================================================================


@pytest.mark.asyncio
async def test_end_to_end_secret_lifecycle():
    """Test complete secret lifecycle: resolution → injection → usage → cleanup."""
    os.environ["E2E_API_KEY"] = "sk-e2e-test-123"

    try:
        execution_data = {}

        def use_secret(x=None, **kwargs):
            """Use secret during pipeline execution."""
            from hexai.core.context import get_port

            memory = get_port("memory")
            if memory:
                # Secret should be available during execution
                execution_data["secret_value"] = memory.storage.get("secret:E2E_API_KEY")
                execution_data["secret_exists"] = "secret:E2E_API_KEY" in memory.storage
            return "completed"

        graph = DirectedGraph()
        graph.add(NodeSpec(name="worker", fn=use_secret, deps=set()))

        memory = InMemoryMemory()
        orchestrator = Orchestrator(
            ports={
                "memory": memory,
                "secret": LocalSecretAdapter(),
            },
            pre_hook_config=HookConfig(
                enable_secret_injection=True,
                secret_keys=["E2E_API_KEY"],
            ),
            post_hook_config=PostDagHookConfig(
                enable_secret_cleanup=True,
            ),
        )

        await orchestrator.run(graph, None)

        # Verify secret was available during execution
        assert execution_data["secret_value"] == "sk-e2e-test-123"
        assert execution_data["secret_exists"] is True

        # Verify secret was cleaned up after execution
        assert await memory.aget("secret:E2E_API_KEY") is None

    finally:
        del os.environ["E2E_API_KEY"]


@pytest.mark.asyncio
async def test_secret_isolation_between_pipelines():
    """Test secrets are properly isolated between pipeline runs."""
    os.environ["SHARED_KEY"] = "sk-shared-value"

    try:
        captured_values = []

        def capture(x=None, **kwargs):
            from hexai.core.context import get_port

            memory = get_port("memory")
            if memory:
                captured_values.append(memory.storage.get("secret:SHARED_KEY"))
            return "done"

        graph = DirectedGraph()
        graph.add(NodeSpec(name="capture", fn=capture, deps=set()))

        memory = InMemoryMemory()
        orchestrator = Orchestrator(
            ports={
                "memory": memory,
                "secret": LocalSecretAdapter(),
            },
            pre_hook_config=HookConfig(
                enable_secret_injection=True,
                secret_keys=["SHARED_KEY"],
            ),
            post_hook_config=PostDagHookConfig(
                enable_secret_cleanup=True,
            ),
        )

        # Run twice
        await orchestrator.run(graph, None)
        await orchestrator.run(graph, None)

        # Both runs should have access to secret
        assert len(captured_values) == 2
        assert captured_values[0] == "sk-shared-value"
        assert captured_values[1] == "sk-shared-value"

        # Secret should be cleaned up
        assert await memory.aget("secret:SHARED_KEY") is None

    finally:
        del os.environ["SHARED_KEY"]
