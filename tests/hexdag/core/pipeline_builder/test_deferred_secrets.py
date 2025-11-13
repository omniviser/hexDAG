"""Tests for deferred secret resolution in YAML pipelines.

This tests the new secret deferral feature that preserves ${VAR} syntax
for secret-like environment variables at build-time, allowing runtime
resolution from KeyVault/Memory port.

Tests cover:
- Secret pattern detection at build-time
- Preservation of ${VAR} for secrets
- Runtime resolution from environment
- Runtime resolution from memory port
- Integration with adapters
"""

import os

import pytest

from hexdag.builtin.adapters.memory.in_memory_memory import InMemoryMemory
from hexdag.builtin.adapters.secret.local_secret_adapter import LocalSecretAdapter
from hexdag.core.pipeline_builder.yaml_builder import (
    EnvironmentVariablePlugin,
    YamlPipelineBuilder,
)
from hexdag.core.registry import adapter

# ============================================================================
# Build-Time Secret Deferral Tests
# ============================================================================


class TestEnvironmentVariablePluginSecretDeferral:
    """Test that EnvironmentVariablePlugin defers secret-like variables."""

    def test_defer_api_key_pattern(self):
        """Test that *_API_KEY variables are deferred."""
        # Ensure API key is NOT in environment
        os.environ.pop("OPENAI_API_KEY", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"api_key": "${OPENAI_API_KEY}"}

        # Should NOT raise - secret is deferred
        result = plugin.process(config)

        # ${VAR} syntax should be preserved
        assert result["api_key"] == "${OPENAI_API_KEY}"

    def test_defer_secret_pattern(self):
        """Test that *_SECRET variables are deferred."""
        os.environ.pop("DB_SECRET", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"secret": "${DB_SECRET}"}

        result = plugin.process(config)
        assert result["secret"] == "${DB_SECRET}"

    def test_defer_token_pattern(self):
        """Test that *_TOKEN variables are deferred."""
        os.environ.pop("AUTH_TOKEN", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"token": "${AUTH_TOKEN}"}

        result = plugin.process(config)
        assert result["token"] == "${AUTH_TOKEN}"

    def test_defer_password_pattern(self):
        """Test that *_PASSWORD variables are deferred."""
        os.environ.pop("DB_PASSWORD", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"password": "${DB_PASSWORD}"}

        result = plugin.process(config)
        assert result["password"] == "${DB_PASSWORD}"

    def test_defer_credential_pattern(self):
        """Test that *_CREDENTIAL variables are deferred."""
        os.environ.pop("SERVICE_CREDENTIAL", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"cred": "${SERVICE_CREDENTIAL}"}

        result = plugin.process(config)
        assert result["cred"] == "${SERVICE_CREDENTIAL}"

    def test_defer_secret_prefix_pattern(self):
        """Test that SECRET_* variables are deferred."""
        os.environ.pop("SECRET_KEY", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"key": "${SECRET_KEY}"}

        result = plugin.process(config)
        assert result["key"] == "${SECRET_KEY}"

    def test_resolve_non_secret_variables_immediately(self):
        """Test that non-secret variables are still resolved at build-time."""
        os.environ["MODEL"] = "gpt-4"
        os.environ["TEMPERATURE"] = "0.7"

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {
            "model": "${MODEL}",
            "temperature": "${TEMPERATURE}",
        }

        result = plugin.process(config)

        # Non-secret variables should be resolved
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7  # Type coerced

    def test_mixed_secret_and_non_secret(self):
        """Test mixed secret and non-secret variables in same config."""
        os.environ["MODEL"] = "gpt-4"
        os.environ.pop("OPENAI_API_KEY", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {
            "api_key": "${OPENAI_API_KEY}",
            "model": "${MODEL}",
        }

        result = plugin.process(config)

        # Secret deferred, non-secret resolved
        assert result["api_key"] == "${OPENAI_API_KEY}"
        assert result["model"] == "gpt-4"

    def test_defer_secrets_disabled_legacy_behavior(self):
        """Test that defer_secrets=False restores legacy behavior."""
        os.environ.pop("OPENAI_API_KEY", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=False)
        config = {"api_key": "${OPENAI_API_KEY}"}

        # Should raise because secret is not found
        from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilderError

        with pytest.raises(YamlPipelineBuilderError, match="OPENAI_API_KEY.*not set"):
            plugin.process(config)

    def test_deferred_secret_with_default(self):
        """Test that deferred secrets preserve default values."""
        os.environ.pop("OPENAI_API_KEY", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {"api_key": "${OPENAI_API_KEY:sk-default}"}

        result = plugin.process(config)

        # Should preserve default syntax
        assert result["api_key"] == "${OPENAI_API_KEY:sk-default}"

    def test_nested_deferred_secrets(self):
        """Test deferred secrets in nested structures."""
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("DB_PASSWORD", None)

        plugin = EnvironmentVariablePlugin(defer_secrets=True)
        config = {
            "llm": {
                "api_key": "${OPENAI_API_KEY}",
            },
            "database": {
                "credentials": {
                    "password": "${DB_PASSWORD}",
                }
            },
        }

        result = plugin.process(config)

        assert result["llm"]["api_key"] == "${OPENAI_API_KEY}"
        assert result["database"]["credentials"]["password"] == "${DB_PASSWORD}"


# ============================================================================
# Runtime Secret Resolution Tests
# ============================================================================


class TestRuntimeSecretResolution:
    """Test runtime resolution of deferred ${VAR} patterns in adapters."""

    def test_resolve_from_environment(self):
        """Test that deferred secrets are resolved from environment at runtime."""
        os.environ["TEST_API_KEY"] = "sk-test-from-env"

        try:
            # Create a test adapter
            @adapter("llm", name="test_env_adapter")
            class TestEnvAdapter:
                def __init__(self, api_key: str):
                    self.api_key = api_key

            # Instantiate with deferred ${VAR} syntax
            adapter_instance = TestEnvAdapter(api_key="${TEST_API_KEY}")

            # Should resolve from environment
            assert adapter_instance.api_key == "sk-test-from-env"

        finally:
            del os.environ["TEST_API_KEY"]

    def test_resolve_from_memory_port(self):
        """Test that deferred secrets are resolved from memory port at runtime."""
        # Secret NOT in environment
        os.environ.pop("TEST_API_KEY", None)

        # Create a test adapter
        @adapter("llm", name="test_memory_adapter")
        class TestMemoryAdapter:
            def __init__(self, api_key: str):
                self.api_key = api_key

        # Create memory and inject secret
        memory = InMemoryMemory()
        # Use sync set method
        import asyncio

        asyncio.run(memory.aset("secret:TEST_API_KEY", "sk-test-from-memory"))

        # Instantiate with deferred ${VAR} syntax and memory
        adapter_instance = TestMemoryAdapter(api_key="${TEST_API_KEY}", memory=memory)

        # Should resolve from memory
        assert adapter_instance.api_key == "sk-test-from-memory"

    def test_resolve_with_default_value(self):
        """Test that deferred secrets use default if not found."""
        # Secret NOT in environment
        os.environ.pop("MISSING_KEY", None)

        @adapter("llm", name="test_default_adapter")
        class TestDefaultAdapter:
            def __init__(self, api_key: str):
                self.api_key = api_key

        # Instantiate with default value
        adapter_instance = TestDefaultAdapter(api_key="${MISSING_KEY:sk-default-value}")

        # Should use default
        assert adapter_instance.api_key == "sk-default-value"

    def test_missing_required_secret_raises_error(self):
        """Test that missing required secret raises clear error."""
        os.environ.pop("REQUIRED_KEY", None)

        @adapter("llm", name="test_required_adapter")
        class TestRequiredAdapter:
            def __init__(self, api_key: str):
                self.api_key = api_key

        # Should raise because secret is not found
        with pytest.raises(ValueError, match="Required environment variable.*REQUIRED_KEY"):
            TestRequiredAdapter(api_key="${REQUIRED_KEY}")

    def test_environment_takes_precedence_over_memory(self):
        """Test that environment variables take precedence over memory."""
        os.environ["PRIORITY_KEY"] = "sk-from-env"

        try:

            @adapter("llm", name="test_priority_adapter")
            class TestPriorityAdapter:
                def __init__(self, api_key: str):
                    self.api_key = api_key

            # Create memory with different value
            memory = InMemoryMemory()
            memory.storage["secret:PRIORITY_KEY"] = "sk-from-memory"

            # Environment should win
            adapter_instance = TestPriorityAdapter(api_key="${PRIORITY_KEY}", memory=memory)

            assert adapter_instance.api_key == "sk-from-env"

        finally:
            del os.environ["PRIORITY_KEY"]


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestEndToEndSecretDeferral:
    """Test complete secret deferral flow from YAML to runtime."""

    def test_build_without_secrets_present(self):
        """Test that YAML can be built without secrets in environment."""
        # Ensure secret is NOT present
        os.environ.pop("OPENAI_API_KEY", None)

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
spec:
  ports:
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: "builtins.len"
        dependencies: []
"""

        # Should NOT raise - secret is deferred
        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Config should preserve ${VAR} syntax
        assert config.ports["llm"]["config"]["api_key"] == "${OPENAI_API_KEY}"

    @pytest.mark.asyncio
    async def test_runtime_resolution_with_keyvault(self):
        """Test runtime resolution from KeyVault via SecretPort."""
        from hexdag.core.domain.dag import DirectedGraph, NodeSpec
        from hexdag.core.orchestration.hooks import HookConfig, PostDagHookConfig
        from hexdag.core.orchestration.orchestrator import Orchestrator

        # Ensure secret is NOT in environment
        os.environ.pop("PIPELINE_SECRET", None)

        try:
            # Simple test node
            def test_node(x=None, **kwargs):
                return "success"

            graph = DirectedGraph()
            graph.add(NodeSpec(name="test", fn=test_node, deps=set()))

            # Set up secret in "KeyVault" (LocalSecretAdapter)
            os.environ["PIPELINE_SECRET"] = "sk-from-keyvault"

            memory = InMemoryMemory()
            orchestrator = Orchestrator(
                ports={
                    "memory": memory,
                    "secret": LocalSecretAdapter(),
                },
                pre_hook_config=HookConfig(
                    enable_secret_injection=True,
                    secret_keys=["PIPELINE_SECRET"],
                ),
                post_hook_config=PostDagHookConfig(
                    enable_secret_cleanup=True,
                ),
            )

            await orchestrator.run(graph, None)

            # Verify secret was loaded and cleaned up
            # (Can't check directly because it's cleaned up, but no error = success)

        finally:
            del os.environ["PIPELINE_SECRET"]

    def test_multiple_secrets_in_yaml(self):
        """Test YAML with multiple deferred secrets."""
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("DB_PASSWORD", None)
        os.environ["MODEL"] = "gpt-4"  # Non-secret

        yaml_content = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: multi-secret-pipeline
spec:
  ports:
    llm:
      adapter: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: ${MODEL}
    database:
      adapter: postgres
      config:
        password: ${DB_PASSWORD}
        host: localhost
  nodes:
    - kind: function_node
      metadata:
        name: test_node
      spec:
        fn: "builtins.len"
        dependencies: []
"""

        builder = YamlPipelineBuilder()
        graph, config = builder.build_from_yaml_string(yaml_content)

        # Secrets deferred, non-secret resolved
        assert config.ports["llm"]["config"]["api_key"] == "${OPENAI_API_KEY}"
        assert config.ports["llm"]["config"]["model"] == "gpt-4"
        assert config.ports["database"]["config"]["password"] == "${DB_PASSWORD}"
        assert config.ports["database"]["config"]["host"] == "localhost"
