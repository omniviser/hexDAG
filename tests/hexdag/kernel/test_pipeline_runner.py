"""Tests for PipelineRunner — one-liner YAML pipeline execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexdag.kernel.pipeline_runner import (
    PipelineRunner,
    PipelineRunnerError,
    _find_missing_env_vars,
)

if TYPE_CHECKING:
    from pathlib import Path

# Minimal YAML that works without any ports (data_node outputs a static value)
SIMPLE_YAML = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-simple
spec:
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""

# YAML with two nodes in sequence
TWO_NODE_YAML = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-two-nodes
spec:
  nodes:
    - kind: data_node
      metadata:
        name: first
      spec:
        output:
          value: "a"
      dependencies: []
    - kind: data_node
      metadata:
        name: second
      spec:
        output:
          value: "b"
      dependencies: [first]
"""

# YAML with a port that uses an env var
YAML_WITH_ENV_VAR = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-env-var
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses: "${{MOCK_RESPONSE}}"
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""

# YAML with env var that has a default
YAML_WITH_DEFAULT_ENV = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-default-env
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses: "${{MOCK_RESPONSE:default_val}}"
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""


class TestPipelineRunnerInit:
    """Test PipelineRunner construction."""

    def test_default_initialization(self) -> None:
        runner = PipelineRunner()
        assert runner._port_overrides is None
        assert runner._secrets_provider is None
        assert runner._max_concurrent_nodes == 10
        assert runner._secrets_loaded is False

    def test_custom_port_overrides(self) -> None:
        mock_llm = MagicMock()
        runner = PipelineRunner(port_overrides={"llm": mock_llm})
        assert runner._port_overrides == {"llm": mock_llm}

    def test_custom_concurrency(self) -> None:
        runner = PipelineRunner(max_concurrent_nodes=5, default_node_timeout=60.0)
        assert runner._max_concurrent_nodes == 5
        assert runner._default_node_timeout == 60.0

    def test_custom_environment(self) -> None:
        runner = PipelineRunner(environment="staging")
        assert runner._environment == "staging"

    def test_custom_secrets_provider(self) -> None:
        mock_secret = MagicMock()
        runner = PipelineRunner(
            secrets_provider=mock_secret,
            secret_keys=["MY-KEY"],
        )
        assert runner._secrets_provider is mock_secret
        assert runner._secret_keys == ["MY-KEY"]


class TestPipelineRunnerRun:
    """Test PipelineRunner.run() from file."""

    @pytest.mark.asyncio
    async def test_run_simple_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(SIMPLE_YAML)

        runner = PipelineRunner()
        result = await runner.run(yaml_file)
        assert isinstance(result, dict)
        assert "start" in result

    @pytest.mark.asyncio
    async def test_run_file_not_found(self) -> None:
        runner = PipelineRunner()
        with pytest.raises(PipelineRunnerError, match="not found"):
            await runner.run("/nonexistent/pipeline.yaml")

    @pytest.mark.asyncio
    async def test_run_with_input_data(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(SIMPLE_YAML)

        runner = PipelineRunner()
        result = await runner.run(yaml_file, input_data={"key": "val"})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_run_none_input_defaults_to_empty(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(SIMPLE_YAML)

        runner = PipelineRunner()
        result = await runner.run(yaml_file, input_data=None)
        assert isinstance(result, dict)


class TestPipelineRunnerRunFromString:
    """Test PipelineRunner.run_from_string()."""

    @pytest.mark.asyncio
    async def test_run_from_string_basic(self) -> None:
        runner = PipelineRunner()
        result = await runner.run_from_string(SIMPLE_YAML)
        assert isinstance(result, dict)
        assert "start" in result

    @pytest.mark.asyncio
    async def test_run_from_string_two_nodes(self) -> None:
        runner = PipelineRunner()
        result = await runner.run_from_string(TWO_NODE_YAML)
        assert "first" in result
        assert "second" in result


class TestPipelineRunnerPortOverrides:
    """Test port override merging."""

    @pytest.mark.asyncio
    async def test_port_override_replaces_yaml_port(self) -> None:
        """Override replaces YAML-declared port of same name."""
        from hexdag.stdlib.adapters.mock import MockLLM

        yaml_with_port = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        responses: "yaml_response"
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""
        override_llm = MockLLM(responses="override_response")
        runner = PipelineRunner(port_overrides={"llm": override_llm})
        result = await runner.run_from_string(yaml_with_port)
        assert isinstance(result, dict)


class TestPipelineRunnerSecretLoading:
    """Test secret pre-loading into os.environ."""

    @pytest.mark.asyncio
    async def test_constructor_secrets_provider_loads_to_environ(self) -> None:
        """Constructor secrets_provider.load_to_environ() called before ports."""
        mock_provider = AsyncMock()
        mock_provider.load_to_environ = AsyncMock(return_value={"TEST_KEY": "loaded"})

        runner = PipelineRunner(
            secrets_provider=mock_provider,
            secret_keys=["TEST-KEY"],
        )
        await runner.run_from_string(SIMPLE_YAML)

        mock_provider.load_to_environ.assert_awaited_once_with(keys=["TEST-KEY"])
        assert runner._secrets_loaded is True

    @pytest.mark.asyncio
    async def test_runner_level_cache_skips_second_load(self) -> None:
        """Second run() skips load_to_environ (runner-level cache)."""
        mock_provider = AsyncMock()
        mock_provider.load_to_environ = AsyncMock(return_value={})

        runner = PipelineRunner(secrets_provider=mock_provider)

        await runner.run_from_string(SIMPLE_YAML)
        await runner.run_from_string(SIMPLE_YAML)

        # load_to_environ called only once
        assert mock_provider.load_to_environ.await_count == 1

    @pytest.mark.asyncio
    async def test_no_secret_provider_skips_loading(self) -> None:
        """No secret port configured → no load_to_environ call."""
        runner = PipelineRunner()
        result = await runner.run_from_string(SIMPLE_YAML)
        assert isinstance(result, dict)
        assert runner._secrets_loaded is False


class TestEnvVarValidation:
    """Test env var validation (pre-flight check)."""

    def test_missing_env_var_detected(self) -> None:
        """${VAR} without default and not in os.environ → flagged."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig

        config = PipelineConfig(
            ports={
                "llm": {
                    "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                    "config": {"api_key": "${MISSING_KEY}"},
                }
            },
            metadata={},
            nodes=[],
        )
        # Make sure MISSING_KEY is not in env
        os.environ.pop("MISSING_KEY", None)
        missing = _find_missing_env_vars(config)
        assert "MISSING_KEY" in missing

    def test_env_var_with_default_not_flagged(self) -> None:
        """${VAR:default} → not flagged."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig

        config = PipelineConfig(
            ports={
                "llm": {
                    "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                    "config": {"api_key": "${SOME_VAR:my_default}"},
                }
            },
            metadata={},
            nodes=[],
        )
        missing = _find_missing_env_vars(config)
        assert "SOME_VAR" not in missing

    def test_present_env_var_not_flagged(self) -> None:
        """${VAR} present in os.environ → not flagged."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig

        os.environ["PRESENT_KEY"] = "some_value"
        try:
            config = PipelineConfig(
                ports={
                    "llm": {
                        "adapter": "hexdag.stdlib.adapters.mock.MockLLM",
                        "config": {"api_key": "${PRESENT_KEY}"},
                    }
                },
                metadata={},
                nodes=[],
            )
            missing = _find_missing_env_vars(config)
            assert "PRESENT_KEY" not in missing
        finally:
            del os.environ["PRESENT_KEY"]

    def test_portless_config_returns_empty(self) -> None:
        """No ports → no missing vars."""
        from hexdag.kernel.domain.pipeline_config import PipelineConfig

        config = PipelineConfig(ports={}, metadata={}, nodes=[])
        assert _find_missing_env_vars(config) == []

    @pytest.mark.asyncio
    async def test_run_raises_on_missing_env_var(self) -> None:
        """PipelineRunner.run raises listing all missing deferred vars.

        Uses a secret-pattern var name so Phase 1 defers it (preserves ${VAR}),
        and the runner's pre-flight env var validation catches it.
        """
        yaml_content = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        api_key: "${NONEXISTENT_API_KEY}"
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""
        os.environ.pop("NONEXISTENT_API_KEY", None)
        runner = PipelineRunner()
        with pytest.raises(PipelineRunnerError, match="NONEXISTENT_API_KEY"):
            await runner.run_from_string(yaml_content)


class TestPipelineRunnerValidate:
    """Test validate() dry-run."""

    @pytest.mark.asyncio
    async def test_valid_pipeline_returns_empty(self) -> None:
        runner = PipelineRunner()
        issues = await runner.validate(yaml_content=SIMPLE_YAML)
        assert issues == []

    @pytest.mark.asyncio
    async def test_invalid_yaml_returns_issues(self) -> None:
        runner = PipelineRunner()
        issues = await runner.validate(yaml_content="not: valid: yaml: {{{{")
        assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_missing_env_var_in_validation(self) -> None:
        yaml_content = """\
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.mock.MockLLM
      config:
        api_key: "${VALIDATE_MISSING}"
  nodes:
    - kind: data_node
      metadata:
        name: start
      spec:
        output:
          value: "hello"
      dependencies: []
"""
        os.environ.pop("VALIDATE_MISSING", None)
        runner = PipelineRunner()
        issues = await runner.validate(yaml_content=yaml_content)
        assert any("VALIDATE_MISSING" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_validate_requires_path_or_content(self) -> None:
        runner = PipelineRunner()
        with pytest.raises(PipelineRunnerError, match="Provide either"):
            await runner.validate()

    @pytest.mark.asyncio
    async def test_validate_from_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(SIMPLE_YAML)

        runner = PipelineRunner()
        issues = await runner.validate(pipeline_path=yaml_file)
        assert issues == []


class TestPipelineRunnerEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_portless_pipeline(self) -> None:
        """Pipeline with only data_node, no ports section."""
        runner = PipelineRunner()
        result = await runner.run_from_string(SIMPLE_YAML)
        assert "start" in result
