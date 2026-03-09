"""Tests for PipelineRunner — one-liner YAML pipeline execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from hexdag.kernel.config.models import DefaultCaps, DefaultLimits, HexDAGConfig, LoggingConfig
from hexdag.kernel.orchestration.models import OrchestratorConfig
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


# ---------------------------------------------------------------------------
# Config propagation tests
# ---------------------------------------------------------------------------


class TestPipelineRunnerConfigDefaults:
    """Test that config defaults are used when no explicit args provided."""

    def test_no_config_uses_hardcoded_defaults(self) -> None:
        runner = PipelineRunner()
        assert runner._max_concurrent_nodes == 10
        assert runner._strict_validation is False
        assert runner._default_node_timeout is None

    def test_config_provides_defaults(self) -> None:
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(
                max_concurrent_nodes=5,
                strict_validation=True,
                default_node_timeout=30.0,
            ),
        )
        runner = PipelineRunner(config=config)
        assert runner._max_concurrent_nodes == 5
        assert runner._strict_validation is True
        assert runner._default_node_timeout == 30.0


class TestPipelineRunnerExplicitOverridesConfig:
    """Test that explicit constructor args override config defaults."""

    def test_explicit_max_concurrent_overrides_config(self) -> None:
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=5),
        )
        runner = PipelineRunner(config=config, max_concurrent_nodes=20)
        assert runner._max_concurrent_nodes == 20

    def test_explicit_strict_validation_overrides_config(self) -> None:
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(strict_validation=True),
        )
        runner = PipelineRunner(config=config, strict_validation=False)
        assert runner._strict_validation is False

    def test_explicit_timeout_overrides_config(self) -> None:
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(default_node_timeout=30.0),
        )
        runner = PipelineRunner(config=config, default_node_timeout=120.0)
        assert runner._default_node_timeout == 120.0

    def test_none_explicit_falls_through_to_config(self) -> None:
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(
                max_concurrent_nodes=7,
                strict_validation=True,
            ),
        )
        runner = PipelineRunner(config=config, max_concurrent_nodes=None)
        assert runner._max_concurrent_nodes == 7
        assert runner._strict_validation is True


class TestPipelineRunnerConfigStored:
    """Test that the config object is accessible."""

    def test_config_stored(self) -> None:
        config = HexDAGConfig(
            limits=DefaultLimits(max_llm_calls=100),
            caps=DefaultCaps(deny=["secret"]),
        )
        runner = PipelineRunner(config=config)
        assert runner._config is config

    def test_config_none_by_default(self) -> None:
        runner = PipelineRunner()
        assert runner._config is None


class TestPipelineRunnerBackwardCompatibility:
    """Test backward compatibility with old-style constructor usage."""

    def test_explicit_int_values_still_work(self) -> None:
        """Callers that pass explicit values should still work identically."""
        runner = PipelineRunner(
            max_concurrent_nodes=15,
            strict_validation=True,
            default_node_timeout=60.0,
        )
        assert runner._max_concurrent_nodes == 15
        assert runner._strict_validation is True
        assert runner._default_node_timeout == 60.0


class TestResolveOrchestratorSettings:
    """Test _resolve_orchestrator_settings priority chain.

    Bug 1 fix: explicit constructor args must NOT be clobbered by
    per-pipeline spec defaults.
    """

    def test_explicit_arg_beats_pipeline_spec(self) -> None:
        """Explicit constructor max_concurrent_nodes=20 should survive
        even when pipeline_config.orchestrator is set."""
        runner = PipelineRunner(max_concurrent_nodes=20)
        # Simulate a pipeline that declares orchestrator with defaults
        pc_orch = OrchestratorConfig(max_concurrent_nodes=3)
        max_nodes, strict, timeout = runner._resolve_orchestrator_settings(pc_orch)
        assert max_nodes == 20  # constructor wins
        assert strict is False  # hardcoded default (no explicit, no pipeline override)
        assert timeout is None

    def test_explicit_strict_beats_pipeline_spec(self) -> None:
        runner = PipelineRunner(strict_validation=True)
        pc_orch = OrchestratorConfig(strict_validation=False)
        _, strict, _ = runner._resolve_orchestrator_settings(pc_orch)
        assert strict is True  # constructor wins

    def test_pipeline_spec_overrides_config_when_no_explicit(self) -> None:
        """Per-pipeline spec should override kind: Config defaults
        when no explicit constructor arg is set."""
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=5),
        )
        runner = PipelineRunner(config=config)  # no explicit max_concurrent_nodes
        pc_orch = OrchestratorConfig(max_concurrent_nodes=3)
        max_nodes, _, _ = runner._resolve_orchestrator_settings(pc_orch)
        assert max_nodes == 3  # pipeline spec wins over config

    def test_pipeline_spec_none_uses_config(self) -> None:
        """When no per-pipeline spec, config defaults apply."""
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(
                max_concurrent_nodes=7,
                strict_validation=True,
                default_node_timeout=45.0,
            ),
        )
        runner = PipelineRunner(config=config)
        max_nodes, strict, timeout = runner._resolve_orchestrator_settings(None)
        assert max_nodes == 7
        assert strict is True
        assert timeout == 45.0

    def test_empty_pipeline_orchestrator_does_not_clobber_config(self) -> None:
        """An empty orchestrator: {} in Pipeline YAML (which creates
        OrchestratorConfig with defaults) should NOT clobber a higher
        config value when constructor didn't set explicit args."""
        config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=5, strict_validation=True),
        )
        runner = PipelineRunner(config=config)
        # Empty orchestrator: {} → OrchestratorConfig() → max=10, strict=False
        empty_orch = OrchestratorConfig()
        max_nodes, strict, _ = runner._resolve_orchestrator_settings(empty_orch)
        # Pipeline spec overrides config (by design), even when "empty".
        # This is correct: an explicit orchestrator: {} means "use defaults".
        assert max_nodes == 10
        assert strict is False

    def test_effective_config_from_inline(self) -> None:
        """effective_config parameter should be used for config-level defaults."""
        runner = PipelineRunner()  # no constructor config
        inline_config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=4),
        )
        max_nodes, _, _ = runner._resolve_orchestrator_settings(
            None, effective_config=inline_config
        )
        assert max_nodes == 4

    def test_constructor_config_beats_inline_config(self) -> None:
        """Constructor config should win over inline config."""
        ctor_config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=8),
        )
        runner = PipelineRunner(config=ctor_config)
        # Constructor config is used as self._config; effective_config
        # should be ctor_config since _effective_config prefers it.
        max_nodes, _, _ = runner._resolve_orchestrator_settings(None, effective_config=ctor_config)
        assert max_nodes == 8


class TestEffectiveConfig:
    """Test _effective_config merging logic (Bug 2 fix)."""

    def test_no_inline_returns_constructor_config(self) -> None:
        config = HexDAGConfig()
        runner = PipelineRunner(config=config)
        # Builder has no inline config by default
        assert runner._effective_config() is config

    def test_no_constructor_no_inline_returns_none(self) -> None:
        runner = PipelineRunner()
        assert runner._effective_config() is None

    def test_inline_config_used_when_no_constructor_config(self) -> None:
        runner = PipelineRunner()
        # Manually set inline config on builder (simulating multi-doc parse)
        inline = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=3),
        )
        runner._builder._inline_config = inline
        assert runner._effective_config() is inline

    def test_constructor_config_wins_over_inline(self) -> None:
        ctor_config = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=8),
        )
        runner = PipelineRunner(config=ctor_config)
        inline = HexDAGConfig(
            orchestrator=OrchestratorConfig(max_concurrent_nodes=3),
        )
        runner._builder._inline_config = inline
        assert runner._effective_config() is ctor_config


class TestResolveField:
    """Test _resolve_field helper (Bug 3 fix — limits/caps resolution)."""

    def test_pipeline_value_wins(self) -> None:
        pipeline_limits = DefaultLimits(max_llm_calls=50)
        config_limits = DefaultLimits(max_llm_calls=100)
        result = PipelineRunner._resolve_field(pipeline_limits, config_limits)
        assert result is pipeline_limits

    def test_config_value_used_when_pipeline_none(self) -> None:
        config_limits = DefaultLimits(max_llm_calls=100)
        result = PipelineRunner._resolve_field(None, config_limits)
        assert result is config_limits

    def test_both_none_returns_none(self) -> None:
        assert PipelineRunner._resolve_field(None, None) is None

    def test_caps_resolution(self) -> None:
        config_caps = DefaultCaps(deny=["secret"])
        result = PipelineRunner._resolve_field(None, config_caps)
        assert result is config_caps
        assert result.deny == ["secret"]


class TestKindConfigLoggingWiring:
    """Test that kind: Config logging section is applied by PipelineRunner."""

    @pytest.mark.asyncio
    async def test_logging_config_applied(self, tmp_path: Path) -> None:
        """kind: Config with logging section should call configure_logging."""
        from unittest.mock import patch

        log_config = LoggingConfig(level="DEBUG", format="json")
        config = HexDAGConfig(logging=log_config)
        runner = PipelineRunner(config=config)

        with patch("hexdag.kernel.pipeline_runner.configure_logging") as mock_configure:
            await runner.run_from_string(SIMPLE_YAML)

            mock_configure.assert_called_once_with(
                level="DEBUG",
                format="json",
                output_file=None,
                use_color=True,
                include_timestamp=True,
                use_rich=False,
                dual_sink=False,
                enable_stdlib_bridge=False,
                backtrace=True,
                diagnose=True,
            )

    @pytest.mark.asyncio
    async def test_logging_config_with_output_file(self, tmp_path: Path) -> None:
        """kind: Config with output_file should pass it to configure_logging."""
        from unittest.mock import patch

        log_file = str(tmp_path / "hexdag.log")
        log_config = LoggingConfig(
            level="INFO",
            format="structured",
            output_file=log_file,
        )
        config = HexDAGConfig(logging=log_config)
        runner = PipelineRunner(config=config)

        with patch("hexdag.kernel.pipeline_runner.configure_logging") as mock_configure:
            await runner.run_from_string(SIMPLE_YAML)

            mock_configure.assert_called_once()
            call_kwargs = mock_configure.call_args
            assert call_kwargs.kwargs["output_file"] == log_file
            assert call_kwargs.kwargs["level"] == "INFO"
            assert call_kwargs.kwargs["format"] == "structured"

    @pytest.mark.asyncio
    async def test_default_config_applies_default_logging(self) -> None:
        """HexDAGConfig() has default LoggingConfig — should still call configure_logging."""
        from unittest.mock import patch

        config = HexDAGConfig()  # Has default LoggingConfig
        runner = PipelineRunner(config=config)

        with patch("hexdag.kernel.pipeline_runner.configure_logging") as mock_configure:
            await runner.run_from_string(SIMPLE_YAML)
            # Default LoggingConfig always exists, so configure_logging is called
            mock_configure.assert_called_once()
            assert mock_configure.call_args.kwargs["level"] == "INFO"
            assert mock_configure.call_args.kwargs["format"] == "structured"

    @pytest.mark.asyncio
    async def test_no_config_skips_configure(self) -> None:
        """No config at all (effective_config is None) should not call configure_logging."""
        from unittest.mock import patch

        runner = PipelineRunner()  # No config

        with patch("hexdag.kernel.pipeline_runner.configure_logging") as mock_configure:
            await runner.run_from_string(SIMPLE_YAML)
            mock_configure.assert_not_called()
