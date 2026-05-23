"""Tests for System — unified runtime API for kind: System manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from hexdag.kernel.domain.system_config import SystemConfig
from hexdag.kernel.system import System, SystemError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dag_config(**overrides: Any) -> SystemConfig:
    """Create a DAG-mode system config."""
    base: dict[str, Any] = {
        "metadata": {"name": "test-dag"},
        "processes": [
            {"name": "extract", "pipeline": "extract.yaml"},
            {"name": "load", "pipeline": "load.yaml"},
        ],
        "pipes": [
            {
                "from": "extract",
                "to": "load",
                "mapping": {"records": "{{ extract.records }}"},
            },
        ],
    }
    base.update(overrides)
    return SystemConfig.model_validate(base)


def _lifecycle_config(**overrides: Any) -> SystemConfig:
    """Create a lifecycle-mode system config."""
    base: dict[str, Any] = {
        "metadata": {"name": "test-lifecycle"},
        "processes": [
            {"name": "investigate", "pipeline": "investigate.yaml"},
        ],
        "state_machines": {
            "ticket": {
                "initial": "OPEN",
                "transitions": {
                    "OPEN": ["INVESTIGATING", "CLOSED"],
                    "INVESTIGATING": ["RESOLVED"],
                    "RESOLVED": ["CLOSED"],
                },
            },
        },
        "states": {
            "INVESTIGATING": {"on_enter": "investigate"},
            "CLOSED": {"terminal": True},
        },
    }
    base.update(overrides)
    return SystemConfig.model_validate(base)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSystemConstruction:
    def test_dag_mode(self) -> None:
        system = System(_dag_config())
        assert not system.is_lifecycle
        assert system.process_names == ["extract", "load"]

    def test_lifecycle_mode(self) -> None:
        system = System(_lifecycle_config())
        assert system.is_lifecycle
        assert "investigate" in system.process_names

    def test_config_property(self) -> None:
        config = _dag_config()
        system = System(config)
        assert system.config is config


class TestSystemFromYaml:
    def test_from_yaml_file(self, tmp_path: Path) -> None:
        for name in ("extract", "load"):
            (tmp_path / f"{name}.yaml").write_text(
                f"apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: {name}\n"
                "spec:\n  nodes: []\n"
            )
        system_yaml = tmp_path / "system.yaml"
        system_yaml.write_text(
            "kind: System\n"
            "metadata:\n  name: test\n"
            "spec:\n"
            "  processes:\n"
            "    - name: extract\n"
            "      pipeline: extract.yaml\n"
            "    - name: load\n"
            "      pipeline: load.yaml\n"
            "  pipes:\n"
            "    - from: extract\n"
            "      to: load\n"
            "      mapping:\n"
            '        records: "{{ extract.records }}"\n'
        )
        system = System.from_yaml(system_yaml)
        assert system.config.metadata["name"] == "test"
        assert not system.is_lifecycle

    def test_from_yaml_string(self) -> None:
        yaml_content = """\
kind: System
metadata:
  name: inline
spec:
  processes:
    - name: step1
      pipeline: step1.yaml
  pipes: []
"""
        # Skip pipeline path validation by providing base_path with dummy files
        with patch("hexdag.compiler.system_builder.SystemBuilder._validate_pipeline_paths"):
            system = System.from_yaml_string(yaml_content)
        assert system.config.metadata["name"] == "inline"


# ---------------------------------------------------------------------------
# DAG mode
# ---------------------------------------------------------------------------


class TestSystemDAGMode:
    @pytest.mark.asyncio
    async def test_run_delegates_to_system_runner(self) -> None:
        system = System(_dag_config())

        with patch.object(system._system_runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"extract": {"records": [1]}, "load": {"ok": True}}
            results = await system.run({"input": "data"})

        mock_run.assert_called_once_with(system._config, input_data={"input": "data"})
        assert "extract" in results
        assert "load" in results

    @pytest.mark.asyncio
    async def test_transition_raises_on_dag_system(self) -> None:
        system = System(_dag_config())
        with pytest.raises(SystemError, match="lifecycle-mode"):
            await system.transition("ticket", "T-1", "INVESTIGATING")


# ---------------------------------------------------------------------------
# Lifecycle mode
# ---------------------------------------------------------------------------


class TestSystemLifecycleMode:
    @pytest.mark.asyncio
    async def test_transition_auto_starts(self) -> None:
        system = System(_lifecycle_config())
        assert system._lifecycle_runner is not None

        # Mock the lifecycle runner methods
        system._lifecycle_runner.start = AsyncMock()  # type: ignore[method-assign]
        system._lifecycle_runner.transition = AsyncMock(  # type: ignore[method-assign]
            return_value={"from_state": "OPEN", "to_state": "INVESTIGATING"}
        )

        result = await system.transition("ticket", "T-1", "INVESTIGATING")

        system._lifecycle_runner.start.assert_called_once_with(system._config)
        system._lifecycle_runner.transition.assert_called_once()
        assert result["to_state"] == "INVESTIGATING"

    @pytest.mark.asyncio
    async def test_transition_starts_only_once(self) -> None:
        system = System(_lifecycle_config())
        assert system._lifecycle_runner is not None

        system._lifecycle_runner.start = AsyncMock()  # type: ignore[method-assign]
        system._lifecycle_runner.transition = AsyncMock(  # type: ignore[method-assign]
            return_value={"from_state": "OPEN", "to_state": "INVESTIGATING"}
        )

        await system.transition("ticket", "T-1", "INVESTIGATING")
        await system.transition("ticket", "T-1", "RESOLVED")

        # start() called only on first transition
        system._lifecycle_runner.start.assert_called_once()
        assert system._lifecycle_runner.transition.call_count == 2

    @pytest.mark.asyncio
    async def test_run_raises_on_lifecycle_system(self) -> None:
        system = System(_lifecycle_config())
        with pytest.raises(SystemError, match="DAG-mode"):
            await system.run({"input": "data"})


# ---------------------------------------------------------------------------
# run_process
# ---------------------------------------------------------------------------


class TestSystemRunProcess:
    @pytest.mark.asyncio
    async def test_run_process_works(self) -> None:
        system = System(_dag_config())

        with patch("hexdag.kernel.system.PipelineRunner") as MockRunner:
            mock_instance = MockRunner.return_value
            mock_result = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)

            result = await system.run_process("extract", {"key": "val"})

        assert result is mock_result
        mock_instance.run.assert_called_once_with(
            pipeline_path="extract.yaml",
            input_data={"key": "val"},
        )

    @pytest.mark.asyncio
    async def test_run_process_unknown_raises(self) -> None:
        system = System(_dag_config())
        with pytest.raises(SystemError, match="Unknown process 'nonexistent'"):
            await system.run_process("nonexistent")

    @pytest.mark.asyncio
    async def test_run_process_works_in_lifecycle_mode(self) -> None:
        system = System(_lifecycle_config())

        with patch("hexdag.kernel.system.PipelineRunner") as MockRunner:
            mock_instance = MockRunner.return_value
            mock_result = MagicMock()
            mock_instance.run = AsyncMock(return_value=mock_result)

            result = await system.run_process("investigate")

        assert result is mock_result


# ---------------------------------------------------------------------------
# Port sharing
# ---------------------------------------------------------------------------


class TestSystemPortSharing:
    @pytest.mark.asyncio
    async def test_same_port_instance_across_runs(self) -> None:
        """Port overrides should be shared across pipeline runs."""
        mock_llm = MagicMock()
        system = System(_dag_config(), port_overrides={"llm": mock_llm})

        # Verify the shared port instance
        assert system.ports["llm"] is mock_llm

        # Verify PipelineRunner receives the shared port
        with patch("hexdag.kernel.system.PipelineRunner") as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.run = AsyncMock(return_value=MagicMock())

            await system.run_process("extract")
            await system.run_process("load")

        # Both calls should create PipelineRunner with the same port_overrides
        assert MockRunner.call_count == 2
        for call in MockRunner.call_args_list:
            assert call.kwargs["port_overrides"]["llm"] is mock_llm

    def test_port_overrides_merge_with_config_ports(self) -> None:
        """User port_overrides should take precedence over config ports."""
        mock_llm = MagicMock()

        with patch.object(System, "_instantiate_system_ports", return_value={"llm": mock_llm}):
            config = _dag_config(ports={"llm": {"adapter": "some.Adapter"}})
            system = System(config, port_overrides={"llm": mock_llm})

        assert system.ports["llm"] is mock_llm


# ---------------------------------------------------------------------------
# Close / context manager
# ---------------------------------------------------------------------------


class TestSystemClose:
    @pytest.mark.asyncio
    async def test_close_stops_lifecycle_runner(self) -> None:
        system = System(_lifecycle_config())
        assert system._lifecycle_runner is not None

        system._lifecycle_runner.start = AsyncMock()  # type: ignore[method-assign]
        system._lifecycle_runner.transition = AsyncMock(  # type: ignore[method-assign]
            return_value={"from_state": "OPEN", "to_state": "INVESTIGATING"}
        )
        system._lifecycle_runner.stop = AsyncMock()  # type: ignore[method-assign]

        # Start by transitioning
        await system.transition("ticket", "T-1", "INVESTIGATING")
        await system.close()

        system._lifecycle_runner.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_calls_aclose_on_ports(self) -> None:
        mock_port = AsyncMock()
        mock_port.aclose = AsyncMock()

        system = System(_dag_config(), port_overrides={"llm": mock_port})
        await system.close()

        mock_port.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_ports_without_aclose(self) -> None:
        """Ports without aclose/close should not raise."""
        plain_port = object()
        system = System(_dag_config(), port_overrides={"data": plain_port})
        await system.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with System(_dag_config()) as system:
            assert system.process_names == ["extract", "load"]
        # close() was called — no assertion needed, just no exception
