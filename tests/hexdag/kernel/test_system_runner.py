"""Tests for SystemRunner — kind: System execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hexdag.kernel.domain.system_config import SystemConfig
from hexdag.kernel.system_runner import SystemRunner, _resolve_template

# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    def test_simple_reference(self) -> None:
        results = {"extract": {"records": [1, 2, 3]}}
        value = _resolve_template("{{ extract.records }}", results)
        assert value == [1, 2, 3]

    def test_preserves_type(self) -> None:
        results = {"proc": {"count": 42}}
        value = _resolve_template("{{ proc.count }}", results)
        assert value == 42
        assert isinstance(value, int)

    def test_missing_process_returns_none(self) -> None:
        value = _resolve_template("{{ missing.field }}", {})
        assert value is None

    def test_missing_field_returns_none(self) -> None:
        results = {"extract": {"other": "value"}}
        value = _resolve_template("{{ extract.missing }}", results)
        assert value is None

    def test_string_interpolation(self) -> None:
        results = {"a": {"x": "hello"}, "b": {"y": "world"}}
        value = _resolve_template("got {{ a.x }} and {{ b.y }}", results)
        assert value == "got hello and world"


# ---------------------------------------------------------------------------
# SystemRunner
# ---------------------------------------------------------------------------


def _make_system_config(
    processes: list[dict[str, Any]],
    pipes: list[dict[str, Any]] | None = None,
) -> SystemConfig:
    return SystemConfig.model_validate({
        "metadata": {"name": "test-system"},
        "processes": processes,
        "pipes": pipes or [],
    })


class TestSystemRunner:
    @pytest.mark.asyncio
    async def test_linear_chain(self, tmp_path: Path) -> None:
        """Test a -> b -> c linear pipeline chain."""
        # Create minimal pipeline YAML files
        for name in ("a", "b", "c"):
            (tmp_path / f"{name}.yaml").write_text(
                f"apiVersion: hexdag/v1\nkind: Pipeline\n"
                f"metadata:\n  name: {name}\nspec:\n  nodes: []\n"
            )

        config = _make_system_config(
            processes=[
                {"name": "a", "pipeline": "a.yaml"},
                {"name": "b", "pipeline": "b.yaml"},
                {"name": "c", "pipeline": "c.yaml"},
            ],
            pipes=[
                {"from": "a", "to": "b", "mapping": {"x": "{{ a.result }}"}},
                {"from": "b", "to": "c", "mapping": {"y": "{{ b.output }}"}},
            ],
        )

        # Mock PipelineRunner.run to track calls
        call_log: list[tuple[str, Any]] = []

        async def mock_run(
            pipeline_path: str | Path, input_data: Any = None, **kw: Any
        ) -> dict[str, Any]:
            name = Path(pipeline_path).stem
            call_log.append((name, input_data))
            return {"result": f"{name}_result", "output": f"{name}_output"}

        runner = SystemRunner(base_path=tmp_path)

        with patch("hexdag.kernel.system_runner.PipelineRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run = AsyncMock(side_effect=mock_run)

            results = await runner.run(config, input_data={"initial": True})

        # Verify execution order
        assert [name for name, _ in call_log] == ["a", "b", "c"]

        # Verify pipe mapping resolution
        _, a_input = call_log[0]
        assert a_input == {"initial": True}  # root gets initial input

        _, b_input = call_log[1]
        assert b_input == {"x": "a_result"}  # resolved from {{ a.result }}

        _, c_input = call_log[2]
        assert c_input == {"y": "b_output"}  # resolved from {{ b.output }}

        # Verify results
        assert "a" in results
        assert "b" in results
        assert "c" in results

    @pytest.mark.asyncio
    async def test_parallel_processes_no_pipes(self, tmp_path: Path) -> None:
        """Processes with no pipes all get the initial input."""
        for name in ("x", "y"):
            (tmp_path / f"{name}.yaml").write_text(
                f"apiVersion: hexdag/v1\nkind: Pipeline\n"
                f"metadata:\n  name: {name}\nspec:\n  nodes: []\n"
            )

        config = _make_system_config(
            processes=[
                {"name": "x", "pipeline": "x.yaml"},
                {"name": "y", "pipeline": "y.yaml"},
            ],
        )

        call_log: list[tuple[str, Any]] = []

        async def mock_run(
            pipeline_path: str | Path, input_data: Any = None, **kw: Any
        ) -> dict[str, Any]:
            name = Path(pipeline_path).stem
            call_log.append((name, input_data))
            return {}

        runner = SystemRunner(base_path=tmp_path)

        with patch("hexdag.kernel.system_runner.PipelineRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run = AsyncMock(side_effect=mock_run)

            await runner.run(config, input_data={"shared": True})

        # Both should receive initial input
        for _, inp in call_log:
            assert inp == {"shared": True}

    @pytest.mark.asyncio
    async def test_process_failure_emits_events(self, tmp_path: Path) -> None:
        """Failing process emits failed events."""
        (tmp_path / "fail.yaml").write_text(
            "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: fail\nspec:\n  nodes: []\n"
        )

        config = _make_system_config(
            processes=[{"name": "fail", "pipeline": "fail.yaml"}],
        )

        observer = AsyncMock()
        runner = SystemRunner(base_path=tmp_path, observer_manager=observer)

        with patch("hexdag.kernel.system_runner.PipelineRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run = AsyncMock(side_effect=RuntimeError("boom"))

            with pytest.raises(RuntimeError, match="boom"):
                await runner.run(config)

        # Check events emitted
        events = [call.args[0] for call in observer.notify.call_args_list]
        event_types = [type(e).__name__ for e in events]

        assert "SystemStarted" in event_types
        assert "ProcessStarted" in event_types
        assert "ProcessCompleted" in event_types
        assert "SystemCompleted" in event_types

        # The ProcessCompleted should have status=failed
        proc_completed = next(e for e in events if type(e).__name__ == "ProcessCompleted")
        assert proc_completed.status == "failed"
        assert "boom" in proc_completed.error

        # The SystemCompleted should have status=failed
        sys_completed = next(e for e in events if type(e).__name__ == "SystemCompleted")
        assert sys_completed.status == "failed"

    @pytest.mark.asyncio
    async def test_events_emitted_on_success(self, tmp_path: Path) -> None:
        """Successful run emits SystemStarted, ProcessStarted/Completed, SystemCompleted."""
        (tmp_path / "ok.yaml").write_text(
            "apiVersion: hexdag/v1\nkind: Pipeline\nmetadata:\n  name: ok\nspec:\n  nodes: []\n"
        )

        config = _make_system_config(
            processes=[{"name": "ok", "pipeline": "ok.yaml"}],
        )

        observer = AsyncMock()
        runner = SystemRunner(base_path=tmp_path, observer_manager=observer)

        with patch("hexdag.kernel.system_runner.PipelineRunner") as MockRunner:
            instance = MockRunner.return_value
            instance.run = AsyncMock(return_value={"node1": "result"})

            results = await runner.run(config)

        assert results == {"ok": {"node1": "result"}}

        events = [
            type(e).__name__ for e in (call.args[0] for call in observer.notify.call_args_list)
        ]
        assert events == [
            "SystemStarted",
            "ProcessStarted",
            "ProcessCompleted",
            "SystemCompleted",
        ]
