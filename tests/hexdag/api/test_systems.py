"""Tests for hexdag.api.systems — System management API."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from hexdag.api import systems


def _mock_system(*, is_lifecycle: bool = False) -> MagicMock:
    """Create a mock System with standard stubs."""
    system = MagicMock()
    system.config.processes = [
        MagicMock(name="extract", pipeline="extract.yaml", input_schema=None, output_schema=None),
        MagicMock(name="load", pipeline="load.yaml", input_schema=None, output_schema=None),
    ]
    # Fix .name on MagicMock (by default it returns another mock)
    system.config.processes[0].name = "extract"
    system.config.processes[0].pipeline = "extract.yaml"
    system.config.processes[0].input_schema = None
    system.config.processes[0].output_schema = None
    system.config.processes[1].name = "load"
    system.config.processes[1].pipeline = "load.yaml"
    system.config.processes[1].input_schema = None
    system.config.processes[1].output_schema = None

    system.run = AsyncMock(return_value={"extract": {"records": [1]}, "load": {"ok": True}})
    system.run_process = AsyncMock(return_value=MagicMock())
    system.transition = AsyncMock(return_value={"from_state": "OPEN", "to_state": "INVESTIGATING"})

    type(system).entity_state = PropertyMock(return_value=MagicMock() if is_lifecycle else None)
    return system


class TestRunSystem:
    @pytest.mark.asyncio
    async def test_delegates_to_system_run(self) -> None:
        system = _mock_system()
        result = await systems.run_system(system, {"key": "val"})

        system.run.assert_called_once_with({"key": "val"})
        assert "extract" in result

    @pytest.mark.asyncio
    async def test_passes_none_input(self) -> None:
        system = _mock_system()
        await systems.run_system(system)

        system.run.assert_called_once_with(None)


class TestRunProcess:
    @pytest.mark.asyncio
    async def test_delegates_to_system_run_process(self) -> None:
        system = _mock_system()
        await systems.run_process(system, "extract", {"input": "data"})

        system.run_process.assert_called_once_with("extract", {"input": "data"})


class TestTransitionEntity:
    @pytest.mark.asyncio
    async def test_delegates_to_system_transition(self) -> None:
        system = _mock_system(is_lifecycle=True)
        result = await systems.transition_entity(
            system,
            "ticket",
            "T-1",
            "INVESTIGATING",
            reason="test",
        )

        system.transition.assert_called_once_with(
            "ticket",
            "T-1",
            "INVESTIGATING",
            reason="test",
            payload=None,
        )
        assert result["to_state"] == "INVESTIGATING"


class TestListProcesses:
    def test_returns_process_info(self) -> None:
        system = _mock_system()
        result = systems.list_processes(system)

        assert len(result) == 2
        assert result[0]["name"] == "extract"
        assert result[0]["pipeline"] == "extract.yaml"
        assert result[1]["name"] == "load"


class TestGetEntityState:
    @pytest.mark.asyncio
    async def test_returns_none_for_dag_mode(self) -> None:
        system = _mock_system(is_lifecycle=False)
        result = await systems.get_entity_state(system, "ticket", "T-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delegates_to_entity_state(self) -> None:
        system = _mock_system(is_lifecycle=True)
        mock_state: dict[str, Any] = {
            "entity_type": "ticket",
            "entity_id": "T-1",
            "state": "OPEN",
        }
        system.entity_state.aget_state = AsyncMock(return_value=mock_state)

        result = await systems.get_entity_state(system, "ticket", "T-1")

        system.entity_state.aget_state.assert_called_once_with("ticket", "T-1")
        assert result == mock_state
