"""Tests for ProcRunsProvider (/proc/runs/ namespace)."""

from __future__ import annotations

import json

import pytest

from hexdag.drivers.vfs.providers.proc_runs_provider import ProcRunsProvider
from hexdag.kernel.domain.pipeline_run import PipelineRun, RunStatus
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError
from hexdag.stdlib.lib.process_registry import ProcessRegistry


def _make_registry(*runs: PipelineRun) -> ProcessRegistry:
    """Create a ProcessRegistry pre-populated with runs."""
    reg = ProcessRegistry()
    for run in runs:
        reg.register(run)
    return reg


class TestReaddir:
    @pytest.mark.asyncio()
    async def test_empty_registry(self) -> None:
        provider = ProcRunsProvider(_make_registry())
        entries = await provider.readdir("")
        assert entries == []

    @pytest.mark.asyncio()
    async def test_lists_all_runs(self) -> None:
        reg = _make_registry(
            PipelineRun(run_id="r1", pipeline_name="p1"),
            PipelineRun(run_id="r2", pipeline_name="p2"),
        )
        provider = ProcRunsProvider(reg)
        entries = await provider.readdir("")
        names = [e.name for e in entries]
        assert "r1" in names
        assert "r2" in names
        for entry in entries:
            assert entry.entry_type == EntryType.FILE

    @pytest.mark.asyncio()
    async def test_specific_run_is_not_directory(self) -> None:
        reg = _make_registry(PipelineRun(run_id="r1", pipeline_name="p1"))
        provider = ProcRunsProvider(reg)
        with pytest.raises(VFSError, match="not a directory"):
            await provider.readdir("r1")


class TestRead:
    @pytest.mark.asyncio()
    async def test_read_run(self) -> None:
        reg = _make_registry(PipelineRun(run_id="r1", pipeline_name="p1", status=RunStatus.RUNNING))
        provider = ProcRunsProvider(reg)
        result = await provider.read("r1")
        data = json.loads(result)
        assert data["run_id"] == "r1"
        assert data["pipeline_name"] == "p1"

    @pytest.mark.asyncio()
    async def test_read_run_status(self) -> None:
        reg = _make_registry(
            PipelineRun(run_id="r1", pipeline_name="p1", status=RunStatus.COMPLETED)
        )
        provider = ProcRunsProvider(reg)
        result = await provider.read("r1/status")
        status = json.loads(result)
        assert status == "completed"

    @pytest.mark.asyncio()
    async def test_read_unknown_run_raises(self) -> None:
        provider = ProcRunsProvider(_make_registry())
        with pytest.raises(VFSError, match="not found"):
            await provider.read("nonexistent")

    @pytest.mark.asyncio()
    async def test_read_directory_raises(self) -> None:
        provider = ProcRunsProvider(_make_registry())
        with pytest.raises(VFSError, match="cannot read directory"):
            await provider.read("")


class TestStat:
    @pytest.mark.asyncio()
    async def test_stat_root(self) -> None:
        reg = _make_registry(
            PipelineRun(run_id="r1", pipeline_name="p1"),
            PipelineRun(run_id="r2", pipeline_name="p2"),
        )
        provider = ProcRunsProvider(reg)
        stat = await provider.stat("")
        assert stat.path == "/proc/runs"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count == 2

    @pytest.mark.asyncio()
    async def test_stat_specific_run(self) -> None:
        reg = _make_registry(
            PipelineRun(
                run_id="r1",
                pipeline_name="order-processing",
                status=RunStatus.RUNNING,
                ref_id="ORD-001",
            )
        )
        provider = ProcRunsProvider(reg)
        stat = await provider.stat("r1")
        assert stat.path == "/proc/runs/r1"
        assert stat.entry_type == EntryType.FILE
        assert stat.entity_type == "run"
        assert stat.status == "running"
        assert "order-processing" in (stat.description or "")
        assert stat.tags["pipeline_name"] == "order-processing"
        assert stat.tags["ref_id"] == "ORD-001"

    @pytest.mark.asyncio()
    async def test_stat_unknown_run_raises(self) -> None:
        provider = ProcRunsProvider(_make_registry())
        with pytest.raises(VFSError, match="not found"):
            await provider.stat("nonexistent")
