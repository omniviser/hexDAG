"""Tests for ProcScheduledProvider (/proc/scheduled/ namespace)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from hexdag.drivers.vfs.providers.proc_scheduled_provider import ProcScheduledProvider
from hexdag.kernel.domain.vfs import EntryType
from hexdag.kernel.exceptions import VFSError


def _make_scheduler(tasks: list[dict]) -> AsyncMock:
    """Create a mock Scheduler whose alist_scheduled returns *tasks*."""
    scheduler = AsyncMock()
    scheduler.alist_scheduled = AsyncMock(return_value=tasks)
    return scheduler


SAMPLE_TASKS = [
    {
        "task_id": "t1",
        "pipeline_name": "order-check",
        "status": "pending",
        "ref_id": "ORD-001",
        "delay_seconds": 60.0,
        "interval_seconds": None,
    },
    {
        "task_id": "t2",
        "pipeline_name": "daily-report",
        "status": "active",
        "ref_id": None,
        "delay_seconds": 0.0,
        "interval_seconds": 300.0,
    },
]


class TestReaddir:
    @pytest.mark.asyncio()
    async def test_empty_scheduler(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler([]))
        entries = await provider.readdir("")
        assert entries == []

    @pytest.mark.asyncio()
    async def test_lists_all_tasks(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        entries = await provider.readdir("")
        names = [e.name for e in entries]
        assert "t1" in names
        assert "t2" in names
        for entry in entries:
            assert entry.entry_type == EntryType.FILE
            assert entry.path.startswith("/proc/scheduled/")

    @pytest.mark.asyncio()
    async def test_specific_task_is_not_directory(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        with pytest.raises(VFSError, match="not a directory"):
            await provider.readdir("t1")


class TestRead:
    @pytest.mark.asyncio()
    async def test_read_task(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        result = await provider.read("t1")
        data = json.loads(result)
        assert data["task_id"] == "t1"
        assert data["pipeline_name"] == "order-check"

    @pytest.mark.asyncio()
    async def test_read_unknown_task_raises(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        with pytest.raises(VFSError, match="not found"):
            await provider.read("nonexistent")

    @pytest.mark.asyncio()
    async def test_read_directory_raises(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler([]))
        with pytest.raises(VFSError, match="cannot read directory"):
            await provider.read("")


class TestStat:
    @pytest.mark.asyncio()
    async def test_stat_root(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        stat = await provider.stat("")
        assert stat.path == "/proc/scheduled"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.child_count == 2

    @pytest.mark.asyncio()
    async def test_stat_specific_task(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        stat = await provider.stat("t1")
        assert stat.path == "/proc/scheduled/t1"
        assert stat.entry_type == EntryType.FILE
        assert stat.entity_type == "scheduled_task"
        assert stat.status == "pending"
        assert "order-check" in (stat.description or "")
        assert stat.tags["pipeline_name"] == "order-check"
        assert stat.tags["ref_id"] == "ORD-001"

    @pytest.mark.asyncio()
    async def test_stat_task_without_ref_id(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        stat = await provider.stat("t2")
        assert stat.path == "/proc/scheduled/t2"
        assert stat.tags["pipeline_name"] == "daily-report"
        assert "ref_id" not in stat.tags

    @pytest.mark.asyncio()
    async def test_stat_unknown_task_raises(self) -> None:
        provider = ProcScheduledProvider(_make_scheduler(SAMPLE_TASKS))
        with pytest.raises(VFSError, match="not found"):
            await provider.stat("nonexistent")
