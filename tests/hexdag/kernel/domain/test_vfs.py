"""Tests for VFS domain models."""

from __future__ import annotations

from hexdag.kernel.domain.vfs import DirEntry, EntryType, StatResult


class TestEntryType:
    def test_file_value(self) -> None:
        assert EntryType.FILE == "file"

    def test_directory_value(self) -> None:
        assert EntryType.DIRECTORY == "directory"


class TestDirEntry:
    def test_create_file_entry(self) -> None:
        entry = DirEntry(name="llm_node", entry_type=EntryType.FILE, path="/lib/nodes/llm_node")
        assert entry.name == "llm_node"
        assert entry.entry_type == EntryType.FILE
        assert entry.path == "/lib/nodes/llm_node"

    def test_create_directory_entry(self) -> None:
        entry = DirEntry(name="nodes", entry_type=EntryType.DIRECTORY, path="/lib/nodes")
        assert entry.entry_type == EntryType.DIRECTORY

    def test_model_dump(self) -> None:
        entry = DirEntry(name="test", entry_type=EntryType.FILE, path="/test")
        dumped = entry.model_dump()
        assert dumped == {"name": "test", "entry_type": "file", "path": "/test"}

    def test_model_json_roundtrip(self) -> None:
        entry = DirEntry(name="test", entry_type=EntryType.FILE, path="/test")
        json_str = entry.model_dump_json()
        restored = DirEntry.model_validate_json(json_str)
        assert restored == entry


class TestStatResult:
    def test_create_minimal(self) -> None:
        stat = StatResult(path="/lib", entry_type=EntryType.DIRECTORY)
        assert stat.path == "/lib"
        assert stat.entry_type == EntryType.DIRECTORY
        assert stat.entity_type is None
        assert stat.description is None
        assert stat.module_path is None
        assert stat.status is None
        assert stat.child_count is None
        assert stat.capabilities == []
        assert stat.tags == {}

    def test_create_full(self) -> None:
        stat = StatResult(
            path="/lib/nodes/llm_node",
            entry_type=EntryType.FILE,
            entity_type="node",
            description="Language model interaction node",
            module_path="hexdag.stdlib.nodes.LLMNode",
            status=None,
            child_count=None,
            capabilities=["read"],
            tags={"is_builtin": "true"},
        )
        assert stat.entity_type == "node"
        assert stat.description == "Language model interaction node"
        assert stat.module_path == "hexdag.stdlib.nodes.LLMNode"
        assert stat.capabilities == ["read"]
        assert stat.tags == {"is_builtin": "true"}

    def test_create_run_stat(self) -> None:
        stat = StatResult(
            path="/proc/runs/abc123",
            entry_type=EntryType.FILE,
            entity_type="run",
            description="Pipeline 'order-processing' run",
            status="running",
            tags={"pipeline_name": "order-processing", "ref_id": "ORD-001"},
        )
        assert stat.status == "running"
        assert stat.tags["pipeline_name"] == "order-processing"

    def test_model_dump_exclude_none(self) -> None:
        stat = StatResult(
            path="/lib/nodes/llm_node",
            entry_type=EntryType.FILE,
            description="Language model interaction node",
        )
        dumped = stat.model_dump(exclude_none=True)
        assert "entity_type" not in dumped
        assert "module_path" not in dumped
        assert "status" not in dumped
        assert "child_count" not in dumped
        assert dumped["path"] == "/lib/nodes/llm_node"
        assert dumped["description"] == "Language model interaction node"

    def test_model_dump_includes_empty_defaults(self) -> None:
        stat = StatResult(path="/lib", entry_type=EntryType.DIRECTORY)
        dumped = stat.model_dump(exclude_none=True)
        # capabilities and tags are empty but not None, so they're included
        assert dumped["capabilities"] == []
        assert dumped["tags"] == {}
