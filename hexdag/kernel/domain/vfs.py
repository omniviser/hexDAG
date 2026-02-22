"""Domain models for the Virtual Filesystem (VFS).

These models represent the data structures returned by VFS operations.
Designed for agent/MCP consumption — fields are chosen to be useful for
AI agents making decisions, not for low-level filesystem metadata.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EntryType(StrEnum):
    """Type of a VFS entry."""

    FILE = "file"
    DIRECTORY = "directory"


class DirEntry(BaseModel):
    """A single entry in a VFS directory listing."""

    name: str
    entry_type: EntryType
    path: str


class StatResult(BaseModel):
    """Metadata about a VFS path.

    Fields are chosen to be useful for AI agents making decisions,
    not for low-level filesystem metadata like byte sizes.

    Attributes
    ----------
    path : str
        Absolute path in the VFS namespace.
    entry_type : EntryType
        Whether this is a file or directory.
    entity_type : str | None
        Semantic type: "node", "adapter", "run", "scheduled_task",
        "entity", "port", "pipeline", "tool", "macro", "tag", "lib".
    description : str | None
        Human-readable summary (from docstrings, config, etc.).
    module_path : str | None
        Python import path (for nodes, adapters, tools).
    status : str | None
        Runtime status (for runs: "running"/"completed", for scheduled: "active"/"paused").
    child_count : int | None
        Number of children (for directories).
    capabilities : list[str]
        What operations are available (e.g., ["read", "exec", "watch"]).
    tags : dict[str, str]
        Arbitrary key-value labels (port_type, is_builtin, etc.).
    """

    path: str
    entry_type: EntryType
    entity_type: str | None = None
    description: str | None = None
    module_path: str | None = None
    status: str | None = None
    child_count: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)


__all__ = ["DirEntry", "EntryType", "StatResult"]
