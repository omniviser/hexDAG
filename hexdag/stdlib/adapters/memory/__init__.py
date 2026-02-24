"""Memory adapter implementations for HexDAG.

This module provides various Memory port implementations:
- InMemoryMemory: Fast in-memory dictionary storage
- FileMemoryAdapter: File-based persistent storage
- SQLiteMemoryAdapter: SQLite-backed persistent storage

And memory plugins with structure-specific operations:
- SessionMemoryPlugin: Conversation history and context
- StateMemoryPlugin: Structured entities, relationships, and beliefs

Schemas for memory plugins:
- ConversationHistory: Session conversation schema
- EntityState: Entity and relationship schema
- BeliefState: Hinton-style belief probability schema
- EventLog: Reasoning trace schema
- ReasoningStep: Individual reasoning step schema
"""

from .collection_memory import InMemoryCollectionStorage
from .file_memory_adapter import FileMemoryAdapter
from .in_memory_memory import InMemoryMemory
from .schemas import (
    BeliefState,
    ConversationHistory,
    EntityState,
    EventLog,
    ReasoningStep,
)
from .session_memory import SessionMemoryPlugin
from .sqlite_memory_adapter import SQLiteMemoryAdapter
from .state_memory import StateMemoryPlugin

__all__ = [
    # Base memory adapters
    "InMemoryCollectionStorage",
    "InMemoryMemory",
    "FileMemoryAdapter",
    "SQLiteMemoryAdapter",
    # Memory plugins
    "SessionMemoryPlugin",
    "StateMemoryPlugin",
    # Schemas
    "BeliefState",
    "ConversationHistory",
    "EntityState",
    "EventLog",
    "ReasoningStep",
]
