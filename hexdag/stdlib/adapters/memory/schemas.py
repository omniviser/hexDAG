"""Pydantic schemas for memory plugins."""

import time
from typing import Any

from pydantic import BaseModel, Field


class ConversationHistory(BaseModel):
    """Schema for conversation history in SessionMemoryPlugin."""

    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityState(BaseModel):
    """Schema for entity state in StateMemoryPlugin."""

    entities: dict[str, dict[str, Any]] = Field(default_factory=dict)
    relationships: list[tuple[str, str, str]] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeliefState(BaseModel):
    """Hinton-style belief state for StateMemoryPlugin.

    Represents probability distribution over hypotheses with confidence scores.
    """

    beliefs: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)


class ReasoningStep(BaseModel):
    """Schema for reasoning step in EventMemoryPlugin."""

    step_num: int
    thought: str
    tool_used: str | None = None
    tool_result: Any = None
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventLog(BaseModel):
    """Schema for event log in EventMemoryPlugin."""

    agent_id: str
    events: list[ReasoningStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
