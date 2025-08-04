"""Base classes for the pipeline event system."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Types of pipeline events."""

    VALIDATION_WARNING = "validation_warning"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_BUILD_STARTED = "pipeline_build_started"
    WAVE_STARTED = "wave_started"
    WAVE_COMPLETED = "wave_completed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    LLM_PROMPT_GENERATED = "llm_prompt_generated"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"


class PipelineEvent(ABC):
    """Base class for all pipeline events."""

    def __init__(self) -> None:
        self.event_type: EventType
        self.timestamp: datetime = datetime.now()
        self.session_id: str = ""
        self.metadata: dict[str, Any] = {}

    def __post_init__(self) -> None:
        """Initialize timestamp and metadata if not provided."""
        if not hasattr(self, "timestamp") or self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata or {},
            **self._extra_fields(),
        }

    @abstractmethod
    def _extra_fields(self) -> dict[str, Any]:
        """Override to add event-specific fields to dictionary."""
        return {}


class Observer(ABC):
    """Base observer interface."""

    @abstractmethod
    async def handle(self, event: PipelineEvent) -> None:
        """Handle pipeline event asynchronously."""
        pass

    def can_handle(self, event: PipelineEvent) -> bool:
        """Check if this observer can handle the event."""
        return True


class SyncObserver(Observer):
    """Base for synchronous observers."""

    async def handle(self, event: PipelineEvent) -> None:
        """Handle event by delegating to sync method."""
        await asyncio.to_thread(self.handle_sync, event)

    @abstractmethod
    def handle_sync(self, event: PipelineEvent) -> None:
        """Handle event synchronously."""
        pass
