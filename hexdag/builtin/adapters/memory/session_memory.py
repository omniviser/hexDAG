"""Session Memory Plugin for conversation history."""

import time
from typing import Any

from hexdag.builtin.adapters.memory.schemas import ConversationHistory
from hexdag.core.logging import get_logger
from hexdag.core.ports.memory import Memory

logger = get_logger(__name__)


class SessionMemoryPlugin:
    """Memory plugin for conversation history and session context.

    Wraps a base Memory adapter (InMemoryMemory, SQLiteMemoryAdapter, etc.)
    and provides domain-specific operations for managing conversation history.

    This plugin does NOT store data itself - it delegates to the underlying
    Memory port implementation and adds session-specific logic.

    Example
    -------
        from hexdag.builtin.adapters.memory import InMemoryMemory

        # Use existing storage adapter
        storage = InMemoryMemory()
        session_memory = SessionMemoryPlugin(storage=storage, max_messages=100)

        # Append messages
        await session_memory.append_message("session123", "user", "Hello!")
        await session_memory.append_message("session123", "assistant", "Hi there!")

        # Get recent history
        recent = await session_memory.get_recent_messages("session123", count=10)

        # Get full history
        history = await session_memory.get_history("session123")
    """

    plugin_type = "session"

    def __init__(self, storage: Memory, max_messages: int = 100):
        """Initialize session memory plugin.

        Args
        ----
            storage: Base Memory port implementation (InMemoryMemory, SQLiteMemoryAdapter, etc.)
            max_messages: Maximum messages to keep per session (auto-truncates)
        """
        self.storage = storage
        self.max_messages = max_messages

    async def aget(self, key: str) -> Any:
        """Get value from session scope.

        Delegates to underlying storage with session:: prefix.
        """
        return await self.storage.aget(f"session::{key}")

    async def aset(self, key: str, value: Any) -> None:
        """Set value in session scope.

        Delegates to underlying storage with session:: prefix.
        """
        await self.storage.aset(f"session::{key}", value)

    # Specialized methods for session memory

    async def get_history(self, session_id: str) -> ConversationHistory:
        """Get conversation history for session.

        Args
        ----
            session_id: Session identifier

        Returns
        -------
            ConversationHistory with messages and timestamps
        """
        data = await self.aget(session_id)
        if data is None:
            return ConversationHistory(
                session_id=session_id,
                messages=[],
                timestamps=[],
            )
        return ConversationHistory.model_validate(data)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append message to conversation history with auto-truncation.

        Args
        ----
            session_id: Session identifier
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional metadata for this message
        """
        history = await self.get_history(session_id)

        # Append new message
        history.messages.append({"role": role, "content": content})
        history.timestamps.append(time.time())

        # Update token count (rough estimate: 4 chars per token)
        history.token_count += len(content) // 4

        # Auto-truncate to max_messages
        if len(history.messages) > self.max_messages:
            overflow = len(history.messages) - self.max_messages
            history.messages = history.messages[overflow:]
            history.timestamps = history.timestamps[overflow:]
            logger.debug(
                "Truncated session %s: removed %d old messages",
                session_id,
                overflow,
            )

        # Update metadata if provided
        if metadata:
            history.metadata.update(metadata)

        await self.aset(session_id, history.model_dump())

    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> list[dict[str, str]]:
        """Get recent N messages from conversation.

        Args
        ----
            session_id: Session identifier
            count: Number of recent messages to return

        Returns
        -------
            List of recent message dicts
        """
        history = await self.get_history(session_id)
        return history.messages[-count:]

    async def clear_session(self, session_id: str) -> None:
        """Clear conversation history for session.

        Args
        ----
            session_id: Session identifier
        """
        empty_history = ConversationHistory(
            session_id=session_id,
            messages=[],
            timestamps=[],
        )
        await self.aset(session_id, empty_history.model_dump())
        logger.info("Cleared session %s", session_id)

    async def get_token_count(self, session_id: str) -> int:
        """Get approximate token count for session.

        Args
        ----
            session_id: Session identifier

        Returns
        -------
            Approximate token count
        """
        history = await self.get_history(session_id)
        return history.token_count
