"""Tests for SessionMemoryPlugin."""

import time

import pytest

from hexdag.stdlib.adapters.memory import InMemoryMemory, SessionMemoryPlugin
from hexdag.stdlib.adapters.memory.schemas import ConversationHistory


@pytest.fixture
def storage():
    """Create in-memory storage for testing."""
    return InMemoryMemory()


@pytest.fixture
def session_memory(storage):
    """Create SessionMemoryPlugin with in-memory storage."""
    return SessionMemoryPlugin(storage=storage, max_messages=10)


class TestSessionMemoryPlugin:
    """Test SessionMemoryPlugin functionality."""

    @pytest.mark.asyncio
    async def test_plugin_type(self, session_memory):
        """Test plugin type identifier."""
        assert session_memory.plugin_type == "session"

    @pytest.mark.asyncio
    async def test_get_nonexistent_history(self, session_memory):
        """Test getting history for non-existent session creates empty history."""
        history = await session_memory.get_history("session_123")

        assert isinstance(history, ConversationHistory)
        assert history.session_id == "session_123"
        assert len(history.messages) == 0
        assert history.token_count == 0

    @pytest.mark.asyncio
    async def test_append_single_message(self, session_memory):
        """Test appending a single message to session."""
        await session_memory.append_message(
            "session_123", role="user", content="Hello, how are you?"
        )

        history = await session_memory.get_history("session_123")

        assert len(history.messages) == 1
        assert history.messages[0]["role"] == "user"
        assert history.messages[0]["content"] == "Hello, how are you?"
        assert len(history.timestamps) == 1
        assert history.token_count > 0

    @pytest.mark.asyncio
    async def test_append_multiple_messages(self, session_memory):
        """Test conversation flow with multiple messages."""
        session_id = "session_456"

        # User message
        await session_memory.append_message(session_id, role="user", content="What is Python?")

        # Assistant message
        await session_memory.append_message(
            session_id,
            role="assistant",
            content="Python is a high-level programming language.",
        )

        # Another user message
        await session_memory.append_message(
            session_id, role="user", content="Tell me more about its features."
        )

        history = await session_memory.get_history(session_id)

        assert len(history.messages) == 3
        assert history.messages[0]["role"] == "user"
        assert history.messages[1]["role"] == "assistant"
        assert history.messages[2]["role"] == "user"
        assert len(history.timestamps) == 3

    @pytest.mark.asyncio
    async def test_message_timestamps_are_sequential(self, session_memory):
        """Test that message timestamps are in order."""
        session_id = "session_789"

        await session_memory.append_message(session_id, role="user", content="First")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        await session_memory.append_message(session_id, role="user", content="Second")
        time.sleep(0.01)
        await session_memory.append_message(session_id, role="user", content="Third")

        history = await session_memory.get_history(session_id)

        assert history.timestamps[0] < history.timestamps[1]
        assert history.timestamps[1] < history.timestamps[2]

    @pytest.mark.asyncio
    async def test_token_count_estimation(self, session_memory):
        """Test token count estimation (rough approximation)."""
        session_id = "session_tokens"

        # Append message with known length
        content = "a" * 100  # 100 characters
        await session_memory.append_message(session_id, role="user", content=content)

        history = await session_memory.get_history(session_id)

        # Token count should be roughly 100 / 4 = 25
        assert history.token_count == 25

    @pytest.mark.asyncio
    async def test_message_truncation_at_max_messages(self, session_memory):
        """Test automatic truncation when exceeding max_messages."""
        session_id = "session_truncate"

        # Append 15 messages (max is 10)
        for i in range(15):
            await session_memory.append_message(session_id, role="user", content=f"Message {i}")

        history = await session_memory.get_history(session_id)

        # Should only keep last 10 messages
        assert len(history.messages) == 10
        assert len(history.timestamps) == 10

        # Should have messages 5-14 (oldest 5 dropped)
        assert history.messages[0]["content"] == "Message 5"
        assert history.messages[-1]["content"] == "Message 14"

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, session_memory):
        """Test getting N most recent messages."""
        session_id = "session_recent"

        # Add 5 messages
        for i in range(5):
            await session_memory.append_message(session_id, role="user", content=f"Message {i}")

        # Get last 3 messages
        recent = await session_memory.get_recent_messages(session_id, count=3)

        assert len(recent) == 3
        assert recent[0]["content"] == "Message 2"
        assert recent[-1]["content"] == "Message 4"

    @pytest.mark.asyncio
    async def test_clear_session(self, session_memory):
        """Test clearing a session's history."""
        session_id = "session_clear"

        # Add messages
        await session_memory.append_message(session_id, role="user", content="Message 1")
        await session_memory.append_message(session_id, role="user", content="Message 2")

        # Verify messages exist
        history = await session_memory.get_history(session_id)
        assert len(history.messages) == 2

        # Clear session
        await session_memory.clear_session(session_id)

        # Verify session is empty
        history = await session_memory.get_history(session_id)
        assert len(history.messages) == 0
        assert history.token_count == 0

    @pytest.mark.asyncio
    async def test_message_metadata(self, session_memory):
        """Test storing metadata with messages."""
        session_id = "session_metadata"

        metadata = {"source": "web", "confidence": 0.95}
        await session_memory.append_message(
            session_id, role="assistant", content="Response", metadata=metadata
        )

        history = await session_memory.get_history(session_id)

        # Note: Current implementation doesn't store per-message metadata
        # This test documents expected behavior if we add it
        assert len(history.messages) == 1

    @pytest.mark.asyncio
    async def test_storage_namespacing(self, storage, session_memory):
        """Test that session memory uses 'session::' prefix in storage."""
        session_id = "test_namespace"

        await session_memory.append_message(session_id, role="user", content="Test message")

        # Direct storage access should require session:: prefix
        raw_data = await storage.aget(f"session::{session_id}")
        assert raw_data is not None
        assert raw_data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self, session_memory):
        """Test that multiple sessions are independent."""
        session_1 = "session_one"
        session_2 = "session_two"

        await session_memory.append_message(session_1, role="user", content="Session 1")
        await session_memory.append_message(session_2, role="user", content="Session 2")

        history_1 = await session_memory.get_history(session_1)
        history_2 = await session_memory.get_history(session_2)

        assert history_1.messages[0]["content"] == "Session 1"
        assert history_2.messages[0]["content"] == "Session 2"
        assert len(history_1.messages) == 1
        assert len(history_2.messages) == 1


class TestSessionMemoryPluginEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_max_messages(self, storage):
        """Test behavior with max_messages=0."""
        session_memory = SessionMemoryPlugin(storage=storage, max_messages=0)
        session_id = "session_zero"

        await session_memory.append_message(session_id, role="user", content="Test")

        history = await session_memory.get_history(session_id)

        # Should truncate immediately to 0 messages
        assert len(history.messages) == 0

    @pytest.mark.asyncio
    async def test_very_large_max_messages(self, storage):
        """Test with very large max_messages setting."""
        session_memory = SessionMemoryPlugin(storage=storage, max_messages=10000)
        session_id = "session_large"

        # Add 100 messages
        for i in range(100):
            await session_memory.append_message(session_id, role="user", content=f"Message {i}")

        history = await session_memory.get_history(session_id)

        # All messages should be retained
        assert len(history.messages) == 100

    @pytest.mark.asyncio
    async def test_empty_message_content(self, session_memory):
        """Test appending message with empty content."""
        session_id = "session_empty"

        await session_memory.append_message(session_id, role="user", content="")

        history = await session_memory.get_history(session_id)

        assert len(history.messages) == 1
        assert history.messages[0]["content"] == ""
        assert history.token_count == 0

    @pytest.mark.asyncio
    async def test_unicode_content(self, session_memory):
        """Test messages with unicode characters."""
        session_id = "session_unicode"

        content = "Hello 世界! 🌍 Привет"
        await session_memory.append_message(session_id, role="user", content=content)

        history = await session_memory.get_history(session_id)

        assert history.messages[0]["content"] == content

    @pytest.mark.asyncio
    async def test_get_recent_messages_exceeds_available(self, session_memory):
        """Test requesting more recent messages than available."""
        session_id = "session_exceed"

        # Add only 3 messages
        for i in range(3):
            await session_memory.append_message(session_id, role="user", content=f"Message {i}")

        # Request 10 messages (more than available)
        recent = await session_memory.get_recent_messages(session_id, count=10)

        # Should return all 3 available messages
        assert len(recent) == 3
