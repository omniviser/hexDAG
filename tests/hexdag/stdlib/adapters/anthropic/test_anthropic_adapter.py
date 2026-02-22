"""Tests for Anthropic LLM adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag.kernel.ports.llm import Message, MessageList
from hexdag.stdlib.adapters.anthropic.anthropic_adapter import AnthropicAdapter


class TestAnthropicAdapter:
    """Test cases for AnthropicAdapter."""

    def test_initialization_with_api_key(self):
        """Test adapter initialization with API key provided."""
        adapter = AnthropicAdapter(api_key="test-key")
        assert adapter.model == "claude-3-5-sonnet-20241022"
        assert adapter.temperature == 0.7
        assert adapter.max_tokens == 4096
        # API key should be hidden in config
        assert adapter.api_key is not None
        # Client should be initialized
        assert adapter.client is not None

    def test_initialization_with_env_variable(self):
        """Test adapter initialization with API key from environment."""
        import os

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            adapter = AnthropicAdapter()
            # API key should be auto-resolved from env and hidden
            assert adapter.api_key is not None
            # Client should be initialized
            assert adapter.client is not None

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY from env
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="api_key required"):
                AnthropicAdapter()

    def test_initialization_with_custom_parameters(self):
        """Test adapter initialization with custom parameters."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client:
            adapter = AnthropicAdapter(
                api_key="test-key",
                model="claude-3-opus-20240229",
                temperature=0.5,
                max_tokens=2000,
                timeout=30.0,
            )
            assert adapter.model == "claude-3-opus-20240229"
            assert adapter.temperature == 0.5
            assert adapter.max_tokens == 2000
            # Check that client was called with expected arguments
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_aresponse_with_user_message(self):
        """Test aresponse with a simple user message."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello! How can I help you?"
            mock_response.content = [mock_content]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result == "Hello! How can I help you?"
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=4096,
                top_p=1.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_with_system_message(self):
        """Test aresponse with system and user messages."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "I'm a helpful assistant!"
            mock_response.content = [mock_content]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="system", content="You are a helpful assistant"),
                Message(role="user", content="Who are you?"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "I'm a helpful assistant!"
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": "Who are you?"}],
                temperature=0.7,
                max_tokens=4096,
                top_p=1.0,
                system="You are a helpful assistant",
            )

    @pytest.mark.asyncio
    async def test_aresponse_with_multiple_system_messages(self):
        """Test aresponse concatenates multiple system messages."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response.content = [mock_content]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="system", content="System message 1"),
                Message(role="system", content="System message 2"),
                Message(role="user", content="Hello"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "Response"
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]
            assert call_args["system"] == "System message 1\nSystem message 2"
            assert call_args["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_aresponse_with_conversation(self):
        """Test aresponse with a full conversation."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "I can help with that!"
            mock_response.content = [mock_content]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="Can you help me?"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "I can help with that!"
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Can you help me?"},
                ],
                temperature=0.7,
                max_tokens=4096,
                top_p=1.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_empty_content_returns_none(self):
        """Test aresponse returns None when response has empty content."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response with empty content
            mock_response = MagicMock()
            mock_response.content = []

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result is None

    @pytest.mark.asyncio
    async def test_aresponse_no_text_attribute_returns_none(self):
        """Test aresponse returns None when content has no text attribute."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock response with content lacking text attribute
            mock_response = MagicMock()
            mock_content = MagicMock(spec=[])  # No text attribute
            mock_response.content = [mock_content]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result is None

    @pytest.mark.asyncio
    async def test_aresponse_exception_returns_none(self):
        """Test aresponse returns None and prints error on exception."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Setup mock client to raise exception
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            with patch(
                "hexdag.stdlib.adapters.anthropic.anthropic_adapter.logger.error"
            ) as mock_log:
                result = await adapter.aresponse(messages)

            assert result is None
            mock_log.assert_called_once_with("Anthropic API error: API Error", exc_info=True)
