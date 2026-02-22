"""Tests for OpenAI LLM adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag.kernel.ports.llm import Message, MessageList
from hexdag.stdlib.adapters.openai.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    """Test cases for OpenAIAdapter."""

    def test_initialization_with_api_key(self):
        """Test adapter initialization with API key provided."""
        adapter = OpenAIAdapter(api_key="test-key")
        assert adapter.model == "gpt-4o-mini"
        assert adapter.temperature == 0.7
        assert adapter.max_tokens is None
        assert adapter.response_format == "text"
        # API key should be set
        assert adapter.api_key == "test-key"
        # Client should be initialized
        assert adapter.client is not None

    def test_initialization_with_env_variable(self):
        """Test adapter initialization with API key from environment."""
        import os

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            adapter = OpenAIAdapter()
            # API key should be auto-resolved from env
            assert adapter.api_key == "env-key"
            # Client should be initialized
            assert adapter.client is not None

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY from env
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="api_key required"):
                OpenAIAdapter()

    def test_initialization_with_custom_parameters(self):
        """Test adapter initialization with custom parameters."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client:
            adapter = OpenAIAdapter(
                api_key="test-key",
                model="gpt-4o",
                temperature=0.5,
                max_tokens=2000,
                timeout=30.0,
            )
            assert adapter.model == "gpt-4o"
            assert adapter.temperature == 0.5
            assert adapter.max_tokens == 2000
            mock_client.assert_called_once_with(api_key="test-key", timeout=30.0, max_retries=2)

    @pytest.mark.asyncio
    async def test_aresponse_with_user_message(self):
        """Test aresponse with a simple user message."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "Hello! How can I help you?"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result == "Hello! How can I help you?"
            # max_tokens is not passed when None (uses model default)
            # Now includes top_p, frequency_penalty, presence_penalty defaults
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_with_system_message(self):
        """Test aresponse with system and user messages."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "I'm a helpful assistant!"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="system", content="You are a helpful assistant"),
                Message(role="user", content="Who are you?"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "I'm a helpful assistant!"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Who are you?"},
                ],
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_with_conversation(self):
        """Test aresponse with a full conversation."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "I can help with that!"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="Can you help me?"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "I can help with that!"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "Can you help me?"},
                ],
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_with_multiple_system_messages(self):
        """Test aresponse with multiple system messages (OpenAI format keeps them separate)."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [
                Message(role="system", content="System message 1"),
                Message(role="system", content="System message 2"),
                Message(role="user", content="Hello"),
            ]

            result = await adapter.aresponse(messages)

            assert result == "Response"
            # OpenAI keeps system messages as separate entries
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "System message 1"},
                    {"role": "system", "content": "System message 2"},
                    {"role": "user", "content": "Hello"},
                ],
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

    @pytest.mark.asyncio
    async def test_aresponse_empty_choices_returns_none(self):
        """Test aresponse returns None when response has no choices."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response with empty choices
            mock_response = MagicMock()
            mock_response.choices = []

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result is None

    @pytest.mark.asyncio
    async def test_aresponse_none_content_returns_none(self):
        """Test aresponse returns None when message content is None."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock response with None content
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Setup mock client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            result = await adapter.aresponse(messages)

            assert result is None

    @pytest.mark.asyncio
    async def test_aresponse_exception_returns_none(self):
        """Test aresponse returns None and prints error on exception."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Setup mock client to raise exception
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key="test-key")
            messages: MessageList = [Message(role="user", content="Hello")]

            with patch("hexdag.stdlib.adapters.openai.openai_adapter.logger.error") as mock_log:
                result = await adapter.aresponse(messages)

            assert result is None
            mock_log.assert_called_once_with("OpenAI API error: API Error", exc_info=True)

    @pytest.mark.asyncio
    async def test_different_model_configurations(self):
        """Test adapter with different OpenAI model configurations."""
        test_cases = [
            ("gpt-4o", 0.3, 500),
            ("gpt-4o-mini", 1.0, 4096),
            ("gpt-3.5-turbo", 0.0, 150),
        ]

        for model, temp, tokens in test_cases:
            with patch(
                "hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI"
            ) as mock_client_class:
                # Setup mock response
                mock_choice = MagicMock()
                mock_choice.message.content = f"Response from {model}"
                mock_response = MagicMock()
                mock_response.choices = [mock_choice]

                # Setup mock client
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                adapter = OpenAIAdapter(
                    api_key="test-key", model=model, temperature=temp, max_tokens=tokens
                )
                messages: MessageList = [Message(role="user", content="Test")]

                result = await adapter.aresponse(messages)

                assert result == f"Response from {model}"
                mock_client.chat.completions.create.assert_called_once_with(
                    model=model,
                    messages=[{"role": "user", "content": "Test"}],
                    temperature=temp,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=tokens,
                )
