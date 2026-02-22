"""Integration tests for LLM adapters with fake and real API scenarios."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexdag.kernel.ports.llm import Message, MessageList
from hexdag.stdlib.adapters.anthropic.anthropic_adapter import AnthropicAdapter
from hexdag.stdlib.adapters.openai.openai_adapter import OpenAIAdapter


class TestLLMAdaptersIntegration:
    """Integration tests for LLM adapters."""

    @pytest.fixture
    def fake_api_key(self):
        """Provide a fake API key for testing."""
        return "fake-api-key-for-testing"

    @pytest.fixture
    def test_messages(self):
        """Provide test messages for LLM interactions."""
        return [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello, can you help me?"),
        ]

    @pytest.mark.asyncio
    async def test_anthropic_adapter_with_fake_key_handles_auth_error(
        self, fake_api_key, test_messages, caplog
    ):
        """Test that Anthropic adapter handles authentication errors gracefully."""
        # Mock the Anthropic client to simulate authentication error
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("401: Invalid API key 'fake-api-key-for-testing'")
            )
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(api_key=fake_api_key)
            result = await adapter.aresponse(test_messages)

            # Adapter should return None on error and log it (using loguru, not standard logging)
            assert result is None

    @pytest.mark.asyncio
    async def test_openai_adapter_with_fake_key_handles_auth_error(
        self, fake_api_key, test_messages, caplog
    ):
        """Test that OpenAI adapter handles authentication errors gracefully."""
        # Mock the OpenAI client to simulate authentication error
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("401: Incorrect API key provided: fake-api-key-for-testing")
            )
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key=fake_api_key)
            result = await adapter.aresponse(test_messages)

            # Adapter should return None on error and log it (using loguru, not standard logging)
            assert result is None

    @pytest.mark.asyncio
    async def test_anthropic_adapter_successful_response_simulation(
        self, fake_api_key, test_messages
    ):
        """Test Anthropic adapter with simulated successful response."""
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_client_class:
            # Simulate successful response
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "I'd be happy to help you! What do you need assistance with?"
            mock_response.content = [mock_content]

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = AnthropicAdapter(
                api_key=fake_api_key, model="claude-3-5-sonnet-20241022", temperature=0.7
            )

            result = await adapter.aresponse(test_messages)

            assert result == "I'd be happy to help you! What do you need assistance with?"
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["system"] == "You are a helpful assistant"
            assert len(call_kwargs["messages"]) == 1
            assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_openai_adapter_successful_response_simulation(self, fake_api_key, test_messages):
        """Test OpenAI adapter with simulated successful response."""
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_client_class:
            # Simulate successful response
            mock_choice = MagicMock()
            mock_choice.message.content = "Of course! I'm here to help. What can I assist you with?"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            adapter = OpenAIAdapter(api_key=fake_api_key, model="gpt-4o-mini", temperature=0.7)

            result = await adapter.aresponse(test_messages)

            assert result == "Of course! I'm here to help. What can I assist you with?"
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["temperature"] == 0.7
            assert len(call_kwargs["messages"]) == 2
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_adapters_handle_rate_limit_errors(self, caplog):
        """Test that adapters handle rate limit errors appropriately."""
        fake_key = "rate-limit-test-key"

        # Test Anthropic rate limit
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("429: Rate limit exceeded")
            )
            mock_anthropic.return_value = mock_client

            adapter = AnthropicAdapter(api_key=fake_key)
            messages: MessageList = [Message(role="user", content="Test")]
            result = await adapter.aresponse(messages)

            # Adapter should return None on error (logged via loguru)
            assert result is None

        # Test OpenAI rate limit
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("429: Too many requests")
            )
            mock_openai.return_value = mock_client

            adapter = OpenAIAdapter(api_key=fake_key)
            result = await adapter.aresponse(messages)

            # Adapter should return None on error (logged via loguru)
            assert result is None

    @pytest.mark.asyncio
    async def test_adapters_handle_network_errors(self, caplog):
        """Test that adapters handle network errors appropriately."""
        fake_key = "network-test-key"

        # Test Anthropic network error
        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("Connection timeout"))
            mock_anthropic.return_value = mock_client

            adapter = AnthropicAdapter(api_key=fake_key)
            messages: MessageList = [Message(role="user", content="Test")]
            result = await adapter.aresponse(messages)

            # Adapter should return None on error (logged via loguru)
            assert result is None

        # Test OpenAI network error
        with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Network unreachable")
            )
            mock_openai.return_value = mock_client

            adapter = OpenAIAdapter(api_key=fake_key)
            result = await adapter.aresponse(messages)

            # Adapter should return None on error (logged via loguru)
            assert result is None

    @pytest.mark.asyncio
    async def test_adapters_with_environment_variables(self):
        """Test that adapters correctly use environment variables for API keys."""
        # Test Anthropic with env var
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-anthropic-key"}):
            with patch(
                "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
            ) as mock_anthropic:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_content = MagicMock()
                mock_content.text = "Response from env key"
                mock_response.content = [mock_content]
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_anthropic.return_value = mock_client

                adapter = AnthropicAdapter()  # No API key provided
                messages: MessageList = [Message(role="user", content="Test")]

                result = await adapter.aresponse(messages)

                assert result == "Response from env key"
                # Check that the client was created with the env API key
                # The adapter also passes timeout and max_retries parameters
                mock_anthropic.assert_called_once()
                call_kwargs = mock_anthropic.call_args[1]
                assert call_kwargs["api_key"] == "env-anthropic-key"

        # Test OpenAI with env var
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            with patch("hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_choice = MagicMock()
                mock_choice.message.content = "Response from env key"
                mock_response = MagicMock()
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                adapter = OpenAIAdapter()  # No API key provided
                messages: MessageList = [Message(role="user", content="Test")]

                result = await adapter.aresponse(messages)

                assert result == "Response from env key"
                # Check that the client was created with the env API key
                # The adapter also passes timeout and max_retries parameters
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "env-openai-key"

    @pytest.mark.asyncio
    async def test_adapters_model_switching(self):
        """Test that adapters work correctly with different model configurations."""
        test_cases = [
            ("anthropic", "claude-3-opus-20240229", AnthropicAdapter),
            ("openai", "gpt-4o", OpenAIAdapter),
        ]

        for provider, model, adapter_class in test_cases:
            if provider == "anthropic":
                with patch(
                    "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
                ) as mock_client_class:
                    mock_response = MagicMock()
                    mock_content = MagicMock()
                    mock_content.text = f"Response from {model}"
                    mock_response.content = [mock_content]

                    mock_client = AsyncMock()
                    mock_client.messages.create = AsyncMock(return_value=mock_response)
                    mock_client_class.return_value = mock_client

                    adapter = adapter_class(api_key="test-key", model=model)
                    messages: MessageList = [Message(role="user", content="Test")]

                    result = await adapter.aresponse(messages)

                    assert result == f"Response from {model}"
                    call_kwargs = mock_client.messages.create.call_args[1]
                    assert call_kwargs["model"] == model

            elif provider == "openai":
                with patch(
                    "hexdag.stdlib.adapters.openai.openai_adapter.AsyncOpenAI"
                ) as mock_client_class:
                    mock_choice = MagicMock()
                    mock_choice.message.content = f"Response from {model}"
                    mock_response = MagicMock()
                    mock_response.choices = [mock_choice]

                    mock_client = AsyncMock()
                    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                    mock_client_class.return_value = mock_client

                    adapter = adapter_class(api_key="test-key", model=model)
                    messages: MessageList = [Message(role="user", content="Test")]

                    result = await adapter.aresponse(messages)

                    assert result == f"Response from {model}"
                    call_kwargs = mock_client.chat.completions.create.call_args[1]
                    assert call_kwargs["model"] == model

    @pytest.mark.asyncio
    async def test_adapters_concurrent_requests(self):
        """Test that adapters handle concurrent requests correctly."""
        import asyncio

        with patch(
            "hexdag.stdlib.adapters.anthropic.anthropic_adapter.AsyncAnthropic"
        ) as mock_anthropic:
            mock_client = AsyncMock()

            # Create responses for each request
            responses = []
            for i in range(5):
                mock_response = MagicMock()
                mock_content = MagicMock()
                mock_content.text = f"Response {i}"
                mock_response.content = [mock_content]
                responses.append(mock_response)

            # Setup mock to return responses in order
            mock_client.messages.create = AsyncMock(side_effect=responses)
            mock_anthropic.return_value = mock_client

            adapter = AnthropicAdapter(api_key="test-key")

            # Create multiple concurrent requests
            messages_list = [[Message(role="user", content=f"Test {i}")] for i in range(5)]

            tasks = [adapter.aresponse(msgs) for msgs in messages_list]
            results = await asyncio.gather(*tasks)

            # Check that all requests completed
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result == f"Response {i}"
