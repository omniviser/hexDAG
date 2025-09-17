"""Integration tests for LLM adapters with bootstrap and registry system."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hexai.core.bootstrap import bootstrap_registry, ensure_bootstrapped
from hexai.core.config.models import HexDAGConfig
from hexai.core.ports.llm import Message, MessageList
from hexai.core.registry import registry as global_registry


class TestLLMAdaptersBootstrap:
    """Test LLM adapters integration with bootstrap system."""

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Ensure registry is clean before and after each test."""
        if global_registry.ready:
            # Check if _cleanup_state method exists
            if hasattr(global_registry, '_cleanup_state'):
                global_registry._cleanup_state()
        yield
        if global_registry.ready:
            if hasattr(global_registry, '_cleanup_state'):
                global_registry._cleanup_state()

    def test_bootstrap_loads_llm_adapters(self):
        """Test that bootstrap loads LLM adapter plugins."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.anthropic_adapter",
                "hexai.adapters.llm.openai_adapter",
            ]
        )

        # Mock both API keys as available
        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {
                 "ANTHROPIC_API_KEY": "test-anthropic-key",
                 "OPENAI_API_KEY": "test-openai-key"
             }):
            bootstrap_registry()

        components = global_registry.list_components()
        adapter_names = [c.name for c in components if c.component_type.value == "adapter"]

        # Both adapters should be available
        assert "anthropic" in adapter_names
        assert "openai" in adapter_names

    def test_get_llm_adapters_after_bootstrap(self):
        """Test retrieving LLM adapters from registry after bootstrap."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.anthropic_adapter",
                "hexai.adapters.llm.openai_adapter",
            ]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {
                 "ANTHROPIC_API_KEY": "test-key",
                 "OPENAI_API_KEY": "test-key"
             }):
            bootstrap_registry()

        # Get adapters for LLM port
        llm_adapters = global_registry.get_adapters_for_port("llm")
        adapter_names = [a.name for a in llm_adapters]

        assert "anthropic" in adapter_names
        assert "openai" in adapter_names

    @pytest.mark.asyncio
    async def test_instantiate_llm_adapters_from_registry(self):
        """Test instantiating LLM adapters from bootstrapped registry."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.anthropic_adapter",
                "hexai.adapters.llm.openai_adapter",
            ]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {
                 "ANTHROPIC_API_KEY": "test-key",
                 "OPENAI_API_KEY": "test-key"
             }):
            bootstrap_registry()

        # Mock the API clients
        with patch("hexai.adapters.llm.anthropic_adapter.AsyncAnthropic") as mock_anthropic, \
             patch("hexai.adapters.llm.openai_adapter.AsyncOpenAI") as mock_openai:

            mock_anthropic.return_value = AsyncMock()
            mock_openai.return_value = AsyncMock()

            # Get adapter info from registry
            anthropic_info = global_registry.get_info("anthropic", namespace="adapters")
            openai_info = global_registry.get_info("openai", namespace="adapters")

            # Instantiate adapters
            anthropic_adapter = anthropic_info.get_instance(api_key="test-key")
            openai_adapter = openai_info.get_instance(api_key="test-key")

            # Verify instances
            assert anthropic_adapter.__class__.__name__ == "AnthropicAdapter"
            assert openai_adapter.__class__.__name__ == "OpenAIAdapter"

    @pytest.mark.asyncio
    async def test_llm_adapter_functionality_after_bootstrap(self):
        """Test that LLM adapters work correctly after bootstrap."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=["hexai.adapters.llm.anthropic_adapter"]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            bootstrap_registry()

        # Mock Anthropic client
        with patch("hexai.adapters.llm.anthropic_adapter.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello from bootstrapped Anthropic!"
            mock_response.content = [mock_content]
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            # Get adapter from registry
            adapter_info = global_registry.get_info("anthropic", namespace="adapters")
            adapter = adapter_info.get_instance(api_key="test-key")

            # Test functionality
            messages: MessageList = [Message(role="user", content="Hello")]
            response = await adapter.aresponse(messages)

            assert response == "Hello from bootstrapped Anthropic!"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_llm_adapters_concurrent_usage(self):
        """Test using multiple LLM adapters concurrently after bootstrap."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.anthropic_adapter",
                "hexai.adapters.llm.openai_adapter",
            ]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {
                 "ANTHROPIC_API_KEY": "test-key",
                 "OPENAI_API_KEY": "test-key"
             }):
            bootstrap_registry()

        # Mock both clients
        with patch("hexai.adapters.llm.anthropic_adapter.AsyncAnthropic") as mock_anthropic, \
             patch("hexai.adapters.llm.openai_adapter.AsyncOpenAI") as mock_openai:

            # Setup Anthropic mock
            anthropic_client = AsyncMock()
            anthropic_response = MagicMock()
            anthropic_content = MagicMock()
            anthropic_content.text = "Response from Anthropic"
            anthropic_response.content = [anthropic_content]
            anthropic_client.messages.create = AsyncMock(return_value=anthropic_response)
            mock_anthropic.return_value = anthropic_client

            # Setup OpenAI mock
            openai_client = AsyncMock()
            openai_choice = MagicMock()
            openai_choice.message.content = "Response from OpenAI"
            openai_response = MagicMock()
            openai_response.choices = [openai_choice]
            openai_client.chat.completions.create = AsyncMock(return_value=openai_response)
            mock_openai.return_value = openai_client

            # Get both adapters
            anthropic_info = global_registry.get_info("anthropic", namespace="adapters")
            openai_info = global_registry.get_info("openai", namespace="adapters")

            anthropic_adapter = anthropic_info.get_instance(api_key="test-key")
            openai_adapter = openai_info.get_instance(api_key="test-key")

            # Test both adapters
            messages: MessageList = [Message(role="user", content="Test")]

            import asyncio
            anthropic_result, openai_result = await asyncio.gather(
                anthropic_adapter.aresponse(messages),
                openai_adapter.aresponse(messages)
            )

            assert anthropic_result == "Response from Anthropic"
            assert openai_result == "Response from OpenAI"

    def test_ensure_bootstrapped_with_llm_adapters(self):
        """Test that ensure_bootstrapped works with LLM adapters."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=["hexai.adapters.llm.openai_adapter"]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # First call bootstraps
            ensure_bootstrapped()

            # Should have OpenAI adapter
            adapters = global_registry.get_adapters_for_port("llm")
            adapter_names = [a.name for a in adapters]
            assert "openai" in adapter_names

            # Second call should be idempotent
            ensure_bootstrapped()
            adapters2 = global_registry.get_adapters_for_port("llm")
            assert len(adapters2) == len(adapters)

    @pytest.mark.asyncio
    async def test_llm_adapter_error_handling_after_bootstrap(self):
        """Test error handling in LLM adapters loaded via bootstrap."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=["hexai.adapters.llm.openai_adapter"]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            bootstrap_registry()

        # Mock OpenAI client to simulate error
        with patch("hexai.adapters.llm.openai_adapter.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error: Invalid key")
            )
            mock_openai.return_value = mock_client

            # Get adapter from registry
            adapter_info = global_registry.get_info("openai", namespace="adapters")
            adapter = adapter_info.get_instance(api_key="invalid-key")

            # Test error handling
            messages: MessageList = [Message(role="user", content="Test")]
            with patch("builtins.print") as mock_print:
                response = await adapter.aresponse(messages)

            assert response is None
            assert "API Error: Invalid key" in str(mock_print.call_args)

    def test_bootstrap_without_api_keys_skips_adapters(self):
        """Test that bootstrap skips LLM adapters when API keys are missing."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=[
                "hexai.adapters.llm.anthropic_adapter",
                "hexai.adapters.llm.openai_adapter",
            ]
        )

        # No API keys in environment
        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {}, clear=True):
            bootstrap_registry()

        assert global_registry.ready

    @pytest.mark.asyncio
    async def test_adapter_configuration_through_bootstrap(self):
        """Test that adapters can be configured through bootstrap."""
        config = HexDAGConfig(
            modules=["hexai.core.ports"],
            plugins=["hexai.adapters.llm.anthropic_adapter"]
        )

        with patch("hexai.core.config.loader.load_config", return_value=config), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            bootstrap_registry()

        with patch("hexai.adapters.llm.anthropic_adapter.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = AsyncMock()

            # Get adapter info
            adapter_info = global_registry.get_info("anthropic", namespace="adapters")

            # Create instances with different configurations
            adapter1 = adapter_info.get_instance(
                api_key="key1",
                model="claude-3-opus-20240229",
                temperature=0.5
            )
            adapter2 = adapter_info.get_instance(
                api_key="key2",
                model="claude-3-5-sonnet-20241022",
                temperature=0.9
            )

            # Verify different configurations
            assert adapter1.model == "claude-3-opus-20240229"
            assert adapter2.model == "claude-3-5-sonnet-20241022"
            assert adapter1.temperature == 0.5
            assert adapter2.temperature == 0.9
