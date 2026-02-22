"""Tests for Azure OpenAI adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hexdag.kernel.ports.llm import Message

from hexdag_plugins.azure.adapters.openai import AzureOpenAIAdapter


@pytest.fixture
def azure_adapter():
    """Create Azure OpenAI adapter for testing."""
    return AzureOpenAIAdapter(
        api_key="test-key",
        resource_name="test-resource",
        deployment_id="gpt-4",
        api_version="2024-02-15-preview",
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_adapter_initialization(azure_adapter):
    """Test adapter initializes with correct parameters."""
    assert azure_adapter.api_key == "test-key"
    assert azure_adapter.resource_name == "test-resource"
    assert azure_adapter.deployment_id == "gpt-4"
    assert azure_adapter.api_version == "2024-02-15-preview"
    assert azure_adapter.temperature == 0.7
    assert azure_adapter.api_base == "https://test-resource.openai.azure.com"


@pytest.mark.asyncio
async def test_aresponse_success(azure_adapter):
    """Test successful response generation."""
    # Mock OpenAI client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello from Azure!"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        messages = [Message(role="user", content="Hello")]
        response = await azure_adapter.aresponse(messages)

        assert response == "Hello from Azure!"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_aresponse_with_tools(azure_adapter):
    """Test response with tool calling."""
    from hexdag.kernel.ports.llm import LLMResponse

    # Mock tool call response
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "search"
    mock_tool_call.function.arguments = {"query": "test"}

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Let me search"
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.choices[0].finish_reason = "tool_calls"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        messages = [Message(role="user", content="Search for cats")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = await azure_adapter.aresponse_with_tools(messages, tools)

        assert isinstance(response, LLMResponse)
        assert response.content == "Let me search"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_aresponse_error_handling(azure_adapter):
    """Test error handling returns None."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        messages = [Message(role="user", content="Hello")]
        response = await azure_adapter.aresponse(messages)

        assert response is None


@pytest.mark.asyncio
async def test_health_check_healthy(azure_adapter):
    """Test health check returns healthy status."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "OK"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        status = await azure_adapter.ahealth_check()

        assert status.status == "healthy"
        assert status.adapter_name == "AzureOpenAI[gpt-4]"
        assert status.latency_ms > 0
        assert status.details["resource"] == "test-resource"
        assert status.details["deployment"] == "gpt-4"


@pytest.mark.asyncio
async def test_health_check_unhealthy(azure_adapter):
    """Test health check handles errors."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Connection failed"))

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        status = await azure_adapter.ahealth_check()

        assert status.status == "unhealthy"
        assert "error" in status.details


@pytest.mark.asyncio
async def test_client_lazy_initialization(azure_adapter):
    """Test OpenAI client is lazily initialized."""
    assert azure_adapter._client is None

    # Patch at module level where it's imported
    with patch("hexdag_plugins.azure.azure_openai_adapter.AsyncAzureOpenAI") as mock_azure:
        mock_client = MagicMock()
        mock_azure.return_value = mock_client

        client = azure_adapter._get_client()

        assert client is mock_client
        assert azure_adapter._client is mock_client
        mock_azure.assert_called_once_with(
            api_key="test-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://test-resource.openai.azure.com",
            timeout=30.0,
        )


@pytest.mark.asyncio
async def test_client_reuse():
    """Test that client is reused across multiple calls."""
    adapter = AzureOpenAIAdapter(
        api_key="test-key",
        resource_name="test-resource",
        deployment_id="gpt-4",
    )

    # Mock AsyncAzureOpenAI creation
    with patch("hexdag_plugins.azure.azure_openai_adapter.AsyncAzureOpenAI") as mock_azure:
        mock_client = MagicMock()
        mock_azure.return_value = mock_client

        # First call creates client
        client1 = adapter._get_client()
        assert mock_azure.call_count == 1

        # Second call reuses client
        client2 = adapter._get_client()
        assert mock_azure.call_count == 1  # Still 1, not 2

        assert client1 is client2
        assert client1 is mock_client


@pytest.mark.asyncio
async def test_custom_timeout(azure_adapter):
    """Test custom timeout is respected."""
    custom_adapter = AzureOpenAIAdapter(
        api_key="test-key",
        resource_name="test-resource",
        deployment_id="gpt-4",
        timeout=60.0,
    )

    assert custom_adapter.timeout == 60.0


@pytest.mark.asyncio
async def test_max_tokens_parameter(azure_adapter):
    """Test max_tokens parameter is passed to API."""
    azure_adapter.max_tokens = 100

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Short response"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_adapter, "_get_client", return_value=mock_client):
        messages = [Message(role="user", content="Hello")]
        await azure_adapter.aresponse(messages)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100


# ========== Embedding Tests ==========


@pytest.fixture
def azure_embedding_adapter():
    """Create Azure OpenAI adapter with embedding support."""
    return AzureOpenAIAdapter(
        api_key="test-key",
        resource_name="test-resource",
        deployment_id="gpt-4",
        embedding_deployment_id="text-embedding-3-small",
        embedding_dimensions=1536,
    )


@pytest.mark.asyncio
async def test_aembed_success(azure_embedding_adapter):
    """Test successful embedding generation."""
    # Mock embedding response
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embedding_data.index = 0

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_embedding_adapter, "_get_client", return_value=mock_client):
        embedding = await azure_embedding_adapter.aembed("Hello, world!")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_aembed_without_deployment_id():
    """Test embedding fails without embedding_deployment_id."""
    adapter = AzureOpenAIAdapter(
        api_key="test-key",
        resource_name="test-resource",
        deployment_id="gpt-4",
        # No embedding_deployment_id set
    )

    with pytest.raises(ValueError, match="embedding_deployment_id must be set"):
        await adapter.aembed("test")


@pytest.mark.asyncio
async def test_aembed_batch_success(azure_embedding_adapter):
    """Test successful batch embedding generation."""
    # Mock batch embedding response
    mock_data_1 = MagicMock()
    mock_data_1.embedding = [0.1, 0.2, 0.3]
    mock_data_1.index = 0

    mock_data_2 = MagicMock()
    mock_data_2.embedding = [0.4, 0.5, 0.6]
    mock_data_2.index = 1

    mock_response = MagicMock()
    mock_response.data = [mock_data_1, mock_data_2]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_embedding_adapter, "_get_client", return_value=mock_client):
        embeddings = await azure_embedding_adapter.aembed_batch(["Hello", "World"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_aembed_with_dimensions(azure_embedding_adapter):
    """Test embedding respects dimensions parameter."""
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 1536
    mock_embedding_data.index = 0

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_data]

    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch.object(azure_embedding_adapter, "_get_client", return_value=mock_client):
        await azure_embedding_adapter.aembed("test")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 1536


@pytest.mark.asyncio
async def test_aembed_image_not_implemented(azure_embedding_adapter):
    """Test image embedding raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Azure OpenAI does not support image embeddings"):
        await azure_embedding_adapter.aembed_image("image.jpg")


@pytest.mark.asyncio
async def test_aembed_image_batch_not_implemented(azure_embedding_adapter):
    """Test batch image embedding raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Azure OpenAI does not support image embeddings"):
        await azure_embedding_adapter.aembed_image_batch(["image1.jpg", "image2.jpg"])
