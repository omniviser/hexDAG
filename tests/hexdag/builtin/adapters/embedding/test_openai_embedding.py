"""Tests for OpenAI Embedding adapter."""

import pytest

from hexdag.builtin.adapters.embedding.openai_embedding import OpenAIEmbeddingAdapter


@pytest.mark.asyncio
async def test_openai_embedding_initialization():
    """Test that OpenAI embedding adapter can be initialized."""
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        model="text-embedding-3-small",
    )

    assert adapter.api_key == "test-key"
    assert adapter.model == "text-embedding-3-small"
    assert adapter.dimensions is None
    assert adapter.timeout == 60.0
    assert adapter.max_retries == 2


@pytest.mark.asyncio
async def test_openai_embedding_with_dimensions():
    """Test that OpenAI embedding adapter supports dimension reduction."""
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        model="text-embedding-3-large",
        dimensions=512,
    )

    assert adapter.dimensions == 512


@pytest.mark.asyncio
async def test_openai_embedding_with_custom_params():
    """Test that OpenAI embedding adapter supports custom parameters."""
    adapter = OpenAIEmbeddingAdapter(
        api_key="test-key",
        model="text-embedding-ada-002",
        timeout=120.0,
        max_retries=5,
        organization="test-org",
        base_url="https://custom-api.example.com",
    )

    assert adapter.timeout == 120.0
    assert adapter.max_retries == 5
    assert "organization" in adapter._extra_kwargs
    assert "base_url" in adapter._extra_kwargs
