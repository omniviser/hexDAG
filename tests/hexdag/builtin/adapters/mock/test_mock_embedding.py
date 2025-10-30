"""Tests for Mock Embedding adapter."""

import pytest

from hexdag.builtin.adapters.mock.mock_embedding import MockEmbedding


@pytest.mark.asyncio
async def test_mock_embedding_initialization():
    """Test that mock embedding adapter can be initialized."""
    mock = MockEmbedding(dimensions=512)

    assert mock.dimensions == 512
    assert mock.delay_seconds == 0.0
    assert mock.call_count == 0
    assert mock.last_texts == []
    assert mock.should_raise is False


@pytest.mark.asyncio
async def test_mock_embedding_single_text():
    """Test embedding a single text."""
    mock = MockEmbedding(dimensions=1536)

    embedding = await mock.aembed("Hello, world!")

    assert len(embedding) == 1536
    assert all(isinstance(val, float) for val in embedding)
    assert all(-1 <= val <= 1 for val in embedding)
    assert mock.call_count == 1
    assert mock.last_texts == ["Hello, world!"]


@pytest.mark.asyncio
async def test_mock_embedding_deterministic():
    """Test that embeddings are deterministic."""
    mock1 = MockEmbedding(dimensions=512)
    mock2 = MockEmbedding(dimensions=512)

    text = "Test text for embedding"
    embedding1 = await mock1.aembed(text)
    embedding2 = await mock2.aembed(text)

    assert embedding1 == embedding2


@pytest.mark.asyncio
async def test_mock_embedding_batch():
    """Test batch embedding generation."""
    mock = MockEmbedding(dimensions=768)

    texts = ["First text", "Second text", "Third text"]
    embeddings = await mock.aembed_batch(texts)

    assert len(embeddings) == 3
    assert all(len(emb) == 768 for emb in embeddings)
    assert mock.call_count == 3
    assert mock.last_texts == texts


@pytest.mark.asyncio
async def test_mock_embedding_different_texts_different_embeddings():
    """Test that different texts produce different embeddings."""
    mock = MockEmbedding(dimensions=512)

    embedding1 = await mock.aembed("First text")
    embedding2 = await mock.aembed("Second text")

    assert embedding1 != embedding2


@pytest.mark.asyncio
async def test_mock_embedding_with_delay():
    """Test that delay works correctly."""
    import time

    mock = MockEmbedding(dimensions=256, delay_seconds=0.1)

    start = time.time()
    await mock.aembed("Test")
    duration = time.time() - start

    assert duration >= 0.1


@pytest.mark.asyncio
async def test_mock_embedding_error_handling():
    """Test error handling."""
    mock = MockEmbedding(dimensions=512)
    mock.should_raise = True

    with pytest.raises(Exception, match="Mock Embedding error"):
        await mock.aembed("Test")


@pytest.mark.asyncio
async def test_mock_embedding_batch_error_handling():
    """Test batch error handling."""
    mock = MockEmbedding(dimensions=512)
    mock.should_raise = True

    with pytest.raises(Exception, match="Mock Embedding batch error"):
        await mock.aembed_batch(["Test1", "Test2"])


@pytest.mark.asyncio
async def test_mock_embedding_health_check():
    """Test health check."""
    mock = MockEmbedding(dimensions=1024)

    status = await mock.ahealth_check()

    assert status.status == "healthy"
    assert status.adapter_name == "MockEmbedding"
    assert status.latency_ms == 0.1
    assert status.details == {"dimensions": 1024}


@pytest.mark.asyncio
async def test_mock_embedding_reset():
    """Test reset functionality."""
    mock = MockEmbedding(dimensions=512)

    await mock.aembed("Test 1")
    await mock.aembed("Test 2")

    assert mock.call_count == 2
    assert len(mock.last_texts) > 0

    mock.reset()

    assert mock.call_count == 0
    assert mock.last_texts == []
    assert mock.should_raise is False
