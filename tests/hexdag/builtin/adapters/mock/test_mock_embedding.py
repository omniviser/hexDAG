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


# Image Embedding Tests


@pytest.mark.asyncio
async def test_mock_embedding_single_image_string():
    """Test embedding a single image from string path."""
    mock = MockEmbedding(dimensions=1536)

    embedding = await mock.aembed_image("/path/to/image.jpg")

    assert len(embedding) == 1536
    assert all(isinstance(val, float) for val in embedding)
    assert all(-1 <= val <= 1 for val in embedding)
    assert mock.call_count == 1
    assert mock.last_images == ["/path/to/image.jpg"]


@pytest.mark.asyncio
async def test_mock_embedding_single_image_bytes():
    """Test embedding a single image from bytes."""
    mock = MockEmbedding(dimensions=768)

    image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    embedding = await mock.aembed_image(image_bytes)

    assert len(embedding) == 768
    assert all(isinstance(val, float) for val in embedding)
    assert all(-1 <= val <= 1 for val in embedding)
    assert mock.call_count == 1
    assert mock.last_images == [image_bytes]


@pytest.mark.asyncio
async def test_mock_embedding_image_base64():
    """Test embedding a base64-encoded image."""
    mock = MockEmbedding(dimensions=512)

    base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
    embedding = await mock.aembed_image(base64_image)

    assert len(embedding) == 512
    assert all(isinstance(val, float) for val in embedding)
    assert mock.call_count == 1
    assert mock.last_images == [base64_image]


@pytest.mark.asyncio
async def test_mock_embedding_image_deterministic():
    """Test that image embeddings are deterministic."""
    mock1 = MockEmbedding(dimensions=512)
    mock2 = MockEmbedding(dimensions=512)

    image = "/path/to/same_image.png"
    embedding1 = await mock1.aembed_image(image)
    embedding2 = await mock2.aembed_image(image)

    assert embedding1 == embedding2


@pytest.mark.asyncio
async def test_mock_embedding_different_images_different_embeddings():
    """Test that different images produce different embeddings."""
    mock = MockEmbedding(dimensions=512)

    embedding1 = await mock.aembed_image("/path/to/image1.jpg")
    embedding2 = await mock.aembed_image("/path/to/image2.jpg")

    assert embedding1 != embedding2


@pytest.mark.asyncio
async def test_mock_embedding_text_vs_image_same_string():
    """Test that text and image embeddings with same string produce same result."""
    mock = MockEmbedding(dimensions=512)

    text_embedding = await mock.aembed("test")
    image_embedding = await mock.aembed_image("test")

    # Same string content produces same hash-based embedding
    # This is deterministic behavior for testing purposes
    assert text_embedding == image_embedding
    assert mock.call_count == 2


@pytest.mark.asyncio
async def test_mock_embedding_image_batch():
    """Test batch image embedding generation."""
    mock = MockEmbedding(dimensions=768)

    images = [
        "/path/to/image1.jpg",
        "/path/to/image2.png",
        b"\x89PNG\r\n\x1a\n",
    ]
    embeddings = await mock.aembed_image_batch(images)

    assert len(embeddings) == 3
    assert all(len(emb) == 768 for emb in embeddings)
    assert mock.call_count == 3
    assert mock.last_images == images


@pytest.mark.asyncio
async def test_mock_embedding_image_batch_deterministic():
    """Test that batch image embeddings are deterministic."""
    mock1 = MockEmbedding(dimensions=256)
    mock2 = MockEmbedding(dimensions=256)

    images = ["/path/img1.jpg", "/path/img2.png"]
    embeddings1 = await mock1.aembed_image_batch(images)
    embeddings2 = await mock2.aembed_image_batch(images)

    assert embeddings1 == embeddings2


@pytest.mark.asyncio
async def test_mock_embedding_image_with_delay():
    """Test that image embedding delay works correctly."""
    import time

    mock = MockEmbedding(dimensions=256, delay_seconds=0.1)

    start = time.time()
    await mock.aembed_image("/path/to/image.jpg")
    duration = time.time() - start

    assert duration >= 0.1


@pytest.mark.asyncio
async def test_mock_embedding_image_error_handling():
    """Test image embedding error handling."""
    mock = MockEmbedding(dimensions=512)
    mock.should_raise = True

    with pytest.raises(Exception, match="Mock Image Embedding error"):
        await mock.aembed_image("/path/to/image.jpg")


@pytest.mark.asyncio
async def test_mock_embedding_image_batch_error_handling():
    """Test batch image embedding error handling."""
    mock = MockEmbedding(dimensions=512)
    mock.should_raise = True

    images = ["/path/img1.jpg", "/path/img2.png"]
    with pytest.raises(Exception, match="Mock Image Embedding batch error"):
        await mock.aembed_image_batch(images)


@pytest.mark.asyncio
async def test_mock_embedding_mixed_text_and_image():
    """Test using both text and image embedding in sequence."""
    mock = MockEmbedding(dimensions=512)

    text_emb = await mock.aembed("Hello world")
    image_emb = await mock.aembed_image("/path/to/image.jpg")

    assert len(text_emb) == 512
    assert len(image_emb) == 512
    assert text_emb != image_emb
    assert mock.call_count == 2
    assert mock.last_texts == ["Hello world"]
    assert mock.last_images == ["/path/to/image.jpg"]


@pytest.mark.asyncio
async def test_mock_embedding_reset_with_images():
    """Test reset functionality with images."""
    mock = MockEmbedding(dimensions=512)

    await mock.aembed("Test text")
    await mock.aembed_image("/path/to/image.jpg")

    assert mock.call_count == 2
    assert len(mock.last_texts) > 0
    assert len(mock.last_images) > 0

    mock.reset()

    assert mock.call_count == 0
    assert mock.last_texts == []
    assert mock.last_images == []
    assert mock.should_raise is False


@pytest.mark.asyncio
async def test_mock_embedding_image_bytes_vs_string():
    """Test that same content as bytes vs string produces different embeddings."""
    mock = MockEmbedding(dimensions=512)

    content = "image_data"
    string_emb = await mock.aembed_image(content)
    bytes_emb = await mock.aembed_image(content.encode())

    # String and bytes of same content should produce same embedding
    assert string_emb == bytes_emb
