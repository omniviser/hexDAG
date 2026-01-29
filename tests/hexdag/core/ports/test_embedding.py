"""Tests for Embedding port interface."""

import pytest

from hexdag.builtin.adapters.mock.mock_embedding import MockEmbedding
from hexdag.core.ports.embedding import Embedding, ImageInput


def test_embedding_protocol_check():
    """Test that MockEmbedding implements the Embedding protocol."""
    mock = MockEmbedding(dimensions=512)
    assert isinstance(mock, Embedding)


def test_image_input_type_alias():
    """Test that ImageInput type alias is properly defined."""
    # ImageInput should be a type alias for str | bytes
    import typing

    # Just verify it's defined and is a TypeAliasType (Python 3.12+)
    assert ImageInput is not None
    assert isinstance(ImageInput, typing.TypeAliasType)
    # Test that the underlying type is the union we expect
    assert typing.get_args(ImageInput.__value__) == (str, bytes)


@pytest.mark.asyncio
async def test_embedding_protocol_has_aembed():
    """Test that Embedding protocol defines aembed method."""
    mock = MockEmbedding(dimensions=768)
    assert hasattr(mock, "aembed")
    assert callable(mock.aembed)

    # Verify it works
    result = await mock.aembed("test")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embedding_protocol_has_aembed_image():
    """Test that Embedding protocol defines aembed_image method."""
    mock = MockEmbedding(dimensions=768)
    assert hasattr(mock, "aembed_image")
    assert callable(mock.aembed_image)

    # Verify it works with string
    result = await mock.aembed_image("/path/to/image.jpg")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)

    # Verify it works with bytes
    result = await mock.aembed_image(b"image_bytes")
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)


@pytest.mark.asyncio
async def test_embedding_protocol_has_aembed_batch():
    """Test that Embedding protocol defines aembed_batch method."""
    mock = MockEmbedding(dimensions=512)
    assert hasattr(mock, "aembed_batch")
    assert callable(mock.aembed_batch)

    # Verify it works
    result = await mock.aembed_batch(["text1", "text2"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(emb, list) for emb in result)


@pytest.mark.asyncio
async def test_embedding_protocol_has_aembed_image_batch():
    """Test that Embedding protocol defines aembed_image_batch method."""
    mock = MockEmbedding(dimensions=512)
    assert hasattr(mock, "aembed_image_batch")
    assert callable(mock.aembed_image_batch)

    # Verify it works with mixed input types
    result = await mock.aembed_image_batch(["/path/to/image.jpg", b"image_bytes"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(emb, list) for emb in result)


@pytest.mark.asyncio
async def test_embedding_protocol_has_ahealth_check():
    """Test that Embedding protocol defines ahealth_check method."""
    mock = MockEmbedding(dimensions=1024)
    assert hasattr(mock, "ahealth_check")
    assert callable(mock.ahealth_check)

    # Verify it works
    status = await mock.ahealth_check()
    assert status.status == "healthy"
    assert status.adapter_name == "MockEmbedding"


@pytest.mark.asyncio
async def test_embedding_text_return_type():
    """Test that text embedding returns correct type."""
    mock = MockEmbedding(dimensions=256)
    result = await mock.aembed("Hello world")

    assert isinstance(result, list)
    assert len(result) == 256
    assert all(isinstance(val, float) for val in result)
    assert all(-1 <= val <= 1 for val in result)


@pytest.mark.asyncio
async def test_embedding_image_return_type():
    """Test that image embedding returns correct type."""
    mock = MockEmbedding(dimensions=512)
    result = await mock.aembed_image("/path/to/image.png")

    assert isinstance(result, list)
    assert len(result) == 512
    assert all(isinstance(val, float) for val in result)
    assert all(-1 <= val <= 1 for val in result)


@pytest.mark.asyncio
async def test_embedding_batch_text_return_type():
    """Test that batch text embedding returns correct type."""
    mock = MockEmbedding(dimensions=128)
    texts = ["text1", "text2", "text3"]
    result = await mock.aembed_batch(texts)

    assert isinstance(result, list)
    assert len(result) == 3
    for emb in result:
        assert isinstance(emb, list)
        assert len(emb) == 128
        assert all(isinstance(val, float) for val in emb)


@pytest.mark.asyncio
async def test_embedding_batch_image_return_type():
    """Test that batch image embedding returns correct type."""
    mock = MockEmbedding(dimensions=256)
    images = ["/path/img1.jpg", b"bytes", "data:image/png;base64,abc"]
    result = await mock.aembed_image_batch(images)

    assert isinstance(result, list)
    assert len(result) == 3
    for emb in result:
        assert isinstance(emb, list)
        assert len(emb) == 256
        assert all(isinstance(val, float) for val in emb)


@pytest.mark.asyncio
async def test_embedding_protocol_method_signatures():
    """Test that all required protocol methods exist with correct signatures."""
    mock = MockEmbedding(dimensions=512)

    # Test aembed signature
    import inspect

    sig = inspect.signature(mock.aembed)
    assert "text" in sig.parameters
    assert sig.parameters["text"].annotation is str

    # Test aembed_image signature
    sig = inspect.signature(mock.aembed_image)
    assert "image" in sig.parameters
    assert sig.parameters["image"].annotation is ImageInput

    # Test aembed_batch signature
    sig = inspect.signature(mock.aembed_batch)
    assert "texts" in sig.parameters

    # Test aembed_image_batch signature
    sig = inspect.signature(mock.aembed_image_batch)
    assert "images" in sig.parameters


@pytest.mark.asyncio
async def test_embedding_different_dimensions():
    """Test that embeddings respect configured dimensions."""
    dimensions_to_test = [128, 256, 512, 768, 1536]

    for dim in dimensions_to_test:
        mock = MockEmbedding(dimensions=dim)

        # Test text embedding
        text_emb = await mock.aembed("test")
        assert len(text_emb) == dim

        # Test image embedding
        image_emb = await mock.aembed_image("/path/to/image.jpg")
        assert len(image_emb) == dim


@pytest.mark.asyncio
async def test_embedding_multimodal_consistency():
    """Test that text and image embeddings have consistent dimensions."""
    mock = MockEmbedding(dimensions=512)

    text_emb = await mock.aembed("test text")
    image_emb = await mock.aembed_image("/path/to/image.jpg")

    # Both should have same dimensions
    assert len(text_emb) == len(image_emb) == 512

    # Both should have same value range
    assert all(-1 <= val <= 1 for val in text_emb)
    assert all(-1 <= val <= 1 for val in image_emb)
