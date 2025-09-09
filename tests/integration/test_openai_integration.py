"""Integration (real API) smoke tests for OpenAIAdapter."""

import os

import pytest

from hexai.adapters.openai import OpenAIAdapter

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_generate_real_api_smoke():
    """Smoke: non-stream call should return at least content/usage when key present."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY in environment")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAIAdapter(model=model)
    out = await adapter.generate(messages=[{"role": "user", "content": "Say 'pong' once."}])

    assert "content" in out
    assert "usage" in out


@pytest.mark.asyncio
async def test_astream_real_api_smoke():
    """Smoke: streaming call should yield content chunks and a finish event."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("no OPENAI_API_KEY in environment")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAIAdapter(model=model)

    messages = [{"role": "user", "content": "Count to three, one token per chunk."}]
    seen = []
    async for ev in adapter.astream(messages=messages):
        seen.append(ev)

    assert any(e["type"] == "content" for e in seen)
    assert any(e["type"] == "finish" for e in seen)


pytestmark = pytest.mark.integration

MODEL = os.getenv("HEXAI_INTEGRATION_MODEL", "gpt-4o-mini")


def _has_openai():
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.asyncio
async def test_openai_integration_basic():
    if not _has_openai():
        pytest.skip("No OPENAI_API_KEY set, skipping integration test")

    ad = OpenAIAdapter(provider="openai", model=MODEL)
    out = await ad.generate(messages=[{"role": "user", "content": "Say 'ok' once."}])

    assert "content" in out
    assert out["content"] is not None
    assert out["usage"] is not None
    assert "raw" in out


@pytest.mark.asyncio
async def test_openai_integration_streaming():
    if not _has_openai():
        pytest.skip("No OPENAI_API_KEY set, skipping integration test")

    ad = OpenAIAdapter(provider="openai", model=MODEL)

    chunks = []
    async for ev in ad.astream(messages=[{"role": "user", "content": "Stream the word 'ok'."}]):
        chunks.append(ev)
        if len(chunks) > 500:
            break

    assert any(c.get("type") == "content" and c.get("data") for c in chunks)
