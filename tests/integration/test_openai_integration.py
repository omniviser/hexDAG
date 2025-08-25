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
