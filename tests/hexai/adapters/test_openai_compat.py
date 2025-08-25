import pytest

from hexai.adapters.openai import OpenAIAdapter


@pytest.mark.asyncio
async def test_ollama_uses_default_baseurl(monkeypatch):
    class _FakeClient:
        def __init__(self, *, api_key=None, base_url=None, default_headers=None, **_):
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers

    import hexai.adapters.openai as openai_adapter

    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", _FakeClient)
    monkeypatch.delenv("OLLAMA_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_COMPAT_API_KEY", raising=False)

    ad = OpenAIAdapter(provider="ollama", model="llama3.1:8b")
    await ad._ensure_client()
    assert ad._client.base_url == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_anthropic_reads_envs(monkeypatch):
    class _FakeClient:
        def __init__(self, *, api_key=None, base_url=None, default_headers=None, **_):
            self.api_key = api_key
            self.base_url = base_url
            self.default_headers = default_headers

    import hexai.adapters.openai as openai_adapter

    monkeypatch.setattr(openai_adapter, "AsyncOpenAI", _FakeClient)

    monkeypatch.setenv("ANTHROPIC_COMPAT_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("ANTHROPIC_COMPAT_API_KEY", "key-123")

    ad = OpenAIAdapter(provider="anthropic", model="claude-3-haiku")
    await ad._ensure_client()
    assert ad._client.base_url == "https://proxy.example/v1"
    assert ad._client.api_key == "key-123"
