import pytest


@pytest.mark.asyncio
async def test_astream_yields_content_and_finish(monkeypatch):
    # Fake chunk/stream objects to emulate OpenAI streaming
    class _Delta:
        def __init__(self, content=None):
            self.content = content

    class _Choice:
        def __init__(self, content=None, finish=None):
            self.delta = _Delta(content)
            self.finish_reason = finish

    class _Chunk:
        def __init__(self, content=None, finish=None):
            self.choices = [_Choice(content, finish)]

    async def _fake_generator():
        yield _Chunk(content="hel")
        yield _Chunk(content="lo")
        yield _Chunk(finish="stop")

    class _Stream:
        def __init__(self, *_args, **_kwargs):
            pass

        def __aiter__(self):
            return _fake_generator()

    class _Completions:
        @staticmethod
        async def create(**kwargs):
            return _Stream()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self):
            self.Chat = self.Chat()

        class Chat:
            class Completions:
                @staticmethod
                async def create(**kwargs):
                    return _Stream()

    from hexai.adapters.openai import OpenAIAdapter

    async def _ensure(self):
        self._client = _FakeClient()

    monkeypatch.setattr(OpenAIAdapter, "_ensure_client", _ensure)

    adapter = OpenAIAdapter(model="gpt-test")
    chunks = []
    async for d in adapter.astream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(d)

    assert any(c["type"] == "content" and c["data"] == "hel" for c in chunks)
    assert any(c["type"] == "content" and c["data"] == "lo" for c in chunks)
    assert any(c["type"] == "finish" and c["data"] == "stop" for c in chunks)
