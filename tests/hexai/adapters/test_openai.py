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
            self.chat = _Chat()

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


@pytest.mark.asyncio
async def test_generate_returns_content(monkeypatch):
    # Fake response objects to mimic OpenAI SDK
    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self):
            self.choices = [_Choice("hello world")]
            self.usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    class _Completions:
        @staticmethod
        async def create(**kwargs):
            return _Resp()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self):
            self.chat = _Chat()

    from hexai.adapters.openai import OpenAIAdapter

    async def _ensure(self):
        self._client = _FakeClient()

    monkeypatch.setattr(OpenAIAdapter, "_ensure_client", _ensure)

    adapter = OpenAIAdapter(model="gpt-test")
    out = await adapter.generate(messages=[{"role": "user", "content": "hi"}])

    assert out["content"] == "hello world"
    assert out["usage"]["total_tokens"] == 5
    assert "raw" in out


@pytest.mark.asyncio
async def test_generate_retries_on_429(monkeypatch):
    calls = {"n": 0}

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self):
            self.choices = [_Choice("ok after retry")]
            self.usage = {"total_tokens": 1}

    class _Err429(Exception):
        def __init__(self):
            self.status = 429

    class _Completions:
        @staticmethod
        async def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Err429()
            return _Resp()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self):
            self.chat = _Chat()

    from hexai.adapters.openai import OpenAIAdapter

    async def _ensure(self):
        self._client = _FakeClient()

    monkeypatch.setattr(OpenAIAdapter, "_ensure_client", _ensure)

    adapter = OpenAIAdapter(model="gpt-test")
    out = await adapter.generate(messages=[{"role": "user", "content": "hi"}])

    assert out["content"] == "ok after retry"
    assert calls["n"] >= 2  # było przynajmniej jedno ponowienie


@pytest.mark.asyncio
async def test_astream_retries_to_create_stream(monkeypatch):
    calls = {"n": 0}

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
        yield _Chunk(content="re")
        yield _Chunk(content="try")
        yield _Chunk(finish="stop")

    class _Err5xx(Exception):
        def __init__(self):
            self.status = 500

    class _Stream:
        def __init__(self, *_a, **_k):
            pass

        def __aiter__(self):
            return _fake_generator()

    class _Completions:
        @staticmethod
        async def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Err5xx()
            return _Stream()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self):
            self.chat = _Chat()

    from hexai.adapters.openai import OpenAIAdapter

    async def _ensure(self):
        self._client = _FakeClient()

    monkeypatch.setattr(OpenAIAdapter, "_ensure_client", _ensure)

    adapter = OpenAIAdapter(model="gpt-test")
    seen = []
    async for d in adapter.astream(messages=[{"role": "user", "content": "hi"}]):
        seen.append(d)

    assert any(c["type"] == "content" and c["data"] == "re" for c in seen)
    assert any(c["type"] == "content" and c["data"] == "try" for c in seen)
    assert any(c["type"] == "finish" and c["data"] == "stop" for c in seen)
    assert calls["n"] >= 2
