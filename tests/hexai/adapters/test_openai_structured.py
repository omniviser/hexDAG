import json

import pytest


@pytest.mark.asyncio
async def test_generate_structured_with_pydantic(monkeypatch):
    # --- pydantic model (locally in test) ---
    try:
        from pydantic import BaseModel
    except Exception:
        pytest.skip("pydantic not installed in env")

    class Person(BaseModel):
        name: str
        age: int

    # --- fake OpenAI async client / response like in adapter ---
    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content_json):
            self.choices = [_Choice(content_json)]
            self.usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    class _FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    payload = json.dumps({"name": "Ala", "age": 7})
                    return _Resp(payload)

            completions = _Completions()

        chat = _Chat()

    # --- monkeypatch AsyncOpenAI in adapter module ---
    import hexai.adapters.openai as mod

    mod.AsyncOpenAI = lambda **_: _FakeClient()

    from hexai.adapters.openai import OpenAIAdapter

    ad = OpenAIAdapter(
        provider="openai", model="gpt-test", secrets_provider=lambda k, d=None: "dummy"
    )
    out = await ad.generate(messages=[{"role": "user", "content": "hi"}], schema=Person)

    assert out["content"] is not None
    assert "structured" in out
    assert isinstance(out["structured"], Person)
    assert out["structured"].name == "Ala"
    assert out["structured"].age == 7


@pytest.mark.asyncio
async def test_generate_structured_fallback_callable(monkeypatch):
    def schema_fn(d):
        return {"x": d.get("x")}

    class _Msg:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content_json):
            self.choices = [_Choice(content_json)]
            self.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    class _FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    return _Resp(json.dumps({"x": 123, "y": "ignored"}))

            completions = _Completions()

        chat = _Chat()

    import hexai.adapters.openai as mod

    mod.AsyncOpenAI = lambda **_: _FakeClient()

    from hexai.adapters.openai import OpenAIAdapter

    ad = OpenAIAdapter(
        provider="openai", model="gpt-test", secrets_provider=lambda k, d=None: "dummy"
    )
    out = await ad.generate(messages=[{"role": "user", "content": "hi"}], schema=schema_fn)

    assert out["structured"] == {"x": 123}


@pytest.mark.asyncio
async def test_generate_structured_invalid_json(monkeypatch):
    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content_str):
            self.choices = [_Choice(content_str)]
            self.usage = {}

    class _FakeClient:
        class _Chat:
            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    return _Resp("not-a-json")

            completions = _Completions()

        chat = _Chat()

    import hexai.adapters.openai as mod

    mod.AsyncOpenAI = lambda **_: _FakeClient()

    from hexai.adapters.openai import OpenAIAdapter

    ad = OpenAIAdapter(
        provider="openai", model="gpt-test", secrets_provider=lambda k, d=None: "dummy"
    )
    out = await ad.generate(messages=[{"role": "user", "content": "hi"}], schema=dict)

    assert "structured" not in out
