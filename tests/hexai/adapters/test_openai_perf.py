import time

import pytest

ITER = 200


class _FakeClient:
    class _Chat:
        class _Completions:
            @staticmethod
            async def create(**kwargs):
                return {
                    "choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        completions = _Completions()

    chat = _Chat()


@pytest.mark.asyncio
async def test_adapter_overhead_under_5pct(monkeypatch):
    async def raw_call():
        return await _FakeClient._Chat._Completions.create()

    import hexai.adapters.openai as mod

    mod.AsyncOpenAI = lambda **_: _FakeClient()

    from hexai.adapters.openai import OpenAIAdapter

    ad = OpenAIAdapter(
        provider="openai", model="gpt-test", secrets_provider=lambda k, d=None: "dummy"
    )

    # warmup
    for _ in range(20):
        await raw_call()
        await ad.generate(messages=[{"role": "user", "content": "hi"}])

    t0 = time.perf_counter()
    for _ in range(ITER):
        await raw_call()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(ITER):
        await ad.generate(messages=[{"role": "user", "content": "hi"}])
    t3 = time.perf_counter()

    raw = t1 - t0
    via_adapter = t3 - t2
    overhead = (via_adapter - raw) / max(raw, 1e-9)

    assert overhead < 0.05, f"Adapter overhead {overhead * 100:.2f}% exceeds 5%"
