"""Tests for LLMFactoryAdapter."""

import pytest

from hexai.adapters.llm_factory_adapter import LLMFactoryAdapter
from hexai.core.ports.llm import MessageList


class DummyModel:
    def response(self, messages: MessageList) -> str:
        return f"Echo: {messages}"


class DummyFactory:
    def create_model(self, model_name: str):
        return DummyModel()


@pytest.mark.asyncio
async def test_llm_factory_adapter_sync():
    model = DummyModel()
    adapter = LLMFactoryAdapter(model)
    result = await adapter.aresponse([{"role": "user", "content": "Hi"}])
    assert result == "Echo: [{'role': 'user', 'content': 'Hi'}]"
