"""Tests for Registry ports."""

import pytest

from hexai.core.ports.llm import LLM, MessageList
from hexai.core.ports.registry import PortRegistry, register_port


@pytest.fixture(scope="module", autouse=True)
def register_ports():
    @register_port(name="test_llm", kind="llm")
    class TestLLMAdapter(LLM):
        async def aresponse(self, messages: MessageList) -> str | None:
            return "test response"

    yield


def test_check_if_port_registered():
    assert "test_llm" in PortRegistry._registry, "Port 'test_llm' not registered!"


@pytest.mark.asyncio
async def test_llm_registration():
    cls = PortRegistry.get("test_llm")
    instance = cls()
    resp = await instance.aresponse([{"role": "user", "content": "Hello"}])
    assert resp == "test response"


def test_registry_find_all():
    ports = PortRegistry.find(kind="llm")
    assert "test_llm" in ports


def test_registry_meta():
    meta = PortRegistry.meta("test_llm")
    assert meta["kind"] == "llm"


def test_unregister():
    PortRegistry.unregister("test_llm")
    assert "test_llm" not in PortRegistry.all()
