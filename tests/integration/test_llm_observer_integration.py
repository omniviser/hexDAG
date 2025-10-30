"""Integration tests for LLM nodes with observer manager.

Tests the fix for the role attribute bug where LLM nodes would fail
with 'dict' object has no attribute 'role' when observer manager was present.
"""

import pytest

from hexdag.builtin.adapters.local import LocalObserverManager
from hexdag.builtin.adapters.mock import MockLLM
from hexdag.builtin.nodes.llm_node import LLMNode
from hexdag.builtin.nodes.raw_llm_node import RawLLMNode
from hexdag.core.bootstrap import ensure_bootstrapped
from hexdag.core.domain.dag import DirectedGraph
from hexdag.core.orchestration.events import LLMPromptSent, LLMResponseReceived
from hexdag.core.orchestration.events.decorators import observer
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.ports.llm import Message


class EventCollector:
    """Observer that collects LLM events."""

    def __init__(self):
        self.events = []

    @observer(event_types=[LLMPromptSent])
    async def on_llm_prompt_sent(self, event: LLMPromptSent):
        """Collect LLMPromptSent events."""
        # Verify messages in event are dicts (for serialization)
        assert all(isinstance(m, dict) for m in event.messages), "Event messages should be dicts"
        assert all("role" in m for m in event.messages), "Event messages must have 'role' key"
        assert all("content" in m for m in event.messages), "Event messages must have 'content' key"
        self.events.append(event)

    @observer(event_types=[LLMResponseReceived])
    async def on_llm_response_received(self, event: LLMResponseReceived):
        """Collect LLMResponseReceived events."""
        self.events.append(event)


@pytest.fixture(autouse=True)
def setup():
    """Ensure bootstrap before each test."""
    ensure_bootstrapped()


class TestLLMNodeWithObserver:
    """Test BaseLLMNode (legacy) with observer manager."""

    @pytest.mark.asyncio
    async def test_llm_node_with_observer_manager(self):
        """Test that LLMNode works with observer manager (fixes role attribute bug)."""
        # Create observer manager with event collector
        observer_mgr = LocalObserverManager(batching_enabled=False)
        collector = EventCollector()
        observer_mgr.register(collector.on_llm_prompt_sent)
        observer_mgr.register(collector.on_llm_response_received)

        # Create DAG with LLM node
        graph = DirectedGraph()
        llm_node = LLMNode.from_template(
            name="test_llm",
            template="Summarize: {{text}}",
        )
        graph.add(llm_node)

        # Create orchestrator and execute with observer manager
        mock_llm = MockLLM(responses="This is a summary")
        orchestrator = Orchestrator()

        # Execute - this should NOT raise AttributeError: 'dict' object has no attribute 'role'
        result = await orchestrator.run(
            graph,
            {"text": "Long text to summarize"},
            additional_ports={"llm": mock_llm, "observer_manager": observer_mgr},
        )

        # Verify execution succeeded
        assert "test_llm" in result
        assert result["test_llm"] == "This is a summary"

        # CRITICAL: Verify LLM received Message objects (not dicts)
        # This is the core of the bug fix - the port wrapper expects Message objects
        assert mock_llm.last_messages is not None, "MockLLM should have received messages"
        assert all(isinstance(m, Message) for m in mock_llm.last_messages), (
            "LLM must receive Message objects, not dicts"
        )
        assert all(hasattr(m, "role") for m in mock_llm.last_messages), (
            "Messages must have .role attribute"
        )
        assert all(hasattr(m, "content") for m in mock_llm.last_messages), (
            "Messages must have .content attribute"
        )

        # Verify events were collected
        assert len(collector.events) == 2
        assert any(isinstance(e, LLMPromptSent) for e in collector.events)
        assert any(isinstance(e, LLMResponseReceived) for e in collector.events)

        # Verify event messages are dicts (for serialization)
        prompt_event = next(e for e in collector.events if isinstance(e, LLMPromptSent))
        assert all(isinstance(m, dict) for m in prompt_event.messages)


class TestRawLLMNodeWithObserver:
    """Test RawLLMNode with observer manager."""

    @pytest.mark.asyncio
    async def test_raw_llm_node_with_messages_input(self):
        """Test RawLLMNode with messages input and observer manager."""
        observer_mgr = LocalObserverManager(batching_enabled=False)
        collector = EventCollector()
        observer_mgr.register(collector.on_llm_prompt_sent)
        observer_mgr.register(collector.on_llm_response_received)

        # Create DAG with RawLLMNode
        graph = DirectedGraph()
        raw_llm = RawLLMNode()
        raw_llm_spec = raw_llm(name="raw_llm")
        graph.add(raw_llm_spec)

        # Create orchestrator
        mock_llm = MockLLM(responses="Raw response")
        orchestrator = Orchestrator()

        # Execute with messages input
        result = await orchestrator.run(
            graph,
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ]
            },
            additional_ports={"llm": mock_llm, "observer_manager": observer_mgr},
        )

        # Verify execution succeeded
        assert "raw_llm" in result
        assert result["raw_llm"].text == "Raw response"

        # CRITICAL: Verify LLM received Message objects (not dicts)
        assert mock_llm.last_messages is not None
        assert all(isinstance(m, Message) for m in mock_llm.last_messages)
        assert mock_llm.last_messages[0].role == "system"
        assert mock_llm.last_messages[1].role == "user"

        # Verify events
        assert len(collector.events) == 2

    @pytest.mark.asyncio
    async def test_raw_llm_node_with_text_input(self):
        """Test RawLLMNode with text input and observer manager."""
        observer_mgr = LocalObserverManager(batching_enabled=False)
        collector = EventCollector()
        observer_mgr.register(collector.on_llm_prompt_sent)
        observer_mgr.register(collector.on_llm_response_received)

        graph = DirectedGraph()
        raw_llm = RawLLMNode()
        raw_llm_spec = raw_llm(name="raw_llm")
        graph.add(raw_llm_spec)

        mock_llm = MockLLM(responses="Text response")
        orchestrator = Orchestrator()

        # Execute with text input (converted to user message internally)
        result = await orchestrator.run(
            graph,
            {"text": "What is AI?"},
            additional_ports={"llm": mock_llm, "observer_manager": observer_mgr},
        )

        # Verify success
        assert result["raw_llm"].text == "Text response"

        # CRITICAL: Verify Message objects
        assert mock_llm.last_messages is not None
        assert len(mock_llm.last_messages) == 1
        assert isinstance(mock_llm.last_messages[0], Message)
        assert mock_llm.last_messages[0].role == "user"
        assert mock_llm.last_messages[0].content == "What is AI?"

    @pytest.mark.asyncio
    async def test_raw_llm_node_with_tools(self):
        """Test RawLLMNode with tools and observer manager."""
        observer_mgr = LocalObserverManager(batching_enabled=False)
        collector = EventCollector()
        observer_mgr.register(collector.on_llm_prompt_sent)
        observer_mgr.register(collector.on_llm_response_received)

        graph = DirectedGraph()
        raw_llm = RawLLMNode()
        raw_llm_spec = raw_llm(
            name="raw_llm_with_tools",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
        graph.add(raw_llm_spec)

        mock_llm = MockLLM(responses="Tool response")
        orchestrator = Orchestrator()

        # Execute with tools
        result = await orchestrator.run(
            graph,
            {"text": "Search for Python tutorials"},
            additional_ports={"llm": mock_llm, "observer_manager": observer_mgr},
        )

        # Verify success
        assert result["raw_llm_with_tools"].text == "Tool response"

        # CRITICAL: Verify aresponse_with_tools received Message objects
        # The MockLLM.aresponse_with_tools internally calls aresponse, so last_messages is set
        assert mock_llm.last_messages is not None
        assert all(isinstance(m, Message) for m in mock_llm.last_messages)

        # Verify events
        assert len(collector.events) == 2
