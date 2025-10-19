"""Mock LLM implementation for testing purposes."""

import asyncio
from typing import TYPE_CHECKING, Any

from hexdag.core.ports.llm import LLM, MessageList
from hexdag.core.registry import adapter

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


@adapter(name="mock_llm", implements_port="llm", namespace="plugin")
class MockLLM(LLM):
    """Mock implementation of the LLM interface for testing.

    The LLM port interface is stateless, but this mock provides testing utilities like response
    sequencing and call inspection without violating the port contract.
    """

    # Type annotations for attributes
    delay_seconds: float
    responses: list[str]
    call_count: int
    last_messages: MessageList | None
    should_raise: bool

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration.

        Args
        ----
            **kwargs: Configuration options (responses, delay_seconds)
        """
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)

        # Process responses (convert to list if needed)
        responses = kwargs.get("responses")
        if responses is not None:
            if isinstance(responses, str):
                self.responses = [responses]
            else:
                self.responses = responses
        else:
            self.responses = ['{"result": "Mock response"}']

        # Non-config state
        self.call_count = 0
        self.last_messages: MessageList | None = None
        self.should_raise = False

    async def aresponse(self, messages: MessageList) -> str | None:
        """Return a response based on the configured responses.

        Parameters
        ----------
        messages : MessageList
            List of messages to process

        Returns
        -------
        str | None
            Mock response string or None

        Raises
        ------
        Exception
            When should_raise is True for testing error conditions
        """
        self.last_messages = messages

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_raise:
            raise Exception("Mock LLM error for testing")

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]  # Repeat last response

        self.call_count += 1
        return response

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Any:
        """Mock implementation of tool calling - returns response without tools.

        For testing purposes, this just returns a regular response without
        actually calling any tools. Real adapters would parse and execute tools.
        """
        from hexdag.core.ports.llm import LLMResponse

        # Get regular response
        response_text = await self.aresponse(messages)

        # Return as LLMResponse without tool calls
        return LLMResponse(content=response_text, tool_calls=None, finish_reason="stop")

    async def ahealth_check(self) -> "HealthStatus":
        """Health check for Mock LLM (always healthy)."""
        from hexdag.core.ports.healthcheck import HealthStatus

        return HealthStatus(
            status="healthy",
            adapter_name="MockLLM",
            latency_ms=0.1,
        )

    # Testing utilities (not part of the LLM port interface)
    def reset(self) -> None:
        """Reset the mock state for testing."""
        self.call_count = 0
        self.last_messages = None
