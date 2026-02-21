"""Mock LLM implementation for testing purposes."""

import asyncio
from typing import TYPE_CHECKING, Any

from hexdag.core.ports.llm import (
    LLM,
    MessageList,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsUsageTracking,
    TokenUsage,
)

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


class MockLLM(LLM, SupportsGeneration, SupportsFunctionCalling, SupportsUsageTracking):
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
    mock_tool_calls: list[dict[str, Any]] | None

    def __init__(
        self,
        responses: str | list[str] | None = None,
        delay_seconds: float = 0.0,
        mock_tool_calls: list[dict[str, Any]] | None = None,
        mock_usage: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with configuration.

        Args
        ----
            responses: Mock response(s) to return. Can be a single string or list of strings.
                Each call cycles through the list. Defaults to '{"result": "Mock response"}'.
            delay_seconds: Simulated latency in seconds before returning response.
            mock_tool_calls: List of tool call configurations for testing tool-using agents.
                Each entry should have 'id', 'name', and 'arguments' keys.
            mock_usage: Token usage to return from get_last_usage(). Dict with
                'input_tokens', 'output_tokens', 'total_tokens' keys.
            **kwargs: Additional options for forward compatibility.
        """
        self.delay_seconds = delay_seconds
        self.mock_usage = mock_usage
        self._last_usage: TokenUsage | None = None

        # Process responses (convert to list if needed)
        if responses is not None:
            if isinstance(responses, str):
                self.responses = [responses]
            else:
                self.responses = list(responses)
        else:
            self.responses = ['{"result": "Mock response"}']

        # Process mock tool calls
        self.mock_tool_calls = mock_tool_calls

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

        # Set mock token usage if configured
        if self.mock_usage:
            self._last_usage = TokenUsage(**self.mock_usage)
        else:
            self._last_usage = None

        return response

    def get_last_usage(self) -> TokenUsage | None:
        """Return token usage from the most recent LLM API call."""
        return self._last_usage

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Any:
        """Mock implementation of tool calling with configurable tool call simulation.

        For testing purposes, this can simulate tool calls based on configuration.
        If mock_tool_calls are configured, it will return those. Otherwise, it
        returns a regular response without tool calls.

        Examples
        --------
        Configure mock to return tool calls::

            mock_llm = MockLLM(
                responses=["I'll search for that"],
                mock_tool_calls=[
                    {
                        "id": "call_123",
                        "name": "search",
                        "arguments": {"query": "test"}
                    }
                ]
            )
        """
        from hexdag.core.ports.llm import LLMResponse, ToolCall

        # Get regular response
        response_text = await self.aresponse(messages)

        # Check if mock tool calls are configured
        mock_tool_calls = getattr(self, "mock_tool_calls", None)

        if mock_tool_calls and self.call_count <= len(mock_tool_calls):
            # Return configured tool calls
            tool_call_data = (
                mock_tool_calls[self.call_count - 1]
                if self.call_count <= len(mock_tool_calls)
                else mock_tool_calls[-1]
            )

            if isinstance(tool_call_data, dict):
                tool_calls_list = [
                    ToolCall(
                        id=tool_call_data.get("id", "call_mock"),
                        name=tool_call_data.get("name", "mock_tool"),
                        arguments=tool_call_data.get("arguments", {}),
                    )
                ]
            elif isinstance(tool_call_data, list):
                tool_calls_list = [
                    ToolCall(
                        id=tc.get("id", f"call_mock_{i}"),
                        name=tc.get("name", "mock_tool"),
                        arguments=tc.get("arguments", {}),
                    )
                    for i, tc in enumerate(tool_call_data)
                ]
            else:
                tool_calls_list = None

            return LLMResponse(
                content=response_text,
                tool_calls=tool_calls_list,
                finish_reason="tool_calls" if tool_calls_list else "stop",
            )

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
