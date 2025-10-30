"""Port interface definitions for Large Language Models (LLMs)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from hexdag.core.registry.decorators import port

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


MessageList = list[Message]


class ToolCall(BaseModel):
    """A tool call made by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Response from LLM with optional tool calls."""

    content: str | None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


@port(
    name="llm",
    namespace="core",
)
@runtime_checkable
class LLM(Protocol):
    """Port interface for Large Language Models (LLMs).


    LLMs provide natural language generation capabilities. Implementations
    may use various backends (OpenAI, Anthropic, local models, etc.) but
    must provide the aresponse method for generating text from messages.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify LLM API connectivity and availability
    - aresponse_with_tools(): Native tool calling support (OpenAI/Anthropic style)
    """

    @abstractmethod
    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response from a list of messages (async).

        Args
        ----
            messages: List of role-message dicts, e.g. [{"role": "user", "content": "..."}]

        Returns
        -------
            The generated response as a string, or None if failed.
        """
        pass

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with native tool calling support (optional).

        This method enables native tool calling for LLM providers that support it
        (OpenAI, Anthropic, Gemini, etc.). If not implemented, the framework will
        fall back to text-based tool calling using INVOKE_TOOL: directives.

        Args
        ----
            messages: Conversation messages
            tools: Tool definitions in provider-specific format
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)

        Returns
        -------
        LLMResponse
            Response with content and optional tool calls

        Examples
        --------
        OpenAI-style tool calling::

            tools = [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]

            response = await llm.aresponse_with_tools(messages, tools)
            # response.content: "Let me search for that"
            # response.tool_calls: [{"id": "call_123", "name": "search", "arguments": {...}}]
        """
        ...

    async def ahealth_check(self) -> "HealthStatus":
        """Check LLM adapter health and connectivity (optional).

        Adapters should verify:
        - API connectivity to the LLM service
        - Model availability
        - Authentication status
        - Rate limit status (if applicable)

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about connectivity and availability

        Examples
        --------
        OpenAI adapter health check::

            status = await openai_adapter.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.latency_ms  # Time taken for health check
            status.details  # {"model": "gpt-4", "rate_limit_remaining": 100}
        """
        ...
