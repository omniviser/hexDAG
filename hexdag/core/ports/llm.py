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


@runtime_checkable
class SupportsFunctionCalling(Protocol):
    """Optional protocol for LLMs that support native function/tool calling.

    This protocol enables structured function calling with automatic parsing,
    commonly used by OpenAI, Anthropic, Google, and other modern LLM providers.
    """

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with native tool calling support.

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

        Force specific tool usage::

            response = await llm.aresponse_with_tools(
                messages,
                tools,
                tool_choice={"type": "function", "function": {"name": "search"}}
            )
        """
        ...


class ImageContent(BaseModel):
    """Image content in a vision-enabled message."""

    type: str = "image"  # "image" or "image_url"
    source: str | dict[str, Any]  # URL, base64, or provider-specific format
    detail: str = "auto"  # "low", "high", or "auto" (for OpenAI)


class VisionMessage(BaseModel):
    """Message with optional image content for vision-enabled LLMs."""

    role: str
    content: str | list[dict[str, Any]]  # Text or mixed text+image content


@runtime_checkable
class SupportsVision(Protocol):
    """Optional protocol for LLMs that support vision/multimodal capabilities.

    This protocol enables processing images alongside text in conversations,
    allowing LLMs to analyze, describe, and reason about visual content.

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - aresponse_with_vision_and_tools(): Vision + tool calling combined
    """

    @abstractmethod
    async def aresponse_with_vision(
        self,
        messages: list[VisionMessage],
        max_tokens: int | None = None,
    ) -> str | None:
        """Generate response from messages containing images and text.

        Args
        ----
            messages: List of messages with optional image content
            max_tokens: Optional maximum tokens in response

        Returns
        -------
            Generated response text or None if failed

        Examples
        --------
        Single image analysis::

            messages = [
                VisionMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"}
                        }
                    ]
                )
            ]
            response = await llm.aresponse_with_vision(messages)

        Multiple images comparison::

            messages = [
                VisionMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Compare these two images"},
                        {"type": "image_url", "image_url": {"url": "image1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "image2.jpg"}}
                    ]
                )
            ]

        Base64 encoded image::

            messages = [
                VisionMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQ...",
                                "detail": "high"
                            }
                        }
                    ]
                )
            ]
        """
        ...

    async def aresponse_with_vision_and_tools(
        self,
        messages: list[VisionMessage],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response with both vision and tool calling capabilities (optional).

        Combines vision and function calling for advanced multimodal workflows,
        allowing the LLM to analyze images and invoke tools based on visual content.

        This method is optional. Not all vision models support native tool calling.

        Args
        ----
            messages: Messages with optional image content
            tools: Tool definitions in provider-specific format
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)
            max_tokens: Optional maximum tokens in response

        Returns
        -------
        LLMResponse
            Response with content and optional tool calls

        Examples
        --------
        Image analysis with tool calls::

            tools = [{
                "type": "function",
                "function": {
                    "name": "identify_product",
                    "description": "Look up product details",
                    "parameters": {
                        "type": "object",
                        "properties": {"product_name": {"type": "string"}},
                        "required": ["product_name"]
                    }
                }
            }]

            messages = [
                VisionMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": "What product is this and what's its price?"},
                        {"type": "image_url", "image_url": {"url": "product.jpg"}}
                    ]
                )
            ]

            response = await llm.aresponse_with_vision_and_tools(messages, tools)
            # LLM sees image, identifies product, and calls identify_product tool
        """
        ...
