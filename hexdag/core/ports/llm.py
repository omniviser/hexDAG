"""Port interface definitions for Large Language Models (LLMs)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

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


class TokenUsage(BaseModel):
    """Token usage from an LLM API call.

    Attributes
    ----------
    input_tokens : int
        Number of tokens in the prompt/input
    output_tokens : int
        Number of tokens in the completion/output
    total_tokens : int
        Total tokens (input + output)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from LLM with optional tool calls."""

    content: str | None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@runtime_checkable
class LLM(Protocol):
    """Port interface for Large Language Models (LLMs).

    LLMs provide natural language generation and/or embedding capabilities.
    Implementations may use various backends (OpenAI, Anthropic, local models, etc.).

    At least ONE of the following protocols must be implemented:
    - **SupportsGeneration**: Text generation capabilities (most common)
    - **SupportsEmbedding**: Embedding generation capabilities

    Optional Capabilities
    ---------------------
    Adapters may optionally implement additional protocols:

    - **SupportsGeneration**: Text generation (most LLM adapters)
        - aresponse(): Generate text from messages

    - **SupportsFunctionCalling**: Native tool calling (OpenAI/Anthropic style)
        - aresponse_with_tools(): Generate responses with tool calls

    - **SupportsVision**: Multimodal vision capabilities
        - aresponse_with_vision(): Process images alongside text
        - aresponse_with_vision_and_tools(): Vision + tool calling

    - **SupportsEmbedding**: Embedding generation (unified LLM+embedding adapters)
        - aembed(): Generate text embeddings
        - aembed_batch(): Batch text embeddings
        - aembed_image(): Generate image embeddings (if supported)
        - aembed_image_batch(): Batch image embeddings (if supported)

    - **Health Checks**: Connectivity monitoring
        - ahealth_check(): Verify API connectivity and availability

    Examples
    --------
    Unified adapter (text generation + embeddings)::

        @adapter("llm", name="unified")
        class UnifiedAdapter(LLM, SupportsGeneration, SupportsEmbedding):
            async def aresponse(self, messages):
                # Text generation
                ...

            async def aembed(self, text):
                # Embedding generation
                ...

    Pure embedding adapter::

        @adapter("llm", name="embeddings_only")
        class EmbeddingAdapter(LLM, SupportsEmbedding):
            async def aembed(self, text):
                # Only embeddings
                ...

    Pure text generation adapter::

        @adapter("llm", name="text_only")
        class TextAdapter(LLM, SupportsGeneration):
            async def aresponse(self, messages):
                # Only text generation
                ...
    """

    # No required methods — adapters must implement at least one sub-protocol
    # (SupportsGeneration, SupportsFunctionCalling, SupportsEmbedding, etc.).

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
class SupportsGeneration(Protocol):
    """Optional protocol for LLMs that support text generation.

    This protocol enables basic text generation from conversation messages.
    Most LLM adapters will implement this protocol.
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

        Examples
        --------
        Basic text generation::

            messages = [
                Message(role="user", content="What is 2+2?")
            ]
            response = await llm.aresponse(messages)
            # Returns: "2+2 equals 4."
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


@runtime_checkable
class SupportsUsageTracking(Protocol):
    """Optional protocol for LLM adapters that track token usage.

    Adapters implementing this protocol expose usage data from the last API call,
    enabling cost profiling and token budgeting without changing the core
    SupportsGeneration return type.

    Examples
    --------
    Implementing usage tracking::

        class MyAdapter(SupportsGeneration, SupportsUsageTracking):
            def __init__(self):
                self._last_usage: TokenUsage | None = None

            async def aresponse(self, messages):
                response = await self._call_api(messages)
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                return response.content

            def get_last_usage(self) -> TokenUsage | None:
                return self._last_usage
    """

    def get_last_usage(self) -> TokenUsage | None:
        """Return token usage from the most recent LLM API call.

        Returns
        -------
        TokenUsage | None
            Token usage data, or None if not available
        """
        ...


type ImageInput = str | bytes


@runtime_checkable
class SupportsEmbedding(Protocol):
    """Optional protocol for LLMs that support embedding generation.

    This protocol enables LLM providers to also serve as embedding adapters,
    useful for unified API management when using services like OpenAI or Azure
    that provide both text generation and embedding capabilities.

    This allows a single adapter to implement both LLM and embedding functionality.
    """

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input.

        Args
        ----
            text: Text string to embed

        Returns
        -------
            List of floats representing the embedding vector

        Examples
        --------
        Single text embedding::

            embedding = await llm.aembed("Hello, world!")
            # Returns: [0.123, -0.456, 0.789, ...]
        """
        ...

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently (optional).

        This method enables batch processing for improved performance when
        embedding multiple texts. If not implemented, the framework will
        fall back to sequential calls to aembed().

        Args
        ----
            texts: List of text strings to embed

        Returns
        -------
            List of embedding vectors, one per input text

        Examples
        --------
        Batch embedding::

            texts = ["Hello", "World", "AI"]
            embeddings = await llm.aembed_batch(texts)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        ...

    async def aembed_image(self, image: ImageInput) -> list[float]:
        """Generate embedding vector for a single image input (optional).

        This method enables image embedding for multimodal LLM providers
        that support vision embeddings (e.g., OpenAI CLIP models).

        Args
        ----
            image: Image to embed, either as:
                - str: File path to image or base64-encoded image data
                - bytes: Raw image bytes

        Returns
        -------
            List of floats representing the embedding vector

        Examples
        --------
        Image embedding from file path::

            embedding = await llm.aembed_image("/path/to/image.jpg")
            # Returns: [0.123, -0.456, 0.789, ...]

        Image embedding from bytes::

            with open("image.jpg", "rb") as f:
                image_bytes = f.read()
            embedding = await llm.aembed_image(image_bytes)
        """
        ...

    async def aembed_image_batch(self, images: list[ImageInput]) -> list[list[float]]:
        """Generate embeddings for multiple images efficiently (optional).

        This method enables batch processing for improved performance when
        embedding multiple images.

        Args
        ----
            images: List of images to embed, each can be:
                - str: File path to image or base64-encoded image data
                - bytes: Raw image bytes

        Returns
        -------
            List of embedding vectors, one per input image

        Examples
        --------
        Batch image embedding::

            images = [
                "/path/to/image1.jpg",
                "/path/to/image2.png",
                image_bytes
            ]
            embeddings = await llm.aembed_image_batch(images)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        ...
