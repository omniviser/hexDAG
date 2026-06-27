"""Port interface definitions for Large Language Models (LLMs)."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from pydantic import BaseModel

from hexdag.kernel.orchestration.events.events import Event, PortCallEvent

type MessageRole = Literal["user", "assistant", "system", "tool", "human", "ai"]
type ToolChoice = Literal["auto", "none", "required"]
type ImageContentType = Literal["image", "image_url"]
type ImageDetail = Literal["low", "high", "auto"]


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
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


class BatchItemStatus(StrEnum):
    """Status of an individual item within a batch."""

    COMPLETED = "completed"
    FAILED = "failed"


class BatchItemResult(BaseModel):
    """Result for a single item in a batch response.

    Attributes
    ----------
    index : int
        Original position in the input list (for ordering).
    content : str | None
        Generated text (None if failed).
    status : BatchItemStatus
        Whether this item succeeded.
    error : str | None
        Error message if status is FAILED.
    usage : TokenUsage | None
        Per-item token usage (if available).
    """

    index: int
    content: str | None
    status: BatchItemStatus = BatchItemStatus.COMPLETED
    error: str | None = None
    usage: TokenUsage | None = None


class BatchResult(BaseModel):
    """Aggregated result of a batch generation call.

    Attributes
    ----------
    items : list[BatchItemResult]
        Per-item results, ordered by original input index.
    total_usage : TokenUsage | None
        Aggregated token usage across all items.
    provider : str | None
        Which provider fulfilled the batch (e.g. ``"gather"``).
    """

    items: list[BatchItemResult]
    total_usage: TokenUsage | None = None
    provider: str | None = None

    @property
    def contents(self) -> list[str | None]:
        """Extract just the content strings in input order."""
        return [item.content for item in sorted(self.items, key=lambda i: i.index)]

    @staticmethod
    def aggregate_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
        """Sum token usage across multiple items.

        Parameters
        ----------
        usages : list[TokenUsage | None]
            Per-item usage values.

        Returns
        -------
        TokenUsage | None
            Aggregated totals, or None if no usage data.
        """
        total_in = total_out = total = 0
        any_usage = False
        for u in usages:
            if u is not None:
                any_usage = True
                total_in += u.input_tokens
                total_out += u.output_tokens
                total += u.total_tokens
        if not any_usage:
            return None
        return TokenUsage(
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total,
        )


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
    #
    # Tool calling lives on SupportsFunctionCalling, NOT here.
    # Health checks are adapter-specific, NOT protocol-mandated.
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
class SupportsStreaming(Protocol):
    """Optional protocol for LLMs that support token streaming.

    Adapters implementing this protocol can yield response text
    incrementally as it is generated, enabling real-time UIs
    (chat interfaces, SSE endpoints) without waiting for the full
    completion.

    The stream yields text deltas; concatenating all deltas produces
    the same text ``aresponse()`` would have returned.  Adapters that
    also implement ``SupportsUsageTracking`` should populate usage
    after the stream is exhausted.

    Examples
    --------
    Consuming a token stream::

        async for delta in llm.astream(messages):
            print(delta, end="", flush=True)
    """

    def astream(self, messages: MessageList) -> AsyncIterator[str]:
        """Stream a response as incremental text deltas (async).

        Args
        ----
            messages: List of Message objects with role and content

        Returns
        -------
        AsyncIterator[str]
            Async iterator of text deltas.  Concatenated deltas form
            the complete response.
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
        tool_choice: ToolChoice | dict[str, Any] = "auto",
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


@runtime_checkable
class SupportsStructuredOutput(Protocol):
    """Optional protocol for LLMs that support native structured output.

    This protocol enables LLM providers to guarantee JSON output conforming
    to a given schema, using native API features (e.g., OpenAI JSON Schema mode,
    Anthropic tool_use). Adapters that don't implement this protocol will be
    automatically wrapped with a fallback middleware that uses prompt injection.

    Examples
    --------
    Native structured output::

        from pydantic import BaseModel

        class Analysis(BaseModel):
            sentiment: str
            confidence: float

        result = await llm.aresponse_structured(
            messages=[Message(role="user", content="Analyze: Great product!")],
            output_schema=Analysis,
        )
        # Returns: {"sentiment": "positive", "confidence": 0.95}
    """

    @abstractmethod
    async def aresponse_structured(
        self,
        messages: MessageList,
        output_schema: dict[str, Any] | type[BaseModel],
    ) -> dict[str, Any]:
        """Generate a response conforming to the given schema.

        Args
        ----
            messages: Conversation messages
            output_schema: Expected output schema — either a Pydantic model class
                or a JSON Schema dict. Pydantic models are converted to JSON Schema
                via ``model.model_json_schema()``.

        Returns
        -------
        dict[str, Any]
            Parsed response data conforming to the schema.

        Raises
        ------
        ParseError
            If the response cannot be parsed or validated against the schema.
        """
        ...


class ImageContent(BaseModel):
    """Image content in a vision-enabled message."""

    type: ImageContentType = "image"
    source: str | dict[str, Any]  # URL, base64, or provider-specific format
    detail: ImageDetail = "auto"


class VisionMessage(BaseModel):
    """Message with optional image content for vision-enabled LLMs."""

    role: MessageRole
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
        tool_choice: ToolChoice | dict[str, Any] = "auto",
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


@runtime_checkable
class SupportsBatchGeneration(Protocol):
    """Optional protocol for batch text generation.

    Adapters or middleware implementing this protocol can process multiple
    independent message lists concurrently, similar to LangChain's ``batch()``.

    The ``BatchGeneration`` middleware implements this protocol for any adapter
    by firing concurrent ``aresponse()`` calls via ``asyncio.gather()`` with
    a semaphore for concurrency control.

    Examples
    --------
    Using via middleware::

        llm = BatchGeneration(OpenAIAdapter(api_key="..."), max_concurrency=10)
        result = await llm.aresponse_batch([
            [Message(role="user", content="Summarise doc A")],
            [Message(role="user", content="Summarise doc B")],
        ])
        print(result.contents)  # ["Summary A...", "Summary B..."]
    """

    @abstractmethod
    async def aresponse_batch(
        self,
        message_lists: list[MessageList],
    ) -> BatchResult:
        """Generate responses for multiple message lists concurrently.

        Parameters
        ----------
        message_lists : list[MessageList]
            Independent conversations to process in batch.

        Returns
        -------
        BatchResult
            Aggregated results with per-item status and usage.
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


# ---------------------------------------------------------------------------
# Port events — unified LLMPortCall replaces old per-capability subtypes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMTokenStreamed(Event):
    """A text delta was streamed from an LLM during node execution.

    Emitted by ``LLMNode`` (one event per delta) when streaming is
    enabled and the adapter implements ``SupportsStreaming``.  Observers
    can forward these to SSE endpoints, websockets, or terminal UIs for
    real-time token display.

    Attributes
    ----------
    node_name : str
        Name of the DAG node that is streaming.
    delta : str
        The incremental text chunk.
    index : int
        Zero-based position of this delta in the stream.
    pipeline_name : str
        Name of the pipeline the node belongs to.
    """

    node_name: str
    delta: str
    index: int
    pipeline_name: str = ""

    def log_message(self) -> str:
        """Format log message for token stream event."""
        return f"LLM token [{self.index}] from '{self.node_name}': {self.delta!r}"


@dataclass(slots=True)
class LLMPortCall(PortCallEvent):
    """Event for LLM port calls.

    One class covers all LLM methods.  The inherited ``method`` field
    (``"aresponse"``, ``"aresponse_with_tools"``, ``"aresponse_structured"``,
    ``"aresponse_with_vision"``, ``"aembed"``, ``"astream"``) distinguishes
    call types.

    Attributes
    ----------
    usage : dict[str, int] | None
        Token usage from the call.
    model : str | None
        Model identifier.
    messages : list[dict[str, str]] | None
        Serialised message list (text calls only).
    response : str
        LLM response text.
    tool_calls : list[dict[str, Any]] | None
        Tool calls returned (function-calling calls only).
    """

    usage: dict[str, int] | None = None
    model: str | None = None
    messages: list[dict[str, str]] | None = None
    response: str = ""
    tool_calls: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated — use LLMPortCall directly)
# ---------------------------------------------------------------------------

LLMEvent = LLMPortCall
"""Deprecated alias for :class:`LLMPortCall`."""

LLMGeneration = LLMPortCall
"""Deprecated alias for :class:`LLMPortCall` (method="aresponse")."""

LLMFunctionCalling = LLMPortCall
"""Deprecated alias for :class:`LLMPortCall` (method="aresponse_with_tools")."""

LLMVision = LLMPortCall
"""Deprecated alias for :class:`LLMPortCall` (method="aresponse_with_vision")."""

LLMEmbedding = LLMPortCall
"""Deprecated alias for :class:`LLMPortCall` (method="aembed")."""
