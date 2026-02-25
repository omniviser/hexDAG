"""OpenAI adapter for LLM interactions with embedding support."""

import json
import os
import time
from typing import Any, Literal

from openai import AsyncOpenAI

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.healthcheck import HealthStatus
from hexdag.kernel.ports.llm import (
    LLM,
    ImageInput,
    LLMResponse,
    MessageList,
    SupportsEmbedding,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsUsageTracking,
    SupportsVision,
    TokenUsage,
    ToolCall,
    ToolChoice,
    VisionMessage,
)
from hexdag.kernel.types import (
    FrequencyPenalty,
    PresencePenalty,
    RetryCount,
    Temperature02,
    TimeoutSeconds,
    TokenCount,
    TopP,
)
from hexdag.stdlib.adapters.base import HexDAGAdapter

logger = get_logger(__name__)

# Convention: OpenAI model options for dropdown menus in Studio UI
OpenAIModel = Literal[
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

# Convention: OpenAI embedding model options for dropdown menus in Studio UI
OpenAIEmbeddingModel = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]


class OpenAIAdapter(
    HexDAGAdapter,
    LLM,
    SupportsGeneration,
    SupportsFunctionCalling,
    SupportsVision,
    SupportsEmbedding,
    SupportsUsageTracking,
    yaml_alias="open_ai_adapter",
    port="llm",
):
    """Unified OpenAI implementation of the LLM port.

    This adapter provides integration with OpenAI's models for:
    - Text generation (GPT-4, GPT-3.5-turbo, etc.)
    - Vision capabilities (GPT-4 Vision)
    - Native tool/function calling
    - Text embeddings (text-embedding-3-small, text-embedding-3-large)

    It implements all optional protocols: SupportsGeneration, SupportsFunctionCalling,
    SupportsVision, and SupportsEmbedding.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit parameter: OpenAIAdapter(api_key="sk-...")
    2. Environment variable: OPENAI_API_KEY
    3. Memory port (orchestrator): secret:OPENAI_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: OpenAIModel = "gpt-4o-mini",
        temperature: Temperature02 = 0.7,
        max_tokens: TokenCount | None = None,
        response_format: Literal["text", "json_object"] = "text",
        seed: int | None = None,
        top_p: TopP = 1.0,
        frequency_penalty: FrequencyPenalty = 0.0,
        presence_penalty: PresencePenalty = 0.0,
        system_prompt: str | None = None,
        timeout: TimeoutSeconds = 60.0,
        max_retries: RetryCount = 2,
        embedding_model: OpenAIEmbeddingModel = "text-embedding-3-small",
        embedding_dimensions: int | None = None,
        **kwargs: Any,  # ← For extra params like organization, base_url
    ):
        """Initialize OpenAI adapter.

        Parameters
        ----------
        api_key : str | None
            OpenAI API key (auto-resolved from OPENAI_API_KEY env var if not provided)
        model : str, default="gpt-4o-mini"
            OpenAI model to use
        temperature : float, default=0.7
            Sampling temperature (0-2)
        max_tokens : int | None, default=None
            Maximum tokens in response
        response_format : Literal["text", "json_object"], default="text"
            Output format
        seed : int | None, default=None
            Random seed for deterministic responses
        top_p : float, default=1.0
            Nucleus sampling parameter
        frequency_penalty : float, default=0.0
            Frequency penalty (-2.0 to 2.0)
        presence_penalty : float, default=0.0
            Presence penalty (-2.0 to 2.0)
        system_prompt : str | None, default=None
            System prompt to prepend to messages
        timeout : float, default=60.0
            Request timeout in seconds
        max_retries : int, default=2
            Maximum retry attempts
        embedding_model : str, default="text-embedding-3-small"
            OpenAI embedding model to use
        embedding_dimensions : int | None, default=None
            Embedding dimensionality (for text-embedding-3 models)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key required (pass directly or set OPENAI_API_KEY)")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.seed = seed
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self._extra_kwargs = kwargs  # Store extra params

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        if org := kwargs.get("organization"):
            client_kwargs["organization"] = org
        if base_url := kwargs.get("base_url"):
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self._last_usage: TokenUsage | None = None

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connection pool resources."""
        await self.client.close()

    def get_last_usage(self) -> TokenUsage | None:
        """Return token usage from the most recent LLM API call."""
        return self._last_usage

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using OpenAI's modern API format.

        Args
        ----
            messages: List of Message objects with role and content

        Returns
        -------
            The generated response text, or None if failed
        """
        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            if self.system_prompt and not any(msg["role"] == "system" for msg in openai_messages):
                openai_messages.insert(0, {"role": "system", "content": self.system_prompt})

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }

            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.seed is not None:
                request_params["seed"] = self.seed

            # Stop sequences from extra kwargs
            if stop_seq := self._extra_kwargs.get("stop_sequences"):
                request_params["stop"] = stop_seq

            if self.response_format == "json_object":
                request_params["response_format"] = {"type": "json_object"}

            # Make API call with modern format
            response = await self.client.chat.completions.create(**request_params)

            # Capture token usage
            self._last_usage = None
            if response.usage:
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content: str = str(message.content)

                    return content

            logger.warning("No content in OpenAI response")
            return None

        except Exception as e:
            logger.error("OpenAI API error: {}", e, exc_info=True)
            return None

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with native OpenAI tool calling.

        Args
        ----
            messages: Conversation messages
            tools: Tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or specific tool dict

        Returns
        -------
        LLMResponse
            Response with content and tool calls

        """
        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            if self.system_prompt and not any(msg["role"] == "system" for msg in openai_messages):
                openai_messages.insert(0, {"role": "system", "content": self.system_prompt})

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "tools": tools,
                "tool_choice": tool_choice,
            }

            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.seed is not None:
                request_params["seed"] = self.seed

            # Stop sequences from extra kwargs
            if stop_seq := self._extra_kwargs.get("stop_sequences"):
                request_params["stop"] = stop_seq

            if self.response_format == "json_object":
                request_params["response_format"] = {"type": "json_object"}

            # Make API call
            response = await self.client.chat.completions.create(**request_params)

            # Capture token usage
            self._last_usage = None
            if response.usage:
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            if not response.choices or len(response.choices) == 0:
                logger.warning("No choices in OpenAI response")
                return LLMResponse(content=None, tool_calls=None)

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Extract content
            content = str(message.content) if message.content else None

            # Extract tool calls
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=self._last_usage,
            )

        except Exception as e:
            logger.error("OpenAI API error with tools: {}", e, exc_info=True)
            raise

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
            response = await adapter.aresponse_with_vision(messages)
        """
        try:
            # Convert VisionMessage to OpenAI format
            openai_messages: list[dict[str, Any]] = []
            for msg in messages:
                if isinstance(msg.content, str):
                    # Simple text message
                    openai_messages.append({"role": msg.role, "content": msg.content})
                else:
                    # Multi-part content (text + images)
                    openai_messages.append({"role": msg.role, "content": msg.content})

            if self.system_prompt and not any(msg["role"] == "system" for msg in openai_messages):
                openai_messages.insert(0, {"role": "system", "content": self.system_prompt})

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }

            # Use provided max_tokens or default
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            elif self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.seed is not None:
                request_params["seed"] = self.seed

            # Stop sequences from extra kwargs
            if stop_seq := self._extra_kwargs.get("stop_sequences"):
                request_params["stop"] = stop_seq

            # Make API call
            response = await self.client.chat.completions.create(**request_params)

            # Capture token usage
            self._last_usage = None
            if response.usage:
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    return str(message.content)

            logger.warning("No content in OpenAI vision response")
            return None

        except Exception as e:
            logger.error("OpenAI API error with vision: {}", e, exc_info=True)
            return None

    async def aresponse_with_vision_and_tools(
        self,
        messages: list[VisionMessage],
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice | dict[str, Any] = "auto",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response with both vision and tool calling capabilities.

        Args
        ----
            messages: Messages with optional image content
            tools: Tool definitions in OpenAI format
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
                        {"type": "text", "text": "What product is this?"},
                        {"type": "image_url", "image_url": {"url": "product.jpg"}}
                    ]
                )
            ]

            response = await adapter.aresponse_with_vision_and_tools(messages, tools)
        """
        try:
            # Convert VisionMessage to OpenAI format
            openai_messages: list[dict[str, Any]] = []
            for msg in messages:
                if isinstance(msg.content, str):
                    openai_messages.append({"role": msg.role, "content": msg.content})
                else:
                    openai_messages.append({"role": msg.role, "content": msg.content})

            if self.system_prompt and not any(msg["role"] == "system" for msg in openai_messages):
                openai_messages.insert(0, {"role": "system", "content": self.system_prompt})

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "tools": tools,
                "tool_choice": tool_choice,
            }

            # Use provided max_tokens or default
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            elif self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.seed is not None:
                request_params["seed"] = self.seed

            # Stop sequences from extra kwargs
            if stop_seq := self._extra_kwargs.get("stop_sequences"):
                request_params["stop"] = stop_seq

            # Make API call
            response = await self.client.chat.completions.create(**request_params)

            # Capture token usage
            self._last_usage = None
            if response.usage:
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            if not response.choices or len(response.choices) == 0:
                logger.warning("No choices in OpenAI vision+tools response")
                return LLMResponse(content=None, tool_calls=None)

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Extract content
            content = str(message.content) if message.content else None

            # Extract tool calls
            tool_calls_list = None
            if message.tool_calls:
                tool_calls_list = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=content,
                tool_calls=tool_calls_list,
                finish_reason=finish_reason,
                usage=self._last_usage,
            )

        except Exception as e:
            logger.error("OpenAI API error with vision+tools: {}", e, exc_info=True)
            raise

    # ========== Embedding Methods (SupportsEmbedding Protocol) ==========

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

            embedding = await adapter.aembed("Hello, world!")
            # Returns: [0.123, -0.456, 0.789, ...]
        """
        try:
            request_params: dict[str, Any] = {
                "model": self.embedding_model,
                "input": text,
            }

            if self.embedding_dimensions is not None:
                request_params["dimensions"] = self.embedding_dimensions

            response = await self.client.embeddings.create(**request_params)

            if response.data and len(response.data) > 0:
                embedding: list[float] = response.data[0].embedding
                return embedding

            logger.warning("No embedding data in OpenAI response")
            return []

        except Exception as e:
            logger.error("OpenAI embedding API error: {}", e, exc_info=True)
            raise

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

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
            embeddings = await adapter.aembed_batch(texts)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        try:
            request_params: dict[str, Any] = {
                "model": self.embedding_model,
                "input": texts,
            }

            if self.embedding_dimensions is not None:
                request_params["dimensions"] = self.embedding_dimensions

            response = await self.client.embeddings.create(**request_params)

            if response.data:
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            logger.warning("No embedding data in OpenAI batch response")
            return [[] for _ in texts]

        except Exception as e:
            logger.error("OpenAI batch embedding API error: {}", e, exc_info=True)
            raise

    async def aembed_image(self, image: ImageInput) -> list[float]:
        """Generate embedding vector for a single image input.

        OpenAI does not currently support image embeddings via the embeddings API.

        Args
        ----
            image: Image to embed

        Raises
        ------
            NotImplementedError: OpenAI doesn't support image embeddings
        """
        raise NotImplementedError(
            "OpenAI does not support image embeddings via the embeddings API. "
            "For multimodal use cases, consider using vision models with aresponse_with_vision()."
        )

    async def aembed_image_batch(self, images: list[ImageInput]) -> list[list[float]]:
        """Generate embeddings for multiple images efficiently.

        OpenAI does not currently support image embeddings via the embeddings API.

        Args
        ----
            images: List of images to embed

        Raises
        ------
            NotImplementedError: OpenAI doesn't support image embeddings
        """
        raise NotImplementedError("OpenAI does not support image embeddings via the embeddings API")

    # ========== Health Check ==========

    async def ahealth_check(self) -> HealthStatus:
        """Check OpenAI adapter health and connectivity.

        Returns
        -------
        HealthStatus
            Current health status with connectivity details
        """
        try:
            # Try a minimal request to verify connectivity
            start = time.time()

            # Use a simple text generation request for health check
            from hexdag.kernel.ports.llm import Message  # lazy: avoid heavy import for health check

            test_messages = [Message(role="user", content="Hi")]
            await self.aresponse(test_messages)

            latency_ms = (time.time() - start) * 1000

            return HealthStatus(
                status="healthy",
                adapter_name=f"OpenAI[{self.model}]",
                latency_ms=latency_ms,
                details={
                    "model": self.model,
                    "embedding_model": self.embedding_model,
                },
            )
        except Exception as e:
            logger.error("Health check failed: {}", e)
            return HealthStatus(
                status="unhealthy",
                adapter_name=f"OpenAI[{self.model}]",
                latency_ms=0.0,
                details={"error": str(e)},
            )
