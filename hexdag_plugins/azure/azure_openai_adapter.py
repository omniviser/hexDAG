"""Azure OpenAI LLM adapter for hexDAG framework."""

import time
from typing import Any

from hexdag.core.ports.healthcheck import HealthStatus
from hexdag.core.ports.llm import (
    LLM,
    ImageInput,
    LLMResponse,
    Message,
    MessageList,
    SupportsEmbedding,
    SupportsGeneration,
    ToolCall,
)
from hexdag.core.registry import adapter
from openai import AsyncAzureOpenAI


@adapter(
    "llm",
    name="azure_openai",
    secrets={
        "api_key": "AZURE_OPENAI_API_KEY",
    },
)
class AzureOpenAIAdapter(LLM, SupportsGeneration, SupportsEmbedding):
    """Azure OpenAI adapter for LLM port with embedding support.

    Supports Azure-hosted OpenAI endpoints with deployment-based model access.
    Compatible with GPT-4, GPT-3.5-turbo, fine-tuned models, and text-embedding models.

    This adapter implements both LLM (text generation) and embedding functionality,
    allowing unified API management for Azure OpenAI resources.

    Parameters
    ----------
    api_key : str
        Azure OpenAI API key (auto-resolved from AZURE_OPENAI_API_KEY)
    resource_name : str
        Azure OpenAI resource name (e.g., "my-openai-resource")
    deployment_id : str
        Azure deployment name (e.g., "gpt-4", "gpt-35-turbo")
    api_version : str, optional
        Azure OpenAI API version (default: "2024-02-15-preview")
    temperature : float, optional
        Sampling temperature 0.0-2.0 (default: 0.7)
    max_tokens : int, optional
        Maximum tokens in response (default: None - model default)
    timeout : float, optional
        Request timeout in seconds (default: 30.0)

    Examples
    --------
    YAML configuration::

        nodes:
          - kind: llm_node
            metadata:
              name: azure_analyzer
            spec:
              adapter:
                type: azure_openai
                params:
                  resource_name: "my-openai-eastus"
                  deployment_id: "gpt-4"
                  api_version: "2024-02-15-preview"
                  temperature: 0.7
              prompt_template: "Analyze: {{input}}"

    Python usage (LLM + Embeddings)::

        from hexdag_plugins.azure import AzureOpenAIAdapter

        # Unified adapter for both text generation and embeddings
        adapter = AzureOpenAIAdapter(
            api_key="your-key",  # or auto-resolved from env
            resource_name="my-openai-resource",
            deployment_id="gpt-4",
            embedding_deployment_id="text-embedding-3-small",
        )

        # Text generation
        messages = [{"role": "user", "content": "Hello"}]
        response = await adapter.aresponse(messages)

        # Embeddings
        embedding = await adapter.aembed("Hello, world!")
        embeddings = await adapter.aembed_batch(["Text 1", "Text 2"])

    Pure embedding adapter::

        # For embedding-only use cases, deployment_id is still required
        # but can be a placeholder since aresponse() won't be used
        adapter = AzureOpenAIAdapter(
            resource_name="my-openai-resource",
            deployment_id="gpt-4",  # Required by protocol
            embedding_deployment_id="text-embedding-3-small",
        )

        # Only use embedding methods
        embedding = await adapter.aembed("Document text")
    """

    def __init__(
        self,
        api_key: str,
        resource_name: str,
        deployment_id: str,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 30.0,
        embedding_deployment_id: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        """Initialize Azure OpenAI adapter.

        Args
        ----
            api_key: Azure OpenAI API key
            resource_name: Azure OpenAI resource name
            deployment_id: Azure deployment name for text generation
            api_version: API version (default: "2024-02-15-preview")
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: None)
            timeout: Request timeout in seconds (default: 30.0)
            embedding_deployment_id: Azure deployment name for embeddings (optional)
            embedding_dimensions: Embedding dimensionality for text-embedding-3 models (optional)
        """
        self.api_key = api_key
        self.resource_name = resource_name
        self.deployment_id = deployment_id
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.embedding_deployment_id = embedding_deployment_id
        self.embedding_dimensions = embedding_dimensions

        # Construct Azure endpoint
        self.api_base = f"https://{resource_name}.openai.azure.com"

        # Lazy import to avoid dependency issues
        self._client = None

    def _get_client(self) -> AsyncAzureOpenAI:
        """Get or create OpenAI client configured for Azure."""
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                timeout=self.timeout,
            )
        return self._client

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate response from Azure OpenAI.

        Args
        ----
            messages: List of conversation messages

        Returns
        -------
            Generated response text or None if failed
        """
        try:
            client = self._get_client()

            # Convert MessageList to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            response = await client.chat.completions.create(
                model=self.deployment_id,  # Azure uses deployment_id as model
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            # Log error but don't expose details
            print(f"Azure OpenAI error: {e}")
            return None

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with native tool calling support.

        Args
        ----
            messages: Conversation messages
            tools: Tool definitions in OpenAI format
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)

        Returns
        -------
            LLMResponse with content and optional tool calls
        """
        try:
            client = self._get_client()

            # Convert MessageList to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            response = await client.chat.completions.create(
                model=self.deployment_id,
                messages=openai_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Extract tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=(
                            tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                        ),
                    )
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        except Exception as e:
            print(f"Azure OpenAI tool calling error: {e}")
            return LLMResponse(
                content=None,
                tool_calls=None,
                finish_reason="error",
            )

    async def ahealth_check(self) -> HealthStatus:
        """Check Azure OpenAI adapter health and connectivity.

        Returns
        -------
            HealthStatus with connectivity and model availability details
        """
        try:
            start_time = time.time()

            # Simple health check with minimal token usage
            test_messages = [Message(role="user", content="Hi")]
            response = await self.aresponse(test_messages)

            latency_ms = (time.time() - start_time) * 1000

            if response:
                return HealthStatus(
                    status="healthy",
                    adapter_name=f"AzureOpenAI[{self.deployment_id}]",
                    latency_ms=latency_ms,
                    details={
                        "resource": self.resource_name,
                        "deployment": self.deployment_id,
                        "api_version": self.api_version,
                    },
                )
            else:
                return HealthStatus(
                    status="unhealthy",
                    adapter_name=f"AzureOpenAI[{self.deployment_id}]",
                    latency_ms=latency_ms,
                    details={"error": "No response from API"},
                )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                adapter_name=f"AzureOpenAI[{self.deployment_id}]",
                latency_ms=0.0,
                details={"error": str(e)},
            )

    # ========== Embedding Methods (SupportsEmbedding Protocol) ==========

    async def aembed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text input.

        Uses the embedding_deployment_id if configured, otherwise raises error.

        Args
        ----
            text: Text string to embed

        Returns
        -------
            List of floats representing the embedding vector

        Raises
        ------
            ValueError: If embedding_deployment_id not configured
        """
        if not self.embedding_deployment_id:
            raise ValueError(
                "embedding_deployment_id must be set to use embedding functionality. "
                "Create adapter with: AzureOpenAIAdapter(..., "
                "embedding_deployment_id='text-embedding-3-small')"
            )

        try:
            client = self._get_client()

            request_params: dict[str, Any] = {
                "model": self.embedding_deployment_id,
                "input": text,
            }

            if self.embedding_dimensions is not None:
                request_params["dimensions"] = self.embedding_dimensions

            response = await client.embeddings.create(**request_params)

            if response.data and len(response.data) > 0:
                return response.data[0].embedding

            return []

        except Exception as e:
            print(f"Azure OpenAI embedding error: {e}")
            raise

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args
        ----
            texts: List of text strings to embed

        Returns
        -------
            List of embedding vectors, one per input text
        """
        if not self.embedding_deployment_id:
            raise ValueError("embedding_deployment_id must be set to use embedding functionality")

        try:
            client = self._get_client()

            request_params: dict[str, Any] = {
                "model": self.embedding_deployment_id,
                "input": texts,
            }

            if self.embedding_dimensions is not None:
                request_params["dimensions"] = self.embedding_dimensions

            response = await client.embeddings.create(**request_params)

            if response.data:
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            return [[] for _ in texts]

        except Exception as e:
            print(f"Azure OpenAI batch embedding error: {e}")
            raise

    async def aembed_image(self, image: ImageInput) -> list[float]:
        """Generate embedding vector for a single image input.

        Note: Azure OpenAI does not currently support image embeddings via the
        embeddings API. This method is included for protocol compliance but will
        raise NotImplementedError.

        Args
        ----
            image: Image to embed (file path, bytes, or base64)

        Raises
        ------
            NotImplementedError: Azure OpenAI doesn't support image embeddings
        """
        raise NotImplementedError(
            "Azure OpenAI does not support image embeddings via the embeddings API. "
            "For multimodal embeddings, consider using vision models with aresponse_with_vision()."
        )

    async def aembed_image_batch(self, images: list[ImageInput]) -> list[list[float]]:
        """Generate embeddings for multiple images efficiently.

        Note: Azure OpenAI does not currently support image embeddings via the
        embeddings API. This method is included for protocol compliance but will
        raise NotImplementedError.

        Args
        ----
            images: List of images to embed

        Raises
        ------
            NotImplementedError: Azure OpenAI doesn't support image embeddings
        """
        raise NotImplementedError(
            "Azure OpenAI does not support image embeddings via the embeddings API"
        )
