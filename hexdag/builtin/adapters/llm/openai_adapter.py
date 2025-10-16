"""OpenAI adapter for LLM interactions."""

from typing import Any, Literal

from openai import AsyncOpenAI

from hexdag.core.logging import get_logger
from hexdag.core.ports.llm import MessageList
from hexdag.core.registry import adapter
from hexdag.core.types import (
    FrequencyPenalty,
    PresencePenalty,
    RetryCount,
    Temperature02,
    TimeoutSeconds,
    TokenCount,
    TopP,
)

logger = get_logger(__name__)


@adapter(
    name="openai",
    implements_port="llm",
    namespace="core",
    description="OpenAI GPT adapter for language model interactions",
    secrets={"api_key": "OPENAI_API_KEY"},
)
class OpenAIAdapter:
    """OpenAI implementation of the LLM port.

    This adapter provides integration with OpenAI's GPT models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and OpenAI's format.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit parameter: OpenAIAdapter(api_key="sk-...")
    2. Environment variable: OPENAI_API_KEY
    3. Memory port (orchestrator): secret:OPENAI_API_KEY
    """

    def __init__(
        self,
        api_key: str,  # ← Auto-resolved by @adapter decorator
        model: str = "gpt-4o-mini",
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
        **kwargs: Any,  # ← For extra params like organization, base_url
    ):
        """Initialize OpenAI adapter.

        Parameters
        ----------
        api_key : str
            OpenAI API key (auto-resolved from OPENAI_API_KEY env var)
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
        """
        # Store configuration
        self.api_key = api_key
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
        self._extra_kwargs = kwargs  # Store extra params

        # Initialize OpenAI client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        # Add extra kwargs (organization, base_url)
        if org := kwargs.get("organization"):
            client_kwargs["organization"] = org
        if base_url := kwargs.get("base_url"):
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

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
            # Convert MessageList to OpenAI's message format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Add system prompt if configured
            if self.system_prompt and not any(msg["role"] == "system" for msg in openai_messages):
                openai_messages.insert(0, {"role": "system", "content": self.system_prompt})

            # Build request parameters with modern API format
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }

            # Add optional parameters only if set
            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.seed is not None:
                request_params["seed"] = self.seed

            # Stop sequences from extra kwargs
            if stop_seq := self._extra_kwargs.get("stop_sequences"):
                request_params["stop"] = stop_seq

            # Handle response_format for structured output
            if self.response_format == "json_object":
                request_params["response_format"] = {"type": "json_object"}

            # Make API call with modern format
            response = await self.client.chat.completions.create(**request_params)

            # Extract content from response with better error handling
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content: str = str(message.content)

                    return content

            logger.warning("No content in OpenAI response")
            return None

        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return None
