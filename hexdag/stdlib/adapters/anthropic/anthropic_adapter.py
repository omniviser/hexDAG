"""Anthropic adapter for LLM interactions."""

import os
from typing import Any, Literal

from anthropic import AsyncAnthropic

from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    LLM,
    MessageList,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsUsageTracking,
    TokenUsage,
)
from hexdag.kernel.types import (
    PositiveInt,
    RetryCount,
    Temperature01,
    TimeoutSeconds,
    TopP,
)

logger = get_logger(__name__)

# Convention: Anthropic model options for dropdown menus in Studio UI
AnthropicModel = Literal[
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class AnthropicAdapter(LLM, SupportsGeneration, SupportsFunctionCalling, SupportsUsageTracking):
    """Anthropic implementation of the LLM port.

    This adapter provides integration with Anthropic's Claude models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and Anthropic's format.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit parameter: AnthropicAdapter(api_key="sk-...")
    2. Environment variable: ANTHROPIC_API_KEY
    3. Memory port (orchestrator): secret:ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: AnthropicModel = "claude-3-5-sonnet-20241022",
        temperature: Temperature01 = 0.7,
        max_tokens: PositiveInt = 4096,
        top_p: TopP = 1.0,
        top_k: PositiveInt | None = None,
        system_prompt: str | None = None,
        timeout: TimeoutSeconds = 60.0,
        max_retries: RetryCount = 2,
        **kwargs: Any,  # ← For extra params like base_url
    ):
        """Initialize Anthropic adapter.

        Parameters
        ----------
        api_key : str | None
            Anthropic API key (auto-resolved from ANTHROPIC_API_KEY env var if not provided)
        model : str, default="claude-3-5-sonnet-20241022"
            Claude model to use
        temperature : float, default=0.7
            Sampling temperature (0-1)
        max_tokens : int, default=4096
            Maximum tokens in response
        top_p : float, default=1.0
            Nucleus sampling parameter
        top_k : int | None, default=None
            Top-k sampling parameter
        system_prompt : str | None, default=None
            System prompt to use
        timeout : float, default=60.0
            Request timeout in seconds
        max_retries : int, default=2
            Maximum retry attempts
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("api_key required (pass directly or set ANTHROPIC_API_KEY)")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self._extra_kwargs = kwargs  # Store extra params

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }

        if base_url := kwargs.get("base_url"):
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(**client_kwargs)
        self._last_usage: TokenUsage | None = None

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connection pool resources."""
        await self.client.close()

    def get_last_usage(self) -> TokenUsage | None:
        """Return token usage from the most recent LLM API call."""
        return self._last_usage

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using Anthropic's API.

        Args
        ----
            messages: List of Message objects with role and content

        Returns
        -------
            The generated response text, or None if failed
        """
        try:
            # Anthropic requires system messages to be separate
            system_message = self.system_prompt
            anthropic_messages = []

            for msg in messages:
                if msg.role == "system":
                    # Concatenate multiple system messages if present
                    if system_message:
                        system_message += "\n" + msg.content
                    else:
                        system_message = msg.content
                else:
                    anthropic_messages.append({"role": msg.role, "content": msg.content})

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            if system_message is not None:
                request_params["system"] = system_message

            if self.top_k is not None:
                request_params["top_k"] = self.top_k

            if stop_sequences := self._extra_kwargs.get("stop_sequences"):
                request_params["stop_sequences"] = stop_sequences

            response = await self.client.messages.create(**request_params)

            # Capture token usage
            self._last_usage = None
            if response.usage:
                self._last_usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )

            if response.content and len(response.content) > 0:
                first_content = response.content[0]
                if hasattr(first_content, "text"):
                    return str(first_content.text)

            logger.warning("No text content in Anthropic response")
            return None

        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return None
