"""OpenAI adapter for LLM interactions."""

import json
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from hexai.core.configurable import ConfigurableAdapter
from hexai.core.logging import get_logger
from hexai.core.ports.llm import MessageList
from hexai.core.registry import adapter
from hexai.core.types import (
    FrequencyPenalty,
    PresencePenalty,
    RetryCount,
    Temperature02,
    TimeoutSeconds,
    TokenCount,
    TopP,
)
from hexai.helpers.secrets import Secret

logger = get_logger(__name__)


@adapter(
    name="openai",
    implements_port="llm",
    namespace="core",
    description="OpenAI GPT adapter for language model interactions",
)
class OpenAIAdapter(ConfigurableAdapter):
    """OpenAI implementation of the LLM port.

    This adapter provides integration with OpenAI's GPT models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and OpenAI's format.
    """

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for OpenAI adapter."""

        api_key: str | None = Field(
            default=None, description="OpenAI API key (or use OPENAI_API_KEY env var)"
        )
        model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
        temperature: Temperature02 = 0.7
        max_tokens: TokenCount | None = None
        response_format: Literal["text", "json_object"] = Field(
            default="text", description="Output format (text or json_object)"
        )
        seed: int | None = None
        top_p: TopP = 1.0
        frequency_penalty: FrequencyPenalty = 0.0
        presence_penalty: PresencePenalty = 0.0
        system_prompt: str | None = None
        timeout: TimeoutSeconds = 60.0
        max_retries: RetryCount = 2

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(self, **kwargs: Any):
        """Initialize OpenAI adapter.

        Args
        ----
            **kwargs: Configuration options (api_key, model, temperature, etc.)
        """
        # Initialize config (accessible via self.config.field_name)
        super().__init__(**kwargs)

        # Get API key (from config or environment)
        api_key_str = self.config.api_key
        if not api_key_str:
            try:
                api_secret = Secret.retrieve_secret_from_env("OPENAI_API_KEY")
                api_key_str = api_secret.get()
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"OpenAI API key must be provided either as parameter or "
                    f"through OPENAI_API_KEY environment variable: {e}"
                ) from e

        # Initialize OpenAI client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key_str,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        # Add extra kwargs (organization, base_url) not in config schema
        if org := self.get_extra_kwarg("organization"):
            client_kwargs["organization"] = org
        if base_url := self.get_extra_kwarg("base_url"):
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
            if self.config.system_prompt and not any(
                msg["role"] == "system" for msg in openai_messages
            ):
                openai_messages.insert(0, {"role": "system", "content": self.config.system_prompt})

            # Build request parameters with modern API format
            request_params: dict[str, Any] = {
                "model": self.config.model,
                "messages": openai_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
            }

            # Add optional parameters only if set
            if self.config.max_tokens is not None:
                request_params["max_tokens"] = self.config.max_tokens

            if self.config.seed is not None:
                request_params["seed"] = self.config.seed

            # Stop sequences from extra kwargs (not in config schema)
            if stop_seq := self.get_extra_kwarg("stop_sequences"):
                request_params["stop"] = stop_seq

            # Add response format if JSON mode is requested
            if self.config.response_format == "json_object":
                request_params["response_format"] = {"type": "json_object"}

            # Make API call with modern format
            response = await self.client.chat.completions.create(**request_params)

            # Extract content from response with better error handling
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message.content:
                    return str(message.content)

                # Handle function calls or tool calls if present (for future extensibility)
                if (
                    hasattr(message, "tool_calls")
                    and message.tool_calls
                    and len(message.tool_calls) > 0
                ):
                    # Return tool call information as JSON string
                    return json.dumps([
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ])

            return None

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
