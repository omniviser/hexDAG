"""OpenAI adapter for LLM interactions."""

import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from hexai.core.ports.configurable import ConfigurableComponent
from hexai.core.ports.llm import MessageList
from hexai.core.registry import adapter
from hexai.helpers.secrets import Secret


@adapter(
    name="openai",
    implements_port="llm",
    namespace="core",
    description="OpenAI GPT adapter for language model interactions",
)
class OpenAIAdapter(ConfigurableComponent):
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
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
        max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
        response_format: Literal["text", "json_object"] = Field(
            default="text", description="Output format (text or json_object)"
        )
        seed: int | None = Field(default=None, description="Seed for deterministic generation")
        top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
        frequency_penalty: float = Field(
            default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
        )
        presence_penalty: float = Field(
            default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
        )
        system_prompt: str | None = Field(default=None, description="Default system prompt")
        timeout: float = Field(default=60.0, gt=0, description="Request timeout in seconds")
        max_retries: int = Field(default=2, ge=0, description="Maximum retry attempts")

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return configuration schema."""
        return cls.Config

    def __init__(self, **kwargs: Any):
        """Initialize OpenAI adapter.

        Args
        ----
            **kwargs: Configuration options (api_key, model, temperature, etc.)
        """
        # Create config from kwargs using the Config schema
        config_data = {}
        for field_name in self.Config.model_fields:
            if field_name in kwargs:
                config_data[field_name] = kwargs[field_name]

        # Create and validate config
        config = self.Config(**config_data)

        # Store configuration
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.response_format = config.response_format
        self.seed = config.seed
        self.top_p = config.top_p
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.stop_sequences = kwargs.get("stop_sequences")  # Not in config schema
        self.system_prompt = config.system_prompt

        # Get API key
        api_key_str = config.api_key
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
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if "organization" in kwargs:
            client_kwargs["organization"] = kwargs["organization"]
        if "base_url" in kwargs:
            client_kwargs["base_url"] = kwargs["base_url"]

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

            if self.stop_sequences:
                request_params["stop"] = self.stop_sequences

            # Add response format if JSON mode is requested
            if self.response_format == "json_object":
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
            logging.error(f"OpenAI API error: {e}")
            return None
