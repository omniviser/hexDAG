"""Anthropic adapter for LLM interactions."""

from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from hexai.core.logging import get_logger
from hexai.core.ports.configurable import ConfigurableComponent
from hexai.core.ports.llm import MessageList
from hexai.core.registry import adapter
from hexai.core.types import (
    PositiveInt,
    RetryCount,
    Temperature01,
    TimeoutSeconds,
    TopP,
)
from hexai.helpers.secrets import Secret

logger = get_logger(__name__)


@adapter(
    name="anthropic",
    implements_port="llm",
    namespace="core",
    description="Anthropic Claude adapter for language model interactions",
)
class AnthropicAdapter(ConfigurableComponent):
    """Anthropic implementation of the LLM port.

    This adapter provides integration with Anthropic's Claude models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and Anthropic's format.
    """

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for Anthropic adapter."""

        api_key: str | None = Field(
            default=None, description="Anthropic API key (or use ANTHROPIC_API_KEY env var)"
        )
        model: str = Field(default="claude-3-5-sonnet-20241022", description="Claude model to use")
        temperature: Temperature01 = 0.7
        max_tokens: PositiveInt = 4096
        top_p: TopP = 1.0
        top_k: PositiveInt | None = None
        system_prompt: str | None = None
        timeout: TimeoutSeconds = 60.0
        max_retries: RetryCount = 2

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return configuration schema."""
        return cls.Config

    def __init__(self, **kwargs: Any):
        """Initialize Anthropic adapter.

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
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.system_prompt = config.system_prompt
        self.stop_sequences = kwargs.get("stop_sequences")  # Not in config schema

        # Get API key
        api_key_str = config.api_key
        if not api_key_str:
            try:
                api_secret = Secret.retrieve_secret_from_env("ANTHROPIC_API_KEY")
                api_key_str = api_secret.get()
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Anthropic API key must be provided either as parameter or "
                    f"through ANTHROPIC_API_KEY environment variable: {e}"
                ) from e

        # Initialize Anthropic client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key_str,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if "base_url" in kwargs:
            client_kwargs["base_url"] = kwargs["base_url"]

        self.client = AsyncAnthropic(**client_kwargs)

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
            # Convert MessageList to Anthropic format
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
                    # Convert "user" and "assistant" messages
                    anthropic_messages.append({"role": msg.role, "content": msg.content})

            # Make API call
            # Build request parameters
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            # Add optional parameters
            if system_message is not None:
                request_params["system"] = system_message

            if self.top_k is not None:
                request_params["top_k"] = self.top_k

            if self.stop_sequences:
                request_params["stop_sequences"] = self.stop_sequences

            response = await self.client.messages.create(**request_params)

            # Extract content from response
            if response.content and len(response.content) > 0:
                # Get the first text content block
                first_content = response.content[0]
                if hasattr(first_content, "text"):
                    return str(first_content.text)

            return None

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
