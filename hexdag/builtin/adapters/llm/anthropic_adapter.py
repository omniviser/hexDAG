"""Anthropic adapter for LLM interactions."""

from typing import Any

from anthropic import AsyncAnthropic
from pydantic import Field, SecretStr

from hexdag.core.configurable import AdapterConfig, ConfigurableAdapter, SecretField
from hexdag.core.logging import get_logger
from hexdag.core.ports.llm import MessageList
from hexdag.core.registry import adapter
from hexdag.core.types import (
    PositiveInt,
    RetryCount,
    Temperature01,
    TimeoutSeconds,
    TopP,
)

logger = get_logger(__name__)


@adapter(
    name="anthropic",
    implements_port="llm",
    namespace="core",
    description="Anthropic Claude adapter for language model interactions",
)
class AnthropicAdapter(ConfigurableAdapter):
    """Anthropic implementation of the LLM port.

    This adapter provides integration with Anthropic's Claude models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and Anthropic's format.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit config: AnthropicAdapter(api_key="sk-...")
    2. Environment variable: ANTHROPIC_API_KEY
    3. Memory port (orchestrator): secret:ANTHROPIC_API_KEY

    The API key is automatically hidden in logs and repr using Pydantic SecretStr.
    """

    # Configuration schema for TOML generation
    class Config(AdapterConfig):
        """Configuration schema for Anthropic adapter."""

        # Secret field - auto-resolved from env/memory, auto-hidden in logs
        api_key: SecretStr | None = SecretField(
            env_var="ANTHROPIC_API_KEY", description="Anthropic API key (auto-hidden in logs)"
        )

        # Regular configuration fields
        model: str = Field(default="claude-3-5-sonnet-20241022", description="Claude model to use")
        temperature: Temperature01 = 0.7
        max_tokens: PositiveInt = 4096
        top_p: TopP = 1.0
        top_k: PositiveInt | None = None
        system_prompt: str | None = None
        timeout: TimeoutSeconds = 60.0
        max_retries: RetryCount = 2

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(self, **kwargs: Any):
        """Initialize Anthropic adapter.

        Args
        ----
            **kwargs: Configuration options (api_key, model, temperature, etc.)
                     Secrets are auto-resolved from environment or memory port.
        """
        # Initialize config - secrets are auto-resolved in super().__init__()
        super().__init__(**kwargs)

        # Extract API key from config (already resolved)
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Provide via:\n"
                "1. api_key parameter\n"
                "2. ANTHROPIC_API_KEY environment variable\n"
                "3. Memory port (secret:ANTHROPIC_API_KEY) from orchestrator"
            )

        # Initialize Anthropic client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        # Add base_url if provided as extra kwarg
        if base_url := self.get_extra_kwarg("base_url"):
            client_kwargs["base_url"] = base_url

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
            system_message = self.config.system_prompt
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

            # Build request parameters
            request_params: dict[str, Any] = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

            # Add optional parameters
            if system_message is not None:
                request_params["system"] = system_message

            if self.config.top_k is not None:
                request_params["top_k"] = self.config.top_k

            if stop_sequences := self.get_extra_kwarg("stop_sequences"):
                request_params["stop_sequences"] = stop_sequences

            response = await self.client.messages.create(**request_params)

            # Extract content from response
            if response.content and len(response.content) > 0:
                # Get the first text content block
                first_content = response.content[0]
                if hasattr(first_content, "text"):
                    return str(first_content.text)

            logger.warning("No text content in Anthropic response")
            return None

        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return None
