"""OpenAI adapter for LLM interactions."""

from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import Field, SecretStr

from hexdag.core.configurable import AdapterConfig, ConfigurableAdapter, SecretField
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
)
class OpenAIAdapter(ConfigurableAdapter):
    """OpenAI implementation of the LLM port.

    This adapter provides integration with OpenAI's GPT models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and OpenAI's format.

    Secret Management
    -----------------
    API key resolution order:
    1. Explicit config: OpenAIAdapter(api_key="sk-...")
    2. Environment variable: OPENAI_API_KEY
    3. Memory port (orchestrator): secret:OPENAI_API_KEY

    The API key is automatically hidden in logs and repr using Pydantic SecretStr.
    """

    # Configuration schema for TOML generation
    class Config(AdapterConfig):
        """Configuration schema for OpenAI adapter."""

        # Secret field - auto-resolved from env/memory, auto-hidden in logs
        api_key: SecretStr | None = SecretField(
            env_var="OPENAI_API_KEY", description="OpenAI API key (auto-hidden in logs)"
        )

        # Regular configuration fields
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
                     Secrets are auto-resolved from environment or memory port.
        """
        # Initialize config - secrets are auto-resolved in super().__init__()
        super().__init__(**kwargs)

        # Extract API key from config (already resolved)
        api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Provide via:\n"
                "1. api_key parameter\n"
                "2. OPENAI_API_KEY environment variable\n"
                "3. Memory port (secret:OPENAI_API_KEY) from orchestrator"
            )

        # Initialize OpenAI client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
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

            # Handle response_format for structured output
            if self.config.response_format == "json_object":
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
