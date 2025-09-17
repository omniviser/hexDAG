"""Anthropic adapter for LLM interactions."""

import os
from typing import Any

from anthropic import AsyncAnthropic

from hexai.core.ports.llm import MessageList
from hexai.core.registry import adapter


@adapter(
    name="anthropic",
    implements_port="llm",
    namespace="adapters",
    description="Anthropic Claude adapter for language model interactions",
)
class AnthropicAdapter:
    """Anthropic implementation of the LLM port.

    This adapter provides integration with Anthropic's Claude models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and Anthropic's format.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ):
        """Initialize Anthropic adapter.

        Args:
            model: The Claude model to use (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env variable
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to Anthropic client
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Anthropic client
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key must be provided either as parameter or "
                "through ANTHROPIC_API_KEY environment variable"
            )

        self.client = AsyncAnthropic(api_key=api_key, **kwargs)

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using Anthropic's API.

        Args:
            messages: List of Message objects with role and content

        Returns:
            The generated response text, or None if failed
        """
        try:
            # Convert MessageList to Anthropic format
            # Anthropic requires system messages to be separate
            system_message = None
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
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if system_message is not None:
                kwargs["system"] = system_message

            response = await self.client.messages.create(**kwargs)

            # Extract content from response
            if response.content and len(response.content) > 0:
                # Get the first text content block
                first_content = response.content[0]
                if hasattr(first_content, "text"):
                    return str(first_content.text)

            return None

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None
