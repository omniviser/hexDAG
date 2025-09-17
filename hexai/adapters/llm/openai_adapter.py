"""OpenAI adapter for LLM interactions."""

import os
from typing import Any

from openai import AsyncOpenAI

from hexai.core.ports.llm import MessageList
from hexai.core.registry import adapter


@adapter(
    name="openai",
    implements_port="llm",
    namespace="core",
    description="OpenAI GPT adapter for language model interactions",
)
class OpenAIAdapter:
    """OpenAI implementation of the LLM port.

    This adapter provides integration with OpenAI's GPT models through
    their API. It supports async operations and handles message conversion
    between hexDAG's format and OpenAI's format.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ):
        """Initialize OpenAI adapter.

        Args:
            model: The OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env variable
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to OpenAI client
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either as parameter or "
                "through OPENAI_API_KEY environment variable"
            )

        self.client = AsyncOpenAI(api_key=api_key, **kwargs)

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using OpenAI's API.

        Args:
            messages: List of Message objects with role and content

        Returns:
            The generated response text, or None if failed
        """
        try:
            # Convert MessageList to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract content from response
            if response.choices and response.choices[0].message.content:
                return str(response.choices[0].message.content)

            return None

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
