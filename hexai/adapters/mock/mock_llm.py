"""Mock LLM implementation for testing purposes."""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from hexai.core.ports.configurable import ConfigurableComponent
from hexai.core.ports.llm import LLM, MessageList
from hexai.core.registry import adapter


@adapter(implements_port="llm")
class MockLLM(LLM, ConfigurableComponent):
    """Mock implementation of the LLM interface for testing.

    The LLM port interface is stateless, but this mock provides testing utilities like response
    sequencing and call inspection without violating the port contract.
    """

    # Configuration schema for TOML generation
    class Config(BaseModel):
        """Configuration schema for Mock LLM adapter."""

        responses: list[str] | str | None = Field(
            default=None,
            description="Single response string, list of responses, or None for default",
        )
        delay_seconds: float = Field(
            default=0.0, ge=0.0, description="Artificial delay to simulate API latency"
        )

    @classmethod
    def get_config_class(cls) -> type[BaseModel]:
        """Return configuration schema."""
        return cls.Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration.

        Args
        ----
            **kwargs: Configuration options (responses, delay_seconds)
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

        # Process responses
        responses = kwargs.get("responses", config.responses)
        if responses is not None:
            if isinstance(responses, str):
                self.responses = [responses]
            else:
                self.responses = responses
        else:
            self.responses = ['{"result": "Mock response"}']

        self.delay_seconds = config.delay_seconds
        self.call_count = 0
        self.last_messages: MessageList | None = None
        self.should_raise = False

    async def aresponse(self, messages: MessageList) -> str | None:
        """Return a response based on the configured responses.

        Parameters
        ----------
        messages : MessageList
            List of messages to process

        Returns
        -------
        str | None
            Mock response string or None

        Raises
        ------
        Exception
            When should_raise is True for testing error conditions
        """
        self.last_messages = messages

        # Simulate delay if configured
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.should_raise:
            raise Exception("Mock LLM error for testing")

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]  # Repeat last response

        self.call_count += 1
        return response

    # Testing utilities (not part of the LLM port interface)
    def reset(self) -> None:
        """Reset the mock state for testing."""
        self.call_count = 0
        self.last_messages = None
