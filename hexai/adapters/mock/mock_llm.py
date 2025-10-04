"""Mock LLM implementation for testing purposes."""

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import Field

from hexai.core.configurable import AdapterConfig, ConfigurableAdapter
from hexai.core.ports.llm import LLM, MessageList
from hexai.core.registry import adapter

if TYPE_CHECKING:
    from hexai.core.ports.healthcheck import HealthStatus


@adapter(implements_port="llm")
class MockLLM(LLM, ConfigurableAdapter):
    """Mock implementation of the LLM interface for testing.

    The LLM port interface is stateless, but this mock provides testing utilities like response
    sequencing and call inspection without violating the port contract.
    """

    # Configuration schema for TOML generation
    class Config(AdapterConfig):
        """Configuration schema for Mock LLM adapter."""

        responses: list[str] | str | None = Field(
            default=None,
            description="Single response string, list of responses, or None for default",
        )
        delay_seconds: float = Field(
            default=0.0, ge=0.0, description="Artificial delay to simulate API latency"
        )

    # Type hint for mypy to understand self.config has Config fields
    config: Config

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with configuration.

        Args
        ----
            **kwargs: Configuration options (responses, delay_seconds)
        """
        # Initialize config (accessible via self.config.field_name)
        # Note: Explicitly call ConfigurableAdapter.__init__ because Protocol blocks super()
        ConfigurableAdapter.__init__(self, **kwargs)

        # Process responses (convert to list if needed)
        responses = kwargs.get("responses", self.config.responses)
        if responses is not None:
            if isinstance(responses, str):
                self.responses = [responses]
            else:
                self.responses = responses
        else:
            self.responses = ['{"result": "Mock response"}']

        # Non-config state
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
        if self.config.delay_seconds > 0:
            await asyncio.sleep(self.config.delay_seconds)

        if self.should_raise:
            raise Exception("Mock LLM error for testing")

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]  # Repeat last response

        self.call_count += 1
        return response

    async def ahealth_check(self) -> "HealthStatus":
        """Health check for Mock LLM (always healthy)."""
        from hexai.core.ports.healthcheck import HealthStatus

        return HealthStatus(
            status="healthy",
            adapter_name="MockLLM",
            latency_ms=0.1,
        )

    # Testing utilities (not part of the LLM port interface)
    def reset(self) -> None:
        """Reset the mock state for testing."""
        self.call_count = 0
        self.last_messages = None
