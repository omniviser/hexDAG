"""Mock LLM implementation for testing purposes."""

import asyncio

from hexai.adapters.configs import MockLLMConfig
from hexai.core.ports.llm import LLM, MessageList
from hexai.core.registry import adapter


@adapter(implements_port="llm")
class MockLLM(LLM):
    """Mock implementation of the LLM interface for testing.

    The LLM port interface is stateless, but this mock provides testing utilities like response
    sequencing and call inspection without violating the port contract.
    """

    def __init__(
        self, config: MockLLMConfig | None = None, responses: list[str] | str | None = None
    ) -> None:
        """Initialize with configuration.

        Args
        ----
            config: Configuration for the mock LLM
            responses: Optional responses to override config
        """
        if config is None:
            config = MockLLMConfig()

        # Use provided responses or fall back to config
        if responses is not None:
            if isinstance(responses, str):
                self.responses = [responses]
            else:
                self.responses = responses
        elif config.responses is None:
            self.responses = ['{"result": "Mock response"}']
        elif isinstance(config.responses, str):
            self.responses = [config.responses]
        else:
            self.responses = config.responses

        self.delay_seconds = config.delay_seconds
        self.call_count = 0
        self.last_messages: MessageList | None = None
        self.should_raise = False

    async def aresponse(self, messages: MessageList) -> str | None:
        """Return a response based on the configured responses."""
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
