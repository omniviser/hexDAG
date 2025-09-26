"""Mock LLM implementation for testing purposes."""

from hexai.core.ports.llm import LLM, MessageList


class MockLLM(LLM):
    """Mock implementation of the LLM interface for testing.

    The LLM port interface is stateless, but this mock provides testing utilities like response
    sequencing and call inspection without violating the port contract.
    """

    def __init__(self, responses: list[str] | str | None = None) -> None:
        """Initialize with responses.

        Args
        ----
            responses: Single response string, list of responses, or None for default
        """
        if responses is None:
            self.responses = ['{"result": "Mock response"}']
        elif isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = responses

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
