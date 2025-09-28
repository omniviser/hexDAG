"""Port interface definitions for Large Language Models (LLMs)."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from hexai.core.registry.decorators import port


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


MessageList = list[Message]


@port(
    name="llm",
    namespace="core",
)
@runtime_checkable
class LLM(Protocol):
    """Port interface for Large Language Models (LLMs).


    LLMs provide natural language generation capabilities. Implementations
    may use various backends (OpenAI, Anthropic, local models, etc.) but
    must provide the aresponse method for generating text from messages.
    """

    @abstractmethod
    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response from a list of messages (async).

        Args
        ----
            messages: List of role-message dicts, e.g. [{"role": "user", "content": "..."}]

        Returns
        -------
            The generated response as a string, or None if failed.
        """
        pass
