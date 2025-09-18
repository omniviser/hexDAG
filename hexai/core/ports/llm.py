"""Port interface definitions for Large Language Models (LLMs)."""

from typing import Protocol

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


MessageList = list[Message]


class LLM(Protocol):
    """Port interface for Large Language Models (LLMs)."""

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
