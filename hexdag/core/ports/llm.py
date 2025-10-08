"""Port interface definitions for Large Language Models (LLMs)."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from hexdag.core.registry.decorators import port

if TYPE_CHECKING:
    from hexdag.core.ports.healthcheck import HealthStatus


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

    Optional Methods
    ----------------
    Adapters may optionally implement:
    - ahealth_check(): Verify LLM API connectivity and availability
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

    async def ahealth_check(self) -> "HealthStatus":
        """Check LLM adapter health and connectivity (optional).

        Adapters should verify:
        - API connectivity to the LLM service
        - Model availability
        - Authentication status
        - Rate limit status (if applicable)

        This method is optional. If not implemented, the adapter will be
        considered healthy by default.

        Returns
        -------
        HealthStatus
            Current health status with details about connectivity and availability

        Examples
        --------
        OpenAI adapter health check::

            status = await openai_adapter.ahealth_check()
            status.status  # "healthy", "degraded", or "unhealthy"
            status.latency_ms  # Time taken for health check
            status.details  # {"model": "gpt-4", "rate_limit_remaining": 100}
        """
        ...
