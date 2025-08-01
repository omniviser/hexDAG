"""LLM Adapter implementations for the hexAI framework.

This module provides adapter classes that implement the LLM port interface, allowing different LLM
implementations to be used with the framework.
"""

import asyncio
import logging
from typing import Any

from hexai.app.ports.llm import LLM, MessageList

logger = logging.getLogger(__name__)


class LLMFactoryAdapter(LLM):
    """Base adapter class for LLM implementations.

    This adapter wraps various LLM implementations to conform to the LLM protocol.
    """

    def __init__(self, model: Any) -> None:
        """Initialize the adapter with an underlying model.

        Args
        ----
            model: The underlying LLM model implementation
        """
        self.model = model

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using the underlying model.

        Implements the LLM protocol's aresponse method.

        Args
        ----
            messages: List of message dictionaries with 'role' and 'content'

        Returns
        -------
            Generated response string or None if generation fails
        """
        try:
            if hasattr(self.model, "aresponse") and asyncio.iscoroutinefunction(
                self.model.aresponse
            ):
                return await self.model.aresponse(messages)

            elif hasattr(self.model, "response"):
                result = await asyncio.to_thread(self.model.response, messages)
                return str(result) if result is not None else None

            elif callable(self.model):
                result = await asyncio.to_thread(self.model, messages)
                return str(result) if result is not None else None

            else:
                raise AttributeError(
                    f"Model {type(self.model).__name__} does not have a compatible response method"
                )

        except Exception as e:
            logger.error(f"Error in LLM adapter: {e}")
            return None
