"""Ollama adapter for hexDAG LLM port.

Uses Ollama's OpenAI-compatible chat completions API to call local models
(e.g. ``qwen3.5:9b``, ``llama3``) running on the same machine or network.

Examples
--------
YAML configuration::

    spec:
      ports:
        llm:
          adapter: hexdag_plugins.ollama.OllamaAdapter
          config:
            model: qwen3.5:9b
            base_url: http://localhost:11434
"""

import os
from typing import Any

import httpx
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    LLM,
    LLMResponse,
    MessageList,
    SupportsFunctionCalling,
    SupportsGeneration,
    ToolCall,
)

logger = get_logger(__name__)


class OllamaAdapter(LLM, SupportsGeneration, SupportsFunctionCalling):
    """Ollama adapter implementing the hexDAG LLM port.

    Talks to Ollama's OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``qwen3.5:9b``, ``llama3``).
    base_url : str | None
        Ollama server URL. Falls back to ``OLLAMA_BASE_URL`` env var,
        then ``http://localhost:11434``.
    temperature : float
        Sampling temperature (0.0–2.0).
    max_tokens : int
        Maximum output tokens.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_url(self, path: str = "/v1/chat/completions") -> str:
        return f"{self.base_url}{path}"

    def _convert_messages(self, messages: MessageList) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using Ollama.

        Args:
            messages: List of Message objects with role and content.

        Returns:
            The generated response text, or ``None`` if failed.
        """
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }

            url = self._build_url()
            logger.info("Ollama request to %s", self.model)

            response = await self._get_client().post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content")

            logger.warning("No content in Ollama response")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(
                "Ollama HTTP error: %d - %s",
                e.response.status_code,
                e.response.text,
            )
            return None
        except Exception as e:
            logger.error("Ollama error: %s", e, exc_info=True)
            return None

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with tool calling via Ollama.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (OpenAI format).
            tool_choice: Tool selection strategy.

        Returns:
            LLMResponse with content and tool calls.
        """
        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
                "tools": tools,
            }

            if isinstance(tool_choice, str) and tool_choice != "auto":
                payload["tool_choice"] = tool_choice

            url = self._build_url()

            response = await self._get_client().post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices:
                return LLMResponse(content=None, tool_calls=None)

            message = choices[0].get("message", {})
            text_content = message.get("content")
            finish_reason = choices[0].get("finish_reason")

            tool_calls = []
            for tc in message.get("tool_calls", []):
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                    )
                )

            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=finish_reason,
            )

        except Exception as e:
            logger.error("Ollama error with tools: %s", e, exc_info=True)
            raise
