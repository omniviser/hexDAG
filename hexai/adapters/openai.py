"""OpenAI adapter for hexai framework."""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Optional

try:
    from hexai.helpers.secrets import get_secret as _get_secret
except Exception:  # fallback

    # TODO: replace with hexai.helpers.secrets.get_secret
    # once Wojtek's secret handler is merged into main

    def _get_secret(key: str, default: Optional[str] = None, required: bool = False) -> str:
        val = os.getenv(key, default)
        if required and val is None:
            raise RuntimeError(f"Missing required secret: {key}")
        return val or ""


SupportedProvider = Literal["openai", "azure", "anthropic", "gemini", "ollama"]


class OpenAIAdapter:
    """OpenAI-compatible adapter (skeleton)."""

    def __init__(
        self,
        *,
        provider: SupportedProvider = "openai",
        model: Optional[str] = None,
        secrets_provider: Callable[[str, Optional[str], bool], str] = _get_secret,
        **client_kwargs: Any,
    ) -> None:
        self.provider = provider
        self.model = model
        self._get_secret = secrets_provider
        self._client_kwargs = client_kwargs
        self._client = None
