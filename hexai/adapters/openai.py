"""OpenAI adapter for hexai framework."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Callable, Literal, NotRequired, Optional, TypedDict

# Exposed for tests: can be monkeypatched to a fake client
AsyncOpenAI: Any | None = None


class ProviderCompatCfg(TypedDict):
    """Configuration schema for OpenAI-compatible providers."""

    BASE_URL_ENV: str
    API_KEY_ENV: str
    HEADER_KIND: Literal["authorization", "x-api-key"]
    DEFAULT_BASE_URL: Optional[str]


# Names for provider “compat” envs (OpenAI-API compatible gateways)
# e.g. LiteLLM, OpenRouter, custom proxy, or Ollama OpenAI compat server
_PROVIDER_COMPAT: dict[Literal["anthropic", "gemini", "ollama"], ProviderCompatCfg] = {
    "anthropic": {
        "BASE_URL_ENV": "ANTHROPIC_COMPAT_BASE_URL",
        "API_KEY_ENV": "ANTHROPIC_COMPAT_API_KEY",
        "HEADER_KIND": "authorization",
        "DEFAULT_BASE_URL": None,
    },
    "gemini": {
        "BASE_URL_ENV": "GEMINI_COMPAT_BASE_URL",
        "API_KEY_ENV": "GEMINI_COMPAT_API_KEY",
        "HEADER_KIND": "authorization",
        "DEFAULT_BASE_URL": None,
    },
    "ollama": {
        "BASE_URL_ENV": "OLLAMA_COMPAT_BASE_URL",
        "API_KEY_ENV": "OLLAMA_COMPAT_API_KEY",
        "HEADER_KIND": "authorization",
        "DEFAULT_BASE_URL": "http://localhost:11434/v1",
    },
}

# --- simple price map ($ per 1K tokens); TODO: adjust models as needed ---
_MODEL_PRICES = {
    # examples
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
}


def _estimate_cost_usd(model: Optional[str], usage: Any) -> Optional[float]:
    """Return estimated USD cost based on usage.{prompt_tokens, completion_tokens}."""
    try:
        if not model or not usage:
            return None
        prices = _MODEL_PRICES.get(model)
        if not prices:
            return None
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        if prompt is None and isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = getattr(usage, "completion_tokens", None) or getattr(
            usage, "output_tokens", None
        )
        if completion is None and isinstance(usage, dict):
            completion = usage.get("completion_tokens") or usage.get("output_tokens")
        prompt = float(prompt or 0)
        completion = float(completion or 0)
        return (prompt / 1000.0) * prices["input"] + (completion / 1000.0) * prices["output"]
    except Exception:
        return None


def _try_parse_structured(
    content: Optional[str], schema: Optional[Callable[..., Any] | type[Any]]
) -> Optional[Any]:
    """
    If `schema` is provided and `content` looks like JSON, try to parse and validate it.

    Tries:
    - Pydantic v2 via TypeAdapter (if available)
    - Fallback: call schema(data) first; if that fails with TypeError and data is a dict,
      try schema(**data).
    """
    if not content or schema is None:
        return None

    # Best-effort JSON parse
    try:
        data = json.loads(content)
    except Exception:
        return None

    # Pydantic v2 path (optional)
    parsed: Any | None = None
    try:
        from pydantic import TypeAdapter  # optional dependency

        adapter: Any = TypeAdapter(schema)
        parsed = adapter.validate_python(data)
    except Exception:
        parsed = None

    if parsed is not None:
        return parsed

    # Generic callable/class fallback
    try:
        if callable(schema):
            # Prefer single-arg call (works for functions expecting dict)
            try:
                return schema(data)
            except TypeError:
                # If the callable expects kwargs (e.g., dataclass/pydantic-like)
                if isinstance(data, dict):
                    return schema(**data)
    except Exception:
        return None

    return None


class ToolCall(TypedDict):
    """Represents a single function/tool call returned by the model."""

    id: str
    name: str
    arguments: str


class GenerateResult(TypedDict, total=False):
    """Normalized return type for non-streaming generate() results."""

    content: Optional[str]
    usage: Any
    cost_usd: NotRequired[float]
    tool_calls: NotRequired[list[ToolCall]]
    structured: NotRequired[Any]
    raw: Any


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


class TokenBucket:
    """Asynchronous token-bucket rate limiter.

    Parameters
    ----------
    rate_per_s : float
        Number of tokens refilled per second.
    capacity : int
        Maximum number of tokens that can be stored in the bucket.
    """

    def __init__(self, rate_per_s: float, capacity: int) -> None:
        self.rate = rate_per_s
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = asyncio.get_running_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> None:
        """Acquire a given number of tokens, waiting if necessary.

        Parameters
        ----------
        cost : int, optional
            Number of tokens required for the operation (default is 1).

        Raises
        ------
        asyncio.CancelledError
            If the waiting coroutine is cancelled before tokens are available.
        """
        async with self._lock:
            now = asyncio.get_running_loop().time()
            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now

            while self.tokens < cost:
                needed = cost - self.tokens
                await asyncio.sleep(max(0.0, needed / self.rate))
                now = asyncio.get_running_loop().time()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now

            self.tokens -= cost


class OpenAIAdapter:
    """OpenAI-compatible adapter with streaming, rate limiting and retries.

    OpenAI-compatible providers:
    - OpenAI via OPENAI_API_KEY (+ optional OPENAI_BASE_URL)
    - Azure via AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
    - Anthropic via ANTHROPIC_COMPAT_BASE_URL (+ ANTHROPIC_COMPAT_API_KEY if required)
    - Gemini via GEMINI_COMPAT_BASE_URL (+ GEMINI_COMPAT_API_KEY if required)
    - Ollama via OLLAMA_COMPAT_BASE_URL (default http://localhost:11434/v1)
      (+ OLLAMA_COMPAT_API_KEY optional)
    """

    def __init__(
        self,
        *,
        provider: SupportedProvider = "openai",
        model: Optional[str] = None,
        secrets_provider: Callable[[str, Optional[str]], str] = _get_secret,
        **client_kwargs: Any,
    ) -> None:
        self.provider = provider
        self.model = model
        self._get_secret = secrets_provider
        self._client_kwargs = client_kwargs
        self._client: Any | None = None  # explicit type for mypy

    def _require_secret(self, key: str) -> str:
        """Return a non-empty secret value or raise a clear error.

        Parameters
        ----------
        key : str
            Environment/config variable name to fetch.

        Returns
        -------
        str
            Non-empty secret value.

        Raises
        ------
        RuntimeError
            If the secret is missing or empty.
        """
        val = self._get_secret(key, None)
        if not val:
            raise RuntimeError(f"Missing required secret: {key}")
        return val

    async def _ensure_client(self) -> None:
        """Lazily initialize the underlying SDK client for the selected provider.

        Notes
        -----
        MVP supports ``openai`` and ``azure`` using the OpenAI Python SDK (async).
        Other providers will be added in subsequent steps.
        """
        # if already initialized -> exit early
        if self._client is not None:
            return None

        # only these providers supported in MVP
        if self.provider not in ("openai", "azure", "anthropic", "gemini", "ollama"):
            raise NotImplementedError(f"Provider '{self.provider}' not implemented yet.")

        # lazy import so unit tests don't need the SDK installed
        # lazy import with test-friendly override
        global AsyncOpenAI
        if AsyncOpenAI is None:
            try:
                from openai import AsyncOpenAI as _RealAsyncOpenAI

                AsyncOpenAI = _RealAsyncOpenAI
            except Exception as e:  # pragma: no cover
                # If tests monkeypatch AsyncOpenAI earlier, they won't hit this branch.
                raise RuntimeError("OpenAI SDK not installed. Add `openai` to dependencies.") from e

        default_headers: dict[str, str] = {}
        api_key: Optional[str] = None
        base_url: Optional[str] = None

        if self.provider == "openai":
            api_key = self._require_secret("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")  # optional
            # standard Bearer header is set by the SDK when api_key is passed

        elif self.provider == "azure":
            api_key = self._require_secret("AZURE_OPENAI_API_KEY")
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT")  # required in Azure setups
            # Azure-compatible gateways often accept Bearer too; SDK will set it

        else:
            cfg = _PROVIDER_COMPAT[self.provider]
            base_url = os.getenv(cfg["BASE_URL_ENV"]) or cfg["DEFAULT_BASE_URL"]
            api_key_opt = os.getenv(cfg["API_KEY_ENV"])
            api_key = api_key_opt or os.getenv("OPENAI_API_KEY")

            if cfg["HEADER_KIND"] == "x-api-key" and api_key:
                default_headers["x-api-key"] = api_key

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
            **self._client_kwargs,
        )
        return None

    async def astream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        **params: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completions as an async iterator (MVP for OpenAI/Azure/compat).

        Parameters
        ----------
        messages : list of dict
            Chat messages in OpenAI format, e.g. [{"role": "user", "content": "..."}].
        model : str, optional
            Model override for this call; falls back to self.model if not provided.
        **params : Any
            Extra parameters forwarded to the provider SDK.

        Yields
        ------
        dict
            Normalized stream deltas, e.g. {"type": "content", "data": "<chunk>"} or
            {"type": "finish", "data": "<finish_reason>"}.
        """
        bucket = TokenBucket(rate_per_s=5, capacity=10)
        await bucket.acquire(1)

        await self._ensure_client()
        if self._client is None:
            raise RuntimeError("SDK client not initialized")
        client = self._client

        async def _call_stream() -> Any:
            return await client.chat.completions.create(
                model=model or self.model,
                messages=list(messages),
                stream=True,
                **params,
            )

        stream = await self._with_retries(_call_stream)

        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]

            delta = getattr(choice, "delta", None)
            if delta is not None and getattr(delta, "content", None):
                yield {"type": "content", "data": delta.content}

            # tool calls (OpenAI: delta.tool_calls[].function.{name, arguments})
            tcs = getattr(delta, "tool_calls", None) if delta is not None else None
            if tcs:
                for tc in tcs:
                    fn = getattr(tc, "function", None)
                    if fn and getattr(fn, "name", None) is not None:
                        yield {
                            "type": "tool_call",
                            "data": {
                                "id": getattr(tc, "id", ""),
                                "name": fn.name,
                                "arguments": getattr(fn, "arguments", "") or "",
                            },
                        }

            finish = getattr(choice, "finish_reason", None)
            if finish:
                yield {"type": "finish", "data": finish}

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        schema: Optional[Callable[..., Any] | type[Any]] = None,
        **params: Any,
    ) -> GenerateResult:
        """Non-streaming chat completion call.

        Parameters
        ----------
        messages : list of dict
            OpenAI-format chat messages, e.g. [{"role": "user", "content": "hi"}].
        model : str, optional
            Model override for this call; falls back to self.model if not provided.
        **params : Any
            Extra provider parameters passed through to the SDK.

        Returns
        -------
        dict
            Dictionary with keys:
            - "content": str | None
            - "usage": Any
            - "raw": Any
        """
        # simple per-call rate limit (MVP)
        bucket = TokenBucket(rate_per_s=5, capacity=10)
        await bucket.acquire(1)

        await self._ensure_client()
        if self._client is None:
            raise RuntimeError("SDK client not initialized")
        client = self._client  # <- zawężamy typ

        async def _call() -> Any:
            return await client.chat.completions.create(
                model=model or self.model,
                messages=list(messages),
                stream=False,
                **params,
            )

        resp = await self._with_retries(_call)

        content = None
        if getattr(resp, "choices", None):
            msg = getattr(resp.choices[0], "message", None)
            content = getattr(msg, "content", None) if msg else None

        usage = getattr(resp, "usage", None)

        # collect tool calls (if any)
        tool_calls: list[ToolCall] = []
        try:
            tool_calls_raw = resp.choices[0].message.tool_calls  # OpenAI SDK shape
        except Exception:
            tool_calls_raw = None

        if tool_calls_raw:
            for tc in tool_calls_raw:
                fn = getattr(tc, "function", None)
                if fn and getattr(fn, "name", None) is not None:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", "") or "",
                            "name": fn.name,
                            "arguments": getattr(fn, "arguments", "") or "",
                        }
                    )

        # build normalized result
        out: GenerateResult = {
            "content": content,
            "usage": usage,
            "raw": resp,
        }
        if tool_calls:
            out["tool_calls"] = tool_calls

        # cost estimation (if price known for model)
        cost = _estimate_cost_usd(model or self.model, usage)
        if cost is not None:
            out["cost_usd"] = cost

        parsed = _try_parse_structured(content, schema)
        if parsed is not None:
            out["structured"] = parsed

        return out

    async def _with_retries(
        self,
        op: Callable[[], Awaitable[Any]],
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
    ) -> Any:
        """Run async operation with exponential backoff + jitter on transient errors.

        Retries:
        - timeouts
        - HTTP 429 / 5xx (if exception exposes `.status` or `.status_code`)
        - textual hints in error message ("rate limit", "temporarily unavailable")

        Parameters
        ----------
        op : callable
            Zero-arg async function that performs the SDK call.
        max_attempts : int
            Maximum number of attempts (including the first one).
        base_delay : float
            Initial backoff in seconds.
        max_delay : float
            Upper bound for backoff.

        Returns
        -------
        Any
            Result of `op()` if it eventually succeeds.

        Raises
        ------
        Exception
            Last error if all attempts fail or error is non-retryable.
        """
        attempt = 0
        while True:
            try:
                return await op()
            except Exception as e:  # noqa: BLE001 - we normalize retryability below
                attempt += 1

                # Decide if retryable
                status = getattr(e, "status", None) or getattr(e, "status_code", None)
                msg = str(e).lower()
                retryable = (
                    isinstance(e, asyncio.TimeoutError)
                    or status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                    or "rate limit" in msg
                    or "temporarily unavailable" in msg
                    or "timeout" in msg
                )

                if not retryable or attempt >= max_attempts:
                    raise RuntimeError(f"Request failed after {attempt} attempt(s): {e}") from e

                # exponential backoff + jitter
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                jitter = delay * 0.3
                sleep_s = max(0.0, delay - jitter) + (
                    jitter * (os.getpid() % 10) / 10.0
                )  # deterministic-ish
                await asyncio.sleep(sleep_s)
