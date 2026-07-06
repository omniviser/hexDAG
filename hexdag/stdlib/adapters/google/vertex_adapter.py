"""Google Gemini adapter for hexDAG LLM port.

Uses the Vertex AI REST API (``aiplatform.googleapis.com``) with two
authentication modes:

  - **Service account (ADC)** [primary]: Project-scoped endpoint with Bearer
    token. Set GOOGLE_APPLICATION_CREDENTIALS to a service account key
    (file path or inline JSON). Project ID is extracted automatically.
    Recommended for production.
  - **API key** [fallback]: Publisher endpoint with ``?key=`` param.
    Set GOOGLE_API_KEY. Used when ADC is unavailable, or as automatic
    fallback when ADC hits rate limits (HTTP 429).

When both are available, ADC takes priority with API key as fallback.

Note: this package is ``hexdag.stdlib.adapters.google`` — absolute imports
mean ``import google.auth`` below still resolves to the site-packages
``google`` namespace, not this package.
"""

import asyncio
import hashlib
import json
import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, cast

import google.auth
import google.auth.transport.requests
import google.oauth2.service_account
import httpx
from pydantic import BaseModel

from hexdag.kernel.exceptions import ParseError
from hexdag.kernel.logging import get_logger
from hexdag.kernel.ports.llm import (
    LLM,
    LLMResponse,
    MessageList,
    SupportsFunctionCalling,
    SupportsGeneration,
    SupportsStructuredOutput,
    SupportsUsageTracking,
    TokenUsage,
    ToolCall,
)
from hexdag.stdlib.adapters.base import HexDAGAdapter

logger = get_logger(__name__)

_active_instances: weakref.WeakSet["VertexAIAdapter"] = weakref.WeakSet()


async def aclose_all_vertex_clients() -> None:
    """Close HTTP clients on all live VertexAIAdapter instances.

    Intended for application shutdown hooks (e.g. FastAPI lifespan) where
    adapters were created outside an orchestrator-managed lifecycle.
    ``aclose()`` is idempotent, so double-closing is safe.
    """
    for adapter in list(_active_instances):
        await adapter.aclose()
    _active_instances.clear()


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class VertexAIAdapter(
    HexDAGAdapter,
    LLM,
    SupportsGeneration,
    SupportsStructuredOutput,
    SupportsFunctionCalling,
    SupportsUsageTracking,
    yaml_alias="vertex",
    port="llm",
):
    """Vertex AI / Google Gemini adapter implementing the hexDAG LLM port.

    Uses the native Vertex AI REST API. Supports ADC (service account) and
    API key authentication with automatic failover.

    Parameters
    ----------
    api_key : str | None
        Google API key. Falls back to ``GOOGLE_API_KEY`` env var.
        Ignored when Application Default Credentials are available.
    model : str
        Gemini model name (e.g. ``gemini-2.5-flash``, ``gemini-1.5-pro``).
    temperature : float
        Sampling temperature (0.0–2.0).
    max_tokens : int
        Maximum output tokens.
    timeout : float
        Request timeout in seconds.
    thinking_level : str | None
        Gemini thinking level (``minimal``/``low``/``medium``/``high``).
        Set to ``None`` to omit thinkingConfig entirely.
    cache_ttl : float
        Response cache TTL in seconds. Set to 0 to disable.
    credentials_json : str | None
        Inline service account JSON. Takes precedence over
        ``GOOGLE_APPLICATION_CREDENTIALS`` env var when provided.
    project_id : str | None
        GCP project ID override.

    Examples
    --------
    YAML configuration::

        spec:
          ports:
            llm:
              adapter: llm:vertex
              config:
                model: gemini-2.5-flash
                temperature: 0.0
                thinking_level: low
    """

    VALID_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})
    _ADC_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        thinking_level: str | None = "low",
        cache_ttl: float = 900.0,
        credentials_json: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking_level = thinking_level
        self._client: httpx.AsyncClient | None = None
        self._credentials: Any = None
        self._project_id: str = ""
        self._cred_lock = threading.Lock()
        self._explicit_credentials_json = credentials_json
        self._explicit_project_id = project_id
        self._last_usage: TokenUsage | None = None

        self._cache_ttl = cache_ttl
        self._cache: dict[str, _CacheEntry] = {}
        self._inflight: dict[str, asyncio.Event] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_lookups = 0

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._fallback_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        if api_key:
            # Explicit api_key parameter — use API key mode directly,
            # try ADC in background only as runtime fallback for rate limits
            self._use_adc = False
        else:
            # No explicit key — try ADC, fall back to env GOOGLE_API_KEY
            self._use_adc = self._try_init_adc()

        if self._use_adc:
            if self._explicit_project_id:
                self._project_id = self._explicit_project_id
            if not self._project_id:
                self._project_id = os.environ.get("GCP_PROJECT_ID", "")
            if not self._project_id:
                # ADC found credentials but no project ID — fall back to API key if available
                if self.api_key:
                    logger.warning(
                        "ADC credentials found but no GCP project ID. "
                        "Falling back to API key authentication."
                    )
                    self._use_adc = False
                    self._base_url = self._fallback_base_url
                else:
                    raise ValueError(
                        "ADC requires a GCP project ID. "
                        "Ensure service account JSON contains project_id, "
                        "or set GCP_PROJECT_ID."
                    )
            if self._use_adc:
                self._base_url = (
                    f"https://aiplatform.googleapis.com/v1"
                    f"/projects/{self._project_id}"
                    f"/locations/global"
                    f"/publishers/google/models"
                )
                logger.info(
                    "Using ADC (service account), project=%s, API key fallback=%s",
                    self._project_id,
                    "available" if self.api_key else "none",
                )
        else:
            if not self.api_key:
                raise ValueError(
                    "No authentication available. Set "
                    "GOOGLE_APPLICATION_CREDENTIALS for service account auth, "
                    "or GOOGLE_API_KEY for API key auth."
                )
            self._base_url = self._fallback_base_url
            logger.info("Using API key authentication")

        _active_instances.add(self)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _try_init_adc(self) -> bool:
        """Attempt to load Application Default Credentials.

        Checks (in order):
        1. Explicit ``credentials_json`` constructor parameter.
        2. ``GOOGLE_APPLICATION_CREDENTIALS`` env var — if the value starts
           with ``{`` it is treated as inline JSON (e.g. Azure Container
           Apps secrets). Otherwise treated as a file path (standard ADC).
        3. Default ADC (attached service account, metadata server, etc.).

        Returns ``True`` if credentials are available.
        """
        if self._explicit_credentials_json:
            try:
                info = json.loads(self._explicit_credentials_json)
                creds_factory = google.oauth2.service_account.Credentials.from_service_account_info
                self._credentials = creds_factory(info, scopes=self._ADC_SCOPES)
                self._project_id = self._explicit_project_id or info.get("project_id", "")
                return True
            except Exception as exc:
                logger.warning("Failed to parse explicit credentials JSON: %s", exc)

        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gac.strip().startswith("{"):
            try:
                info = json.loads(gac)
                creds_factory = google.oauth2.service_account.Credentials.from_service_account_info
                self._credentials = creds_factory(info, scopes=self._ADC_SCOPES)
                self._project_id = info.get("project_id", "")
                return True
            except Exception as exc:
                logger.warning(
                    "Failed to parse GOOGLE_APPLICATION_CREDENTIALS JSON: %s",
                    exc,
                )

        try:
            credentials, project = google.auth.default(scopes=self._ADC_SCOPES)
            self._credentials = credentials
            self._project_id = project or ""
            return True
        except Exception:
            return False

    def _get_bearer_token(self) -> str:
        """Return a valid Bearer token, refreshing if expired."""
        with self._cred_lock:
            if not self._credentials.valid:
                self._credentials.refresh(google.auth.transport.requests.Request())
            return str(self._credentials.token)

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable HTTP client, creating one if needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.timeout,
                    write=10.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=20,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client and clear cache."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._cache.clear()
        self._inflight.clear()

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def get_last_usage(self) -> TokenUsage | None:
        """Return token usage from the most recent API call.

        Cache hits do not update usage — no API call was made, so the
        value reflects the last request that actually reached Gemini.
        """
        return self._last_usage

    def _capture_usage(self, data: Any) -> None:
        """Record token usage from a Gemini ``usageMetadata`` block."""
        chunks = data if isinstance(data, list) else [data]
        for chunk in chunks:
            meta = chunk.get("usageMetadata")
            if meta:
                self._last_usage = TokenUsage(
                    input_tokens=meta.get("promptTokenCount", 0),
                    output_tokens=meta.get("candidatesTokenCount", 0),
                    total_tokens=meta.get("totalTokenCount", 0),
                )

    # ------------------------------------------------------------------
    # Response cache
    # ------------------------------------------------------------------

    @property
    def _cache_enabled(self) -> bool:
        return self._cache_ttl > 0

    def _build_cache_key(self, messages: MessageList, **extras: Any) -> str:
        """Build a SHA-256 cache key from messages and adapter config."""
        parts = [
            self.model,
            str(self.temperature),
            str(self.thinking_level),
            "|".join(m.model_dump_json() for m in messages),
        ]
        for key in sorted(extras):
            val = extras[key]
            if isinstance(val, type) and issubclass(val, BaseModel):
                val = val.model_json_schema()
            parts.append(f"{key}={json.dumps(val, sort_keys=True, default=str)}")
        raw = "\n".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _cache_get(self, key: str) -> tuple[bool, Any]:
        """Check cache for a hit.

        If another coroutine is computing the same key, wait for it.
        """
        async with self._cache_lock:
            self._cache_lookups += 1
            if self._cache_lookups % 50 == 0:
                self._evict_expired()

            entry = self._cache.get(key)
            if entry is not None and time.monotonic() < entry.expires_at:
                logger.debug("LLM cache hit key=%s", key[:12])
                return True, entry.value

            if entry is not None:
                del self._cache[key]

            event = self._inflight.get(key)

        if event is not None:
            logger.debug("LLM cache waiting on inflight key=%s", key[:12])
            await event.wait()
            async with self._cache_lock:
                entry = self._cache.get(key)
                if entry is not None and time.monotonic() < entry.expires_at:
                    return True, entry.value
            return False, None

        return False, None

    async def _cache_set(self, key: str, value: Any) -> None:
        """Store result in cache and wake waiters."""
        async with self._cache_lock:
            self._cache[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._cache_ttl,
            )
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    async def _cache_mark_inflight(self, key: str) -> None:
        """Mark a cache key as being computed."""
        async with self._cache_lock:
            self._inflight[key] = asyncio.Event()

    async def _cache_cancel_inflight(self, key: str) -> None:
        """Remove inflight marker and wake waiters (on failure)."""
        async with self._cache_lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def _evict_expired(self) -> None:
        """Remove expired cache entries. Must be called under _cache_lock."""
        now = time.monotonic()
        expired = [k for k, v in self._cache.items() if now >= v.expires_at]
        for k in expired:
            del self._cache[k]

    async def clear_cache(self) -> None:
        """Clear all cached responses."""
        async with self._cache_lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        max_retries: int = 2,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                headers: dict[str, str] = {}
                if self._use_adc:
                    headers["Authorization"] = f"Bearer {self._get_bearer_token()}"
                response = await self._get_client().post(
                    url,
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 429 and self._use_adc and self.api_key:
                    fallback_url = self._build_url(fallback=True)
                    logger.warning(
                        "ADC rate-limited (429), falling back to API key",
                    )
                    return await self._get_client().post(
                        fallback_url,
                        json=payload,
                    )
                return response
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                TimeoutError,
            ) as e:
                last_exc = e
                if attempt < max_retries:
                    delay = 1.0 * (2**attempt)
                    logger.warning(
                        "Gemini %s (attempt %d/%d), retrying in %ds",
                        type(e).__name__,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Gemini %s after %d attempts",
                        type(e).__name__,
                        max_retries + 1,
                    )
        raise last_exc  # type: ignore[misc]

    def _build_url(
        self,
        method: str = "generateContent",
        *,
        fallback: bool = False,
    ) -> str:
        """Build the API URL for the given method."""
        if fallback and self.api_key:
            base = f"{self._fallback_base_url}/{self.model}:{method}"
            return f"{base}?key={self.api_key}"
        base = f"{self._base_url}/{self.model}:{method}"
        if self._use_adc:
            return base
        return f"{base}?key={self.api_key}"

    # ------------------------------------------------------------------
    # Request/response helpers
    # ------------------------------------------------------------------

    def _build_generation_config(self, **overrides: Any) -> dict[str, Any]:
        """Build the generationConfig dict, including thinkingConfig."""
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.thinking_level and self.thinking_level in self.VALID_THINKING_LEVELS:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        config.update(overrides)
        return config

    def _convert_messages(self, messages: MessageList) -> tuple[list[dict[str, Any]], str | None]:
        """Convert hexDAG messages to Gemini format.

        Gemini uses ``user`` and ``model`` roles instead of ``user`` and
        ``assistant``. System messages are handled separately via
        ``system_instruction``.
        """
        gemini_contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": msg.content}]})
            else:
                gemini_contents.append({"role": "user", "parts": [{"text": msg.content}]})

        return gemini_contents, system_instruction

    def _extract_text(self, data: Any) -> str | None:
        """Extract text from a Gemini response."""
        chunks = data if isinstance(data, list) else [data]
        text_parts: list[str] = []
        for chunk in chunks:
            for candidate in chunk.get("candidates", []):
                finish_reason = candidate.get("finishReason")
                if finish_reason and finish_reason not in (
                    "STOP",
                    "MAX_TOKENS",
                ):
                    logger.warning("Gemini finishReason: %s", finish_reason)
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("thought"):
                        continue
                    if "text" in part:
                        text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
        logger.warning("No content in Gemini response")
        return None

    # ------------------------------------------------------------------
    # LLM port methods
    # ------------------------------------------------------------------

    async def aresponse(self, messages: MessageList) -> str | None:
        """Generate a response using Gemini.

        Args:
            messages: List of Message objects with role and content.

        Returns:
            The generated response text, or ``None`` if failed.
        """
        cache_key: str | None = None
        if self._cache_enabled:
            cache_key = self._build_cache_key(messages)
            hit, cached = await self._cache_get(cache_key)
            if hit:
                return cast("str | None", cached)
            await self._cache_mark_inflight(cache_key)

        try:
            contents, system_instruction = self._convert_messages(messages)

            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": self._build_generation_config(),
            }

            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            url = self._build_url()
            response = await self._post_with_retry(url, payload)
            response.raise_for_status()
            data = response.json()
            self._capture_usage(data)

            result = self._extract_text(data)
            if cache_key is not None:
                await self._cache_set(cache_key, result)
            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "Gemini API HTTP error: %d - %s",
                e.response.status_code,
                e.response.text,
            )
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.error("Gemini API call timed out", exc_info=True)
            else:
                logger.error("Gemini API error: %s", e, exc_info=True)
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise

    async def aresponse_structured(
        self,
        messages: MessageList,
        output_schema: dict[str, Any] | type[BaseModel],
    ) -> dict[str, Any]:
        """Generate a structured JSON response using Gemini's native JSON mode.

        Uses ``responseMimeType: "application/json"`` and ``responseSchema``
        so Gemini guarantees valid JSON output conforming to the schema.
        """
        cache_key: str | None = None
        if self._cache_enabled:
            schema_for_key = (
                output_schema
                if isinstance(output_schema, dict)
                else output_schema.model_json_schema()
            )
            cache_key = self._build_cache_key(messages, output_schema=schema_for_key)
            hit, cached = await self._cache_get(cache_key)
            if hit:
                return cast("dict[str, Any]", cached)
            await self._cache_mark_inflight(cache_key)

        try:
            contents, system_instruction = self._convert_messages(messages)

            if isinstance(output_schema, dict):
                json_schema = output_schema
            else:
                json_schema = output_schema.model_json_schema()

            gemini_schema = _pydantic_schema_to_gemini(json_schema)

            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": self._build_generation_config(
                    responseMimeType="application/json",
                    responseSchema=gemini_schema,
                ),
            }

            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            url = self._build_url()
            response = await self._post_with_retry(url, payload)
            response.raise_for_status()
            data = response.json()
            self._capture_usage(data)

            text = self._extract_text(data)
            if text is None:
                raise ParseError("Gemini returned no content for structured output request")

            result: dict[str, Any] = json.loads(text)
            if cache_key is not None:
                await self._cache_set(cache_key, result)
            return result

        except (json.JSONDecodeError, ParseError) as e:
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise ParseError(f"Failed to parse structured JSON from Gemini: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(
                "Gemini API HTTP error: %d - %s",
                e.response.status_code,
                e.response.text,
            )
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise ParseError(
                f"Gemini API error {e.response.status_code} during structured output"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("Gemini API timeout during structured output after retries")
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise ParseError("Gemini API timeout during structured output") from e
        except Exception as e:
            if isinstance(e, TimeoutError):
                msg = "Gemini API call timed out during structured output"
            else:
                msg = f"Gemini API error during structured output: {type(e).__name__}: {e}"
            logger.error(msg, exc_info=True)
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise ParseError(msg) from e

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Generate response with native Gemini function calling.

        Args:
            messages: Conversation messages.
            tools: Tool definitions (OpenAI format, will be converted).
            tool_choice: Tool selection strategy.

        Returns:
            LLMResponse with content and tool calls.
        """
        cache_key: str | None = None
        if self._cache_enabled:
            cache_key = self._build_cache_key(messages, tools=tools, tool_choice=tool_choice)
            hit, cached = await self._cache_get(cache_key)
            if hit:
                return cast("LLMResponse", cached)
            await self._cache_mark_inflight(cache_key)

        try:
            contents, system_instruction = self._convert_messages(messages)
            gemini_tools = self._convert_tools_to_gemini(tools)

            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": self._build_generation_config(),
                "tools": gemini_tools,
            }

            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

            if tool_choice == "none":
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            elif tool_choice == "required":
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}

            url = self._build_url()
            response = await self._post_with_retry(url, payload)
            response.raise_for_status()
            data = response.json()
            self._capture_usage(data)

            candidates = data.get("candidates", [])
            if not candidates:
                result = LLMResponse(content=None, tool_calls=None)
                if cache_key is not None:
                    await self._cache_set(cache_key, result)
                return result

            content_data = candidates[0].get("content", {})
            parts = content_data.get("parts", [])
            finish_reason = candidates[0].get("finishReason")

            text_content = None
            tool_calls: list[ToolCall] = []

            for part in parts:
                if part.get("thought"):
                    continue
                if "text" in part:
                    text_content = part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )

            result = LLMResponse(
                content=text_content,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=finish_reason,
            )
            if cache_key is not None:
                await self._cache_set(cache_key, result)
            return result

        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.error("Gemini API call with tools timed out", exc_info=True)
            else:
                logger.error("Gemini API error with tools: %s", e, exc_info=True)
            if cache_key is not None:
                await self._cache_cancel_inflight(cache_key)
            raise

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    def _convert_tools_to_gemini(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool definitions to Gemini format."""
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        return [{"functionDeclarations": function_declarations}]


# ------------------------------------------------------------------
# Schema conversion
# ------------------------------------------------------------------


def _pydantic_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema (from Pydantic) to Gemini's responseSchema format.

    Gemini's responseSchema is a subset of OpenAPI 3.0 Schema Object.
    It does not support ``$defs``, ``anyOf``, ``allOf``, ``oneOf``,
    ``default``, ``examples``, or ``title`` at property level.
    Nullable fields use ``nullable: true`` instead of ``anyOf`` with null.
    """
    defs = schema.get("$defs", {})

    def _resolve(s: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in s:
            ref_name = s["$ref"].split("/")[-1]
            return _resolve(defs.get(ref_name, {}))
        return s

    def _convert(s: dict[str, Any]) -> dict[str, Any]:
        s = _resolve(s)
        out: dict[str, Any] = {}

        if "anyOf" in s:
            variants = [v for v in s["anyOf"] if v.get("type") != "null"]
            is_nullable = any(v.get("type") == "null" for v in s["anyOf"])
            if variants:
                out = _convert(variants[0])
            if is_nullable:
                out["nullable"] = True
            return out

        schema_type = s.get("type")
        type_map = {
            "object": "OBJECT",
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
        }
        if schema_type:
            out["type"] = type_map.get(schema_type, schema_type.upper())

        if "description" in s:
            out["description"] = s["description"]

        if "enum" in s:
            out["enum"] = s["enum"]

        if "properties" in s:
            out["properties"] = {k: _convert(v) for k, v in s["properties"].items()}

        if "required" in s:
            out["required"] = s["required"]

        if "items" in s:
            out["items"] = _convert(s["items"])

        return out

    result = _convert(schema)
    if "type" not in result:
        result["type"] = "OBJECT"

    return result
