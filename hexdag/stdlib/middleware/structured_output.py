"""Structured output fallback middleware.

Wraps any ``SupportsGeneration`` adapter that does NOT natively implement
``SupportsStructuredOutput``.  Adds structured output by injecting schema
instructions into the prompt and parsing JSON from the response.

This is the prompt-based fallback — native adapters (OpenAI, Anthropic)
bypass this entirely via their own ``aresponse_structured`` implementations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import orjson

from hexdag.kernel.exceptions import ParseError
from hexdag.kernel.ports.llm import (
    LLMResponse,
    Message,
    MessageList,
    SupportsGeneration,
    SupportsStructuredOutput,
    ToolChoice,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


class StructuredOutputFallback(SupportsGeneration, SupportsStructuredOutput):
    """Middleware that adds structured output to any SupportsGeneration adapter.

    Implements ``SupportsStructuredOutput`` by:
    1. Injecting schema instructions into the prompt
    2. Calling the inner adapter's ``aresponse()``
    3. Parsing JSON from the response text
    4. Returning the parsed dict

    All other protocol methods are forwarded to the inner adapter.
    """

    def __init__(self, inner: Any) -> None:
        """Initialize with the inner SupportsGeneration adapter."""
        self._inner = inner

    # -- SupportsGeneration: passthrough --

    async def aresponse(self, messages: MessageList) -> str | None:
        """Forward to inner adapter."""
        return await self._inner.aresponse(messages)  # type: ignore[no-any-return]

    # -- SupportsFunctionCalling: passthrough if available --

    async def aresponse_with_tools(
        self,
        messages: MessageList,
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice | dict[str, Any] = "auto",
    ) -> LLMResponse:
        """Forward to inner adapter (if it supports function calling)."""
        return await self._inner.aresponse_with_tools(  # type: ignore[no-any-return]
            messages, tools, tool_choice
        )

    # -- SupportsStructuredOutput: the actual middleware logic --

    async def aresponse_structured(
        self,
        messages: MessageList,
        output_schema: dict[str, Any] | type[BaseModel],
    ) -> dict[str, Any]:
        """Generate structured output via prompt injection + JSON parsing.

        Appends schema instructions to the last user message, calls
        ``aresponse()``, then parses and returns the JSON.
        """
        from pydantic import BaseModel  # lazy: avoid import at module level for optional dep

        # Build schema instruction
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            schema = output_schema.model_json_schema()
        else:
            schema = output_schema

        instruction = _build_schema_instruction(schema)

        # Inject instruction into messages
        enhanced = _inject_schema_instruction(messages, instruction)

        # Call inner adapter
        response = await self._inner.aresponse(enhanced)

        if response is None:
            raise ParseError("LLM returned None — cannot parse structured output")

        # Parse JSON from response
        return _parse_json_response(response)

    # -- Forward attribute access for other protocols --

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to inner adapter."""
        return getattr(self._inner, name)


def _build_schema_instruction(schema: dict[str, Any]) -> str:
    """Build a schema instruction string from a JSON Schema."""
    fields_info: list[str] = []
    if "properties" in schema:
        for field_name, field_schema in schema["properties"].items():
            field_type = field_schema.get("type", "any")
            field_desc = field_schema.get("description", "")
            desc_part = f" - {field_desc}" if field_desc else ""
            fields_info.append(f"  - {field_name}: {field_type}{desc_part}")

    fields_text = "\n".join(fields_info) if fields_info else "  - (no specific fields defined)"

    example_data = {field: f"<{field}_value>" for field in schema.get("properties", {})}
    example_json = orjson.dumps(example_data, option=orjson.OPT_INDENT_2).decode()

    return (
        "\n\n## Output Format\n"
        "Respond with valid JSON matching this schema:\n"
        f"{fields_text}\n\n"
        f"Example: {example_json}\n"
    )


def _inject_schema_instruction(messages: MessageList, instruction: str) -> MessageList:
    """Append schema instruction to the last user message."""
    if not messages:
        return [Message(role="user", content=instruction)]

    enhanced = list(messages)
    # Find last user message and append instruction
    for i in range(len(enhanced) - 1, -1, -1):
        if enhanced[i].role == "user":
            enhanced[i] = Message(
                role="user",
                content=enhanced[i].content + instruction,
            )
            return enhanced

    # No user message found — append as new user message
    enhanced.append(Message(role="user", content=instruction))
    return enhanced


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    from hexdag.kernel.validation.secure_json import SafeJSON  # lazy: avoid circular import

    safe = SafeJSON()
    result = safe.loads_from_text(text)
    if result.ok:
        return result.data  # type: ignore[return-value]

    # Fall back to direct parse
    try:
        data: dict[str, Any] = json.loads(text.strip())
        return data
    except json.JSONDecodeError as e:
        raise ParseError(
            f"Failed to parse JSON from LLM response: {e}\nResponse: {text[:500]}"
        ) from e
