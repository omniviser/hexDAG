"""Enhanced tool utilities for agent nodes.

Supports multiple tool calling formats and tool description management.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hexdag.core.validation.secure_json import SafeJSON


class ToolCallFormat(StrEnum):
    """Tool calling formats supported by INVOKE_TOOL: prefix."""

    FUNCTION_CALL = "function_call"  # INVOKE_TOOL: tool_name(param1='value1')
    JSON = "json"  # INVOKE_TOOL: {"tool": "tool_name", "params": {...}}
    MIXED = "mixed"  # Support both formats


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Parsed tool call information."""

    name: str
    params: dict[str, Any]
    format: ToolCallFormat
    raw_text: str


class ToolParser:
    """Parse tool calls with INVOKE_TOOL: prefix for clear identification."""

    # Function call pattern with INVOKE_TOOL: prefix (supports namespace:tool_name)
    INVOKE_TOOL_PATTERN = re.compile(
        r"INVOKE_TOOL:\s*([\w:]+)\s*\(\s*((?:[^()]*(?:\([^()]*\)[^()]*)*)*)\s*\)"
    )

    # JSON pattern with INVOKE_TOOL: prefix
    INVOKE_TOOL_JSON_PATTERN = re.compile(r"INVOKE_TOOL:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})")

    # Parameter parsing pattern
    PARAM_PATTERN = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s\)]+))")

    @classmethod
    def parse_tool_calls(
        cls, text: str, format: ToolCallFormat = ToolCallFormat.MIXED
    ) -> list[ToolCall]:
        """Extract all tool calls from text using INVOKE_TOOL: prefix.

        Returns
        -------
        list[ToolCall]
            List of parsed tool calls found in the text
        """
        calls = []

        if format in (ToolCallFormat.FUNCTION_CALL, ToolCallFormat.MIXED):
            calls.extend(cls._parse_function_calls(text))

        if format in (ToolCallFormat.JSON, ToolCallFormat.MIXED):
            calls.extend(cls._parse_json_calls(text))

        return calls

    @classmethod
    def _parse_function_calls(cls, text: str) -> list[ToolCall]:
        """Parse function calls with INVOKE_TOOL: prefix.

        Returns
        -------
        list[ToolCall]
            List of parsed function-style tool calls
        """
        calls = []

        for match in cls.INVOKE_TOOL_PATTERN.finditer(text):
            tool_name = match.group(1)
            args_str = match.group(2)
            params = cls._parse_parameters(args_str)

            calls.append(
                ToolCall(
                    name=tool_name,
                    params=params,
                    format=ToolCallFormat.FUNCTION_CALL,
                    raw_text=match.group(0),
                )
            )

        return calls

    @classmethod
    def _parse_json_calls(cls, text: str) -> list[ToolCall]:
        """Parse JSON calls with INVOKE_TOOL: prefix.

        Returns
        -------
        list[ToolCall]
            List of parsed JSON-style tool calls
        """
        calls = []

        for match in cls.INVOKE_TOOL_JSON_PATTERN.finditer(text):
            try:
                json_str = match.group(1)
                data = SafeJSON().loads(json_str)

                if isinstance(data, dict) and "tool" in data:
                    calls.append(
                        ToolCall(
                            name=data["tool"],
                            params=data.get("params", {}),
                            format=ToolCallFormat.JSON,
                            raw_text=match.group(0),
                        )
                    )

            except json.JSONDecodeError:
                continue

        return calls

    @classmethod
    def _parse_parameters(cls, args_str: str) -> dict[str, Any]:
        """Parse function parameters from argument string.

        Returns
        -------
        dict[str, Any]
            Dictionary of parsed parameter names and values
        """
        params = {}

        for match in cls.PARAM_PATTERN.finditer(args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)

            # Try to parse as JSON for complex types
            try:
                params[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                params[key] = value

        return params


def tool_schema_to_openai(
    name: str,
    description: str,
    parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a tool schema (plain dict) to OpenAI function calling format.

    Works directly with the schema dicts from ``tool_schema_from_callable()``,
    avoiding the need for wrapper classes.

    Parameters
    ----------
    name : str
        Tool name
    description : str
        Tool description
    parameters : list[dict[str, Any]]
        Parameter list from ``tool_schema_from_callable()``

    Returns
    -------
    dict[str, Any]
        OpenAI-compatible tool schema
    """
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in parameters:
        param_name = param.get("name", "")
        if param_name.startswith("**"):
            continue
        base_type = param.get("type", "str").split("[")[0].strip()
        json_type = type_mapping.get(base_type, "string")
        properties[param_name] = {
            "type": json_type,
            "description": param.get("description", ""),
        }
        if param.get("required", False):
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
