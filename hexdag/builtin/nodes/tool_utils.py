"""Enhanced tool utilities for agent nodes.

Supports multiple tool calling formats and tool description management.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolCallFormat(StrEnum):
    """Tool calling formats supported by INVOKE_TOOL: prefix."""

    FUNCTION_CALL = "function_call"  # INVOKE_TOOL: tool_name(param1='value1')
    JSON = "json"  # INVOKE_TOOL: {"tool": "tool_name", "params": {...}}
    MIXED = "mixed"  # Support both formats


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str = Field(..., description="Parameter name")
    description: str = Field(..., description="Parameter description")
    param_type: str = Field(default="str", description="Parameter type")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not required")


class ToolDefinition(BaseModel):
    """Complete tool definition with descriptions and parameters."""

    name: str = Field(..., description="Tool name")
    simplified_description: str = Field(..., description="Brief tool description")
    detailed_description: str = Field(..., description="Detailed tool description")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    examples: list[str] = Field(default_factory=list, description="Usage examples")

    def to_simplified_string(self) -> str:
        """Convert to simplified string format.

        Returns
        -------
        str
            Simplified string representation of the tool
        """
        return f"{self.name}: {self.simplified_description}"

    def to_detailed_string(self) -> str:
        """Convert to detailed string format.

        Returns
        -------
        str
            Detailed string representation of the tool with parameters and examples
        """
        lines = [f"Tool: {self.name}", f"Description: {self.detailed_description}", "Parameters:"]

        for param in self.parameters:
            req_str = "required" if param.required else "optional"
            lines.append(f"  - {param.name} ({param.param_type}, {req_str}): {param.description}")

        if self.examples:
            lines.append("Examples:")
            lines.extend(f"  {example}" for example in self.examples)

        return "\n".join(lines)


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
                data = json.loads(json_str)

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


class ToolSchemaConverter:
    """Convert ToolDefinition to various LLM provider formats."""

    @staticmethod
    def to_openai_schema(tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to OpenAI function calling format.

        Args
        ----
            tool_def: Tool definition to convert

        Returns
        -------
        dict[str, Any]
            OpenAI-compatible tool schema
        """
        properties = {}
        required = []

        for param in tool_def.parameters:
            # Map Python types to JSON Schema types
            json_type = ToolSchemaConverter._python_type_to_json_type(param.param_type)

            properties[param.name] = {"type": json_type, "description": param.description}

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.detailed_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @staticmethod
    def to_anthropic_schema(tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to Anthropic tool use format.

        Args
        ----
            tool_def: Tool definition to convert

        Returns
        -------
        dict[str, Any]
            Anthropic-compatible tool schema
        """
        properties = {}
        required = []

        for param in tool_def.parameters:
            json_type = ToolSchemaConverter._python_type_to_json_type(param.param_type)

            properties[param.name] = {"type": json_type, "description": param.description}

            if param.required:
                required.append(param.name)

        return {
            "name": tool_def.name,
            "description": tool_def.detailed_description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    @staticmethod
    def _python_type_to_json_type(python_type: str) -> str:
        """Map Python type strings to JSON Schema types.

        Args
        ----
            python_type: Python type as string (e.g., "str", "int")

        Returns
        -------
        str
            JSON Schema type
        """
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        # Handle generic types like list[str], dict[str, int]
        base_type = python_type.split("[")[0].strip()

        return type_mapping.get(base_type, "string")


class ToolDescriptionManager:
    """Manage tool descriptions with simplified/detailed views."""

    def __init__(self) -> None:
        """Initialize tool description manager."""
        self.tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition.

        Parameters
        ----------
        tool_def : ToolDefinition
            The tool definition to register
        """
        self.tools[tool_def.name] = tool_def

    def register_tools(self, tool_defs: list[ToolDefinition]) -> None:
        """Register multiple tool definitions.

        Parameters
        ----------
        tool_defs : list[ToolDefinition]
            List of tool definitions to register
        """
        for tool_def in tool_defs:
            self.register_tool(tool_def)

    def get_simplified_descriptions(self) -> str:
        """Get simplified tool descriptions for prompt.

        Returns
        -------
        str
            Formatted string with simplified tool descriptions
        """
        lines = ["Available tools:"]

        lines.extend(f"- {tool.to_simplified_string()}" for tool in self.tools.values())

        return "\n".join(lines)

    def get_detailed_descriptions(self) -> str:
        """Get detailed tool descriptions for prompt.

        Returns
        -------
        str
            Formatted string with detailed tool descriptions
        """
        lines = ["Available tools (detailed):"]

        for tool in self.tools.values():
            lines.append(tool.to_detailed_string())
            lines.append("")  # Add spacing between tools

        return "\n".join(lines)

    def get_detailed_description(self, tool_name: str) -> str:
        """Get detailed description for a specific tool.

        Returns
        -------
        str
            Detailed description of the specified tool
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return f"No description available for tool: {tool_name}"

        return tool.to_detailed_string()

    def create_tool_check_function(self) -> Any:
        """Create a function that returns detailed tool descriptions.

        Returns
        -------
        Callable[[str], str]
            Async function that takes a tool name and returns its detailed description
        """

        async def check_tool_description(tool_name: str) -> str:
            """Get detailed description for a tool."""
            return self.get_detailed_description(tool_name)

        return check_tool_description
