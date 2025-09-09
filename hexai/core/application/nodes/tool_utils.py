"""Enhanced tool utilities for agent nodes.

Supports multiple tool calling formats and tool description management.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ...validation.secure_json import loads as secure_json_loads


class ToolCallFormat(Enum):
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
        """Convert to simplified string format."""
        return f"{self.name}: {self.simplified_description}"

    def to_detailed_string(self) -> str:
        """Convert to detailed string format."""
        lines = [f"Tool: {self.name}", f"Description: {self.detailed_description}", "Parameters:"]

        for param in self.parameters:
            req_str = "required" if param.required else "optional"
            lines.append(f"  - {param.name} ({param.param_type}, {req_str}): {param.description}")

        if self.examples:
            lines.append("Examples:")
            for example in self.examples:
                lines.append(f"  {example}")

        return "\n".join(lines)


@dataclass
class ToolCall:
    """Parsed tool call information."""

    name: str
    params: dict[str, Any]
    format: ToolCallFormat
    raw_text: str


class ToolParser:
    """Parse tool calls with INVOKE_TOOL: prefix for clear identification."""

    # Function call pattern with INVOKE_TOOL: prefix
    INVOKE_TOOL_PATTERN = re.compile(
        r"INVOKE_TOOL:\s*(\w+)\s*\(\s*((?:[^()]*(?:\([^()]*\)[^()]*)*)*)\s*\)"
    )

    # JSON pattern with INVOKE_TOOL: prefix
    INVOKE_TOOL_JSON_PATTERN = re.compile(r"INVOKE_TOOL:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})")

    # Parameter parsing pattern
    PARAM_PATTERN = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s\)]+))")

    @classmethod
    def parse_tool_calls(
        cls, text: str, format: ToolCallFormat = ToolCallFormat.MIXED
    ) -> list[ToolCall]:
        """Extract all tool calls from text using INVOKE_TOOL: prefix."""
        calls = []

        if format in (ToolCallFormat.FUNCTION_CALL, ToolCallFormat.MIXED):
            calls.extend(cls._parse_function_calls(text))

        if format in (ToolCallFormat.JSON, ToolCallFormat.MIXED):
            calls.extend(cls._parse_json_calls(text))

        return calls

    @classmethod
    def _parse_function_calls(cls, text: str) -> list[ToolCall]:
        """Parse function calls with INVOKE_TOOL: prefix."""
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
        """Parse JSON calls with INVOKE_TOOL: prefix."""
        calls = []

        for match in cls.INVOKE_TOOL_JSON_PATTERN.finditer(text):
            try:
                json_str = match.group(1)
                data = secure_json_loads(json_str)

                # Check if it's a tool call
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
        """Parse function parameters from argument string."""
        params = {}

        for match in cls.PARAM_PATTERN.finditer(args_str):
            key = match.group(1)
            # Get value from whichever group matched
            value = match.group(2) or match.group(3) or match.group(4)

            # Try to parse as JSON for complex types
            parsed = secure_json_loads(value)
            params[key] = parsed if parsed is not None else value

        return params


class ToolDescriptionManager:
    """Manage tool descriptions with simplified/detailed views."""

    def __init__(self) -> None:
        """Initialize tool description manager."""
        self.tools: dict[str, ToolDefinition] = {}

    def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition."""
        self.tools[tool_def.name] = tool_def

    def register_tools(self, tool_defs: list[ToolDefinition]) -> None:
        """Register multiple tool definitions."""
        for tool_def in tool_defs:
            self.register_tool(tool_def)

    def get_simplified_descriptions(self) -> str:
        """Get simplified tool descriptions for prompt."""
        lines = ["Available tools:"]

        for tool in self.tools.values():
            lines.append(f"- {tool.to_simplified_string()}")

        return "\n".join(lines)

    def get_detailed_descriptions(self) -> str:
        """Get detailed tool descriptions for prompt."""
        lines = ["Available tools (detailed):"]

        for tool in self.tools.values():
            lines.append(tool.to_detailed_string())
            lines.append("")  # Add spacing between tools

        return "\n".join(lines)

    def get_detailed_description(self, tool_name: str) -> str:
        """Get detailed description for a specific tool."""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"No description available for tool: {tool_name}"

        return tool.to_detailed_string()

    def create_tool_check_function(self) -> Any:
        """Create a function that returns detailed tool descriptions."""

        async def check_tool_description(tool_name: str) -> str:
            """Get detailed description for a tool."""
            return self.get_detailed_description(tool_name)

        return check_tool_description
