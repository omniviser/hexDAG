"""Common utilities for node operations.

This module provides shared utilities used across different node types:
- JSON parsing and extraction
- Tool call parsing
- Logging utilities
- Shared regex patterns
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger("hexai.app.application.nodes.utils")


class JsonUtils:
    """Utilities for JSON parsing and extraction from text responses."""

    @staticmethod
    def extract_json_from_response(response: str) -> str | None:
        """Extract JSON from response text.

        Try to extract JSON from markdown code blocks or find JSON objects.
        """
        # Try markdown code blocks first
        json_patterns = [
            r"```json\s*(.*?)\s*```",  # ```json ... ```
            r"```\s*(.*?)\s*```",  # ``` ... ```
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                cleaned = match.strip()
                if JsonUtils._is_valid_json(cleaned):
                    return cleaned  # type: ignore[no-any-return]

        # Try to find JSON objects using simple bracket matching
        candidates = JsonUtils._find_json_objects(response)
        for candidate in candidates:
            if JsonUtils._is_valid_json(candidate):
                return candidate

        return None

    @staticmethod
    def _find_json_objects(text: str) -> list[str]:
        """Find JSON objects using simple bracket matching."""
        objects = []
        stack: list[str] = []
        start = None

        for i, char in enumerate(text):
            if char == "{":
                if not stack:
                    start = i
                stack.append(char)
            elif char == "}" and stack:
                stack.pop()
                if not stack and start is not None:
                    objects.append(text[start : i + 1])  # noqa: E203
                    start = None

        return sorted(objects, key=len, reverse=True)

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def parse_json_safely(text: str, fallback: Any = None) -> Any:
        """Safely parse JSON with fallback value."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return fallback


class ToolUtils:
    """Utilities for parsing tool calls from LLM responses."""

    # Common tool call patterns
    INVOKE_TOOL_PATTERN = r"INVOKE_TOOL:\s*(\w+)\s*(\{.*?\})?"
    TOOL_NAME_PATTERN = r"INVOKE_TOOL:\s*(\w+)"
    TOOL_PARAMS_PATTERN = r"INVOKE_TOOL:\s*\w+\s*(\{.*?\})"

    @staticmethod
    def has_tool_call(response: str) -> bool:
        """Check if response contains a tool call."""
        return bool(re.search(ToolUtils.TOOL_NAME_PATTERN, response))

    @staticmethod
    def extract_tool_name(response: str) -> str | None:
        """Extract tool name from INVOKE_TOOL command."""
        match = re.search(ToolUtils.TOOL_NAME_PATTERN, response)
        return match.group(1) if match else None

    @staticmethod
    def extract_tool_parameters(
        response: str, fallback_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract tool parameters from LLM response.

        Simple approach: look for JSON after INVOKE_TOOL command.
        If not found, return fallback parameters.
        """
        if fallback_params is None:
            fallback_params = {}

        # Look for JSON parameters after INVOKE_TOOL
        params_match = re.search(ToolUtils.TOOL_PARAMS_PATTERN, response, re.DOTALL)
        if params_match:
            json_str = params_match.group(1)
            try:
                return json.loads(json_str)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                # Try simple cleanup and parse again
                cleaned = json_str.strip()
                # Remove trailing commas
                cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
                try:
                    return json.loads(cleaned)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass

        return fallback_params

    @staticmethod
    def parse_tool_call(response: str) -> tuple[str | None, dict[str, Any]]:
        """Parse tool call and return (tool_name, parameters)."""
        tool_name = ToolUtils.extract_tool_name(response)
        params = ToolUtils.extract_tool_parameters(response)
        return tool_name, params


class LoggingUtils:
    """Common logging utilities for node operations."""

    @staticmethod
    def log_node_start(node_name: str, node_type: str = "NODE") -> None:
        """Log the start of a node execution."""
        logger.info("⚙️  %s: %s", node_type, node_name)

    @staticmethod
    def log_node_success(node_name: str, result_type: str = "unknown") -> None:
        """Log successful node completion."""
        logger.info("✅ Node %s completed successfully", node_name)
        logger.debug("📊 Result type: %s", result_type)

    @staticmethod
    def log_node_error(node_name: str, error: Exception) -> None:
        """Log node execution error."""
        logger.error("❌ Node %s failed: %s", node_name, error)
