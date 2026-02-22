"""Simple and secure prompt template system — Phase 3 of the rendering pipeline.

This module handles ``{{variable}}`` substitution at **runtime** when nodes
execute and dependency outputs are available. Uses a lightweight engine
(not Jinja2) for security — no code execution, loops, or conditionals.

Error messages are prefixed with ``[Phase 3: Runtime Template Rendering]``.

See Also
--------
- ``pipeline_builder/preprocessing/template.py`` — Phase 2: build-time Jinja2
- ``pipeline_builder/preprocessing/env_vars.py`` — Phase 1: ``${VAR}`` resolution
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from hexdag.kernel.exceptions import HexDAGError


class PromptTemplateError(HexDAGError):
    """Base exception for prompt template errors."""


class MissingVariableError(PromptTemplateError):
    """Raised when required template variables are missing."""


# ---------------------------------------------------------------------------
# Cached Template Parsing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _extract_variables_cached(template: str) -> tuple[str, ...]:
    """Extract variable names from template with caching.

    This function is cached because template parsing with regex is expensive
    and templates are often reused (e.g., same prompt template for multiple nodes).

    Parameters
    ----------
    template : str
        Template string to analyze

    Returns
    -------
    tuple[str, ...]
        Tuple of unique root variable names found in template
    """
    pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
    matches = re.findall(pattern, template)

    root_vars_set = set()
    for match in matches:
        root_var = match.split(".")[0]
        root_vars_set.add(root_var)

    return tuple(sorted(root_vars_set))


# ---------------------------------------------------------------------------
# Simple Template Engine
# ---------------------------------------------------------------------------


class PromptTemplate:
    """Simple and secure prompt template with variable substitution.

    Supports:
    - Basic substitution: {{variable}}
    - Nested access: {{user.name}}
    - Safe variable extraction and validation

    Does NOT support (for security):
    - Code execution
    - Loops or conditionals
    - Complex expressions
    """

    def __init__(self, template: str, input_vars: list[str] | None = None) -> None:
        r"""Initialize a PromptTemplate with a template string and optional input variables.

        Args
        ----
            template: Template string with {{variable}} placeholders
            input_vars: Optional list of variable names. If None, variables are auto-extracted

        Raises
        ------
            PromptTemplateError: If template has invalid syntax

        Examples
        --------
            # Auto-extract variables
            template = PromptTemplate("Hello {{name}}!")
            assert template.input_vars == ["name"]

            # Manually specify variables
            template = PromptTemplate(
                "User: {{user.name}} ({{user.role}})\nMessage: {{message}}",
                ["user", "message"]
            )
        """
        self.template = template

        if input_vars is not None:
            self.input_vars = list(input_vars)
        else:
            # Use cached extraction for performance
            self.input_vars = list(_extract_variables_cached(template))

    def _get_nested_value(self, data: dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation.

        Args
        ----
            data: Dictionary to extract value from
            key: Key path (e.g., "user.name" -> data["user"]["name"])

        Returns
        -------
        Any
            The value at the specified path
        """
        parts = key.split(".")
        value = data
        for part in parts:
            value = value[part] if isinstance(value, dict) else getattr(value, part)
        return value

    def render(self, **kwargs: Any) -> str:
        """Render the template with provided variables.

        Args
        ----
            **kwargs: Variable values to substitute in template

        Returns
        -------
        str
            Rendered template string

        Raises
        ------
        MissingVariableError
            If required variables are missing

        Examples
        --------
            template = PromptTemplate("Hello {{name}}!")
            result = template.render(name="Alice")  # "Hello Alice!"

            # With nested data
            template = PromptTemplate("User: {{user.name}}")
            result = template.render(user={"name": "Bob"})  # "User: Bob"
        """
        # Check for missing required variables
        missing_vars = [var for var in self.input_vars if var not in kwargs]

        if missing_vars:
            preview = self.template[:200] + ("..." if len(self.template) > 200 else "")
            raise MissingVariableError(
                f"[Phase 3: Runtime Template Rendering] "
                f"Missing required template variables: {missing_vars}. "
                f"Required variables: {self.input_vars}. "
                f"Provided variables: {list(kwargs.keys())}\n"
                f"  Template: {preview}\n"
                f"  Hint: Ensure upstream nodes produce outputs with these keys, "
                f"or check the 'dependencies' list in the YAML node spec."
            )

        # Perform variable substitution
        result = self.template

        # Find all variable references in the template
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"

        def replace_var(match: Any) -> str:
            var_path = match.group(1).strip()
            try:
                value = self._get_nested_value(kwargs, var_path)
                return str(value)
            except (KeyError, AttributeError) as e:
                preview = self.template[:200] + ("..." if len(self.template) > 200 else "")
                raise MissingVariableError(
                    f"[Phase 3: Runtime Template Rendering] "
                    f"Cannot access variable '{var_path}': {e}\n"
                    f"  Template: {preview}\n"
                    f"  Hint: Ensure the upstream node outputs a structure "
                    f"containing '{var_path}'."
                ) from e

        return re.sub(pattern, replace_var, result)

    def parse_output(self, output: str) -> str:
        r"""Parse and post-process rendered template output.

        Default implementation is a passthrough that returns the output unchanged.
        Override this method in subclasses for custom output processing.

        Args
        ----
            output: Rendered template string

        Returns
        -------
        str
            Processed output string (by default, unchanged)

        Examples
        --------
            # Default behavior
            template = PromptTemplate("Hello {{name}}!")
            rendered = template.render(name="Alice")
            parsed = template.parse_output(rendered)  # Same as rendered

            # Custom parsing in subclass
            class UppercaseTemplate(PromptTemplate):
                def parse_output(self, output: str) -> str:
                    return output.upper()
        """
        return output

    def format(self, **kwargs: Any) -> str:
        r"""Render template and parse output in one step.

        Args
        ----
            **kwargs: Variable values to substitute in template

        Returns
        -------
        str
            Rendered and parsed template string

        Examples
        --------
            template = PromptTemplate("Hello {{name}}!")
            result = template.format(name="Alice")  # "Hello Alice!"
        """
        rendered = self.render(**kwargs)
        return self.parse_output(rendered)

    def to_messages(self, system_prompt: str | None = None, **kwargs: Any) -> list[dict[str, str]]:
        r"""Convert template to role-based messages for LLMFactory.

        Args
        ----
            system_prompt: Optional system prompt to include
            **kwargs: Variable values to substitute in template

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries with 'role' and 'content' keys

        Examples
        --------
            template = PromptTemplate("Analyze this data: {{data}}")

            # With system prompt
            messages = template.to_messages(
                system_prompt="You are an expert analyst.",
                data="Q4 sales data"
            )
            # Returns: [
            #   {"role": "system", "content": "You are an expert analyst."},
            #   {"role": "user", "content": "Analyze this data: Q4 sales data"}
            # ]

            # Without system prompt
            messages = template.to_messages(data="Q4 sales data")
            # Returns: [{"role": "user", "content": "Analyze this data: Q4 sales data"}]
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = self.format(**kwargs)
        messages.append({"role": "user", "content": user_content})

        return messages

    def __str__(self) -> str:
        """Return string representation showing template and variables.

        Returns
        -------
        str
            String representation of the template
        """
        return f"PromptTemplate(vars={self.input_vars})"

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns
        -------
        str
            Detailed string representation for debugging
        """
        return f"PromptTemplate(template='{self.template[:50]}...', input_vars={self.input_vars})"

    def __add__(self, other: str | PromptTemplate) -> PromptTemplate:
        """Add text or another template using + operator (builder pattern).

        Args
        ----
            other: Text string or PromptTemplate to append

        Returns
        -------
        PromptTemplate
            New PromptTemplate instance with combined content and variables

        Examples
        --------
            # String composition
            template = PromptTemplate("Hello {{name}}") + "!"

            # Template composition (builder pattern)
            base = PromptTemplate("Analyze {{data}}")
            tools = PromptTemplate("\\nTools: {{tools}}")
            composed = base + tools  # Merges templates and variables
        """
        if isinstance(other, PromptTemplate):
            # Compose two templates - merge content and variables
            enhanced_template = self.template + other.template
            # Merge input variables from both templates
            merged_vars = list(dict.fromkeys(self.input_vars + other.input_vars))
            return PromptTemplate(enhanced_template, merged_vars)

        # String composition
        enhanced_template = self.template + other
        return PromptTemplate(enhanced_template, self.input_vars)

    def add(self, text: str | PromptTemplate) -> PromptTemplate:
        """Add text or template to template (alias for + operator).

        Args
        ----
            text: Text string or PromptTemplate to append

        Returns
        -------
        PromptTemplate
            New PromptTemplate instance with enhanced content
        """
        return self + text
