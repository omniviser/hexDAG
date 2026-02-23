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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from hexdag.kernel.exceptions import (
    MissingVariableError,  # noqa: F401
    PromptTemplateError,  # noqa: F401
)

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


# ---------------------------------------------------------------------------
# Advanced Template Types
# ---------------------------------------------------------------------------


class FewShotPromptTemplate(PromptTemplate):
    r"""PromptTemplate with built-in support for few-shot examples.

    Simplified implementation that builds the template once during initialization.

    Examples
    --------
    Basic few-shot template::

        template = FewShotPromptTemplate(
            template="Classify: {{text}}",
            examples=[
                {"input": "Great product!", "output": "positive"},
                {"input": "Terrible", "output": "negative"}
            ]
        )

        result = template.format(text="Amazing!")
        # Output includes examples + prompt

    Custom example formatting::

        def custom_formatter(ex):
            return f"Q: {ex['input']}\nA: {ex['output']}"

        template = FewShotPromptTemplate(
            template="Q: {{question}}\nA:",
            examples=[...],
            format_example=custom_formatter
        )
    """

    def __init__(
        self,
        template: str,
        examples: list[dict[str, Any]] | None = None,
        *,
        format_example: Callable[[dict[str, Any]], str] | None = None,
        example_separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        r"""Initialize FewShotPromptTemplate with examples.

        Args
        ----
            template: Base template string with {{variables}}
            examples: List of example dicts (typically with 'input' and 'output' keys)
            format_example: Function to format each example (optional)
            example_separator: String to separate examples (default: "\n\n")
            **kwargs: Additional arguments passed to PromptTemplate
        """
        self.examples: list[dict[str, Any]] = examples or []
        self.example_separator = example_separator
        self.base_template = template

        if format_example is None:

            def _default_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Input: {inp}\nOutput: {out}"

            format_example = _default_formatter

        self.format_example = format_example

        full_template = self._build_template()

        super().__init__(full_template, **kwargs)

    def _build_template(self) -> str:
        """Build the complete template with examples prepended.

        Returns
        -------
        str
            Complete template string with examples and base template
        """
        if not self.examples:
            return self.base_template

        examples_text = self.example_separator.join(
            self.format_example(example) for example in self.examples
        )

        return f"{examples_text}{self.example_separator}{self.base_template}"

    def add_example(self, example: dict[str, Any]) -> None:
        """Add an example and rebuild the template.

        Args
        ----
            example: Example dict to add
        """
        self.examples.append(example)

        # Rebuild the complete template
        self.template = self._build_template()

        # Re-extract variables from the new template using cached function
        self.input_vars = list(_extract_variables_cached(self.template))

    def __add__(self, other: str | PromptTemplate) -> FewShotPromptTemplate:
        """Add text or template using + operator.

        Returns
        -------
        FewShotPromptTemplate
            New FewShotPromptTemplate instance with enhanced content
        """
        if isinstance(other, PromptTemplate):
            enhanced_base_template = self.base_template + other.template
        else:
            enhanced_base_template = self.base_template + other

        return FewShotPromptTemplate(
            enhanced_base_template,
            self.examples,
            format_example=self.format_example,
            example_separator=self.example_separator,
        )

    def add(self, text: str | PromptTemplate) -> FewShotPromptTemplate:
        """Add text to template (alias for + operator).

        Returns
        -------
        FewShotPromptTemplate
            New FewShotPromptTemplate instance with enhanced content
        """
        return self + text


class ChatPromptTemplate(PromptTemplate):
    """Enhanced prompt template supporting multi-message conversations like LangChain.

    Supports:
    - System messages
    - Conversation history from context
    - Multi-message templates
    - Role-based message construction

    Examples
    --------
    Simple system + user::

        template = ChatPromptTemplate(
            system_message="You are an expert {{domain}} analyst.",
            human_message="Analyze this {{data_type}}: {{data}}"
        )

        messages = template.to_messages(
            domain="financial",
            data_type="report",
            data="Q4 earnings..."
        )

    Multi-message template::

        template = ChatPromptTemplate(messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": "Context: {{context}}"},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "Analyze: {{data}}"}
        ])
    """

    def __init__(
        self,
        messages: list[dict[str, str]] | None = None,
        system_message: str | None = None,
        human_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatPromptTemplate with multiple message types.

        Args
        ----
            messages: List of message templates with role and content
            system_message: System message template (optional)
            human_message: Human/user message template (optional)
            **kwargs: Additional arguments passed to PromptTemplate
        """
        self.system_message = system_message
        self.human_message = human_message
        self.message_templates = messages or []

        if not self.message_templates:
            if system_message and human_message:
                combined_template = f"{system_message}\n{human_message}"
            elif human_message:
                combined_template = human_message
            elif system_message:
                combined_template = system_message
            else:
                combined_template = ""
        else:
            combined_template = "\n".join(msg.get("content", "") for msg in self.message_templates)

        super().__init__(combined_template, **kwargs)

    def to_messages(
        self,
        system_prompt: str | None = None,
        context_history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Convert to role-based messages with conversation history support.

        Args
        ----
            system_prompt: Optional system prompt to include
            context_history: Previous conversation messages from context
            **kwargs: Variable values for template substitution

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries ready for LLM
        """
        messages = []

        if system_prompt:
            system_content = self._render_template(system_prompt, **kwargs)
            messages.append({"role": "system", "content": system_content})
        elif self.system_message:
            system_content = self._render_template(self.system_message, **kwargs)
            messages.append({"role": "system", "content": system_content})

        if context_history:
            messages.extend(context_history)

        if self.message_templates:
            for msg_template in self.message_templates:
                role = msg_template.get("role", "user")
                content_template = msg_template.get("content", "")
                rendered_content = self._render_template(content_template, **kwargs)
                messages.append({"role": role, "content": rendered_content})
        elif self.human_message:
            human_content = self._render_template(self.human_message, **kwargs)
            messages.append({"role": "user", "content": human_content})

        return messages

    def _render_template(self, template: str, **kwargs: Any) -> str:
        """Render a single template string with variables.

        Returns
        -------
        str
            Rendered template string with variables substituted
        """
        # Use the parent class variable substitution logic
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"

        def replace_var(match: Any) -> str:
            var_path = match.group(1).strip()
            try:
                value = self._get_nested_value(kwargs, var_path)
                return str(value)
            except (KeyError, AttributeError):
                return f"{{{{{var_path}}}}}"  # Keep unreplaced variables

        return re.sub(pattern, replace_var, template)

    @classmethod
    def from_messages(cls, messages: list[dict[str, str]]) -> ChatPromptTemplate:
        """Create ChatPromptTemplate from a list of message dictionaries.

        Args
        ----
            messages: List of messages with 'role' and 'content' keys

        Returns
        -------
        ChatPromptTemplate
            ChatPromptTemplate instance
        """
        return cls(messages=messages)

    def __add__(self, other: str | PromptTemplate) -> ChatPromptTemplate:
        """Add text or template to system message using + operator.

        Returns
        -------
        ChatPromptTemplate
            New ChatPromptTemplate instance with enhanced system message
        """
        if isinstance(other, PromptTemplate):
            enhanced_system = (self.system_message or "") + other.template
        else:
            enhanced_system = (self.system_message or "") + other

        return ChatPromptTemplate(
            messages=self.message_templates,
            system_message=enhanced_system,
            human_message=self.human_message,
        )

    def add(self, text: str | PromptTemplate) -> ChatPromptTemplate:
        """Add text to system message (alias for + operator).

        Returns
        -------
        ChatPromptTemplate
            New ChatPromptTemplate instance with enhanced system message
        """
        return self + text


class ChatFewShotTemplate(ChatPromptTemplate):
    """Chat template with few-shot examples support.

    Combines the role-based messaging of ChatPromptTemplate with the example formatting of
    FewShotPromptTemplate. Has the EXACT same API as ChatPromptTemplate, just enhanced with
    examples.

    Examples
    --------
    Chat with examples::

        template = ChatFewShotTemplate(
            system_message="You are a classifier",
            human_message="Classify: {{text}}",
            examples=[
                {"input": "Great!", "output": "positive"},
                {"input": "Bad", "output": "negative"}
            ]
        )

        messages = template.to_messages(text="Amazing!")
        # System message includes examples
    """

    def __init__(
        self,
        messages: list[dict[str, str]] | None = None,
        system_message: str | None = None,
        human_message: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        example_separator: str = "\n\n",
        format_example: Callable[[dict[str, Any]], str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ChatFewShotTemplate with same API as ChatPromptTemplate plus examples.

        Args
        ----
            messages: List of message templates with role and content (same as ChatPromptTemplate)
            system_message: System message template (same as ChatPromptTemplate)
            human_message: Human/user message template (same as ChatPromptTemplate)
            examples: List of example dicts for few-shot learning (NEW)
            example_separator: String to separate examples (NEW)
            format_example: Function to format each example (NEW)
            **kwargs: Additional arguments passed to ChatPromptTemplate
        """
        self.examples = examples or []
        self.example_separator = example_separator

        if format_example is None:

            def _default_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Input: {inp}\nOutput: {out}"

            format_example = _default_formatter
        self.format_example = format_example

        # If we have examples, enhance the system message
        enhanced_system_message = system_message
        if self.examples:
            examples_text = self.example_separator.join(
                self.format_example(example) for example in self.examples
            )

            if system_message:
                enhanced_system_message = f"{examples_text}{self.example_separator}{system_message}"
            else:
                enhanced_system_message = examples_text

        super().__init__(
            messages=messages,
            system_message=enhanced_system_message,
            human_message=human_message,
            **kwargs,
        )

        self._original_system_message = system_message

    def add_example(self, example: dict[str, Any]) -> None:
        """Add an example and rebuild system message."""
        self.examples.append(example)

        # Rebuild system message with new examples
        examples_text = self.example_separator.join(self.format_example(ex) for ex in self.examples)

        if self._original_system_message:
            self.system_message = (
                f"{examples_text}{self.example_separator}{self._original_system_message}"
            )
        else:
            self.system_message = examples_text
