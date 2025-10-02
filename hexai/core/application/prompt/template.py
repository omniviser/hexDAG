"""Simple and secure prompt template system for hexAI.

This module provides a lightweight template engine that handles variable substitution and nested
object access without the security risks of full template engines like Jinja2.
"""

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any


class PromptTemplateError(Exception):
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
    # Extract variables from {{ variable }} patterns
    pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
    matches = re.findall(pattern, template)

    # Extract root variable names (before any dots) and deduplicate
    root_vars = []
    for match in matches:
        root_var = match.split(".")[0]
        if root_var not in root_vars:
            root_vars.append(root_var)

    return tuple(root_vars)


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
            raise MissingVariableError(
                f"Missing required template variables: {missing_vars}. "
                f"Required variables: {self.input_vars}. "
                f"Provided variables: {list(kwargs.keys())}"
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
                raise MissingVariableError(f"Cannot access variable '{var_path}': {e}") from e

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

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message with rendered template
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

    def __add__(self, other: str) -> "PromptTemplate":
        """Add text to template using + operator.

        Args
        ----
            other: Text to append to the template

        Returns
        -------
        PromptTemplate
            New PromptTemplate instance with enhanced content
        """
        enhanced_template = self.template + other
        return PromptTemplate(enhanced_template, self.input_vars)

    def add(self, text: str) -> "PromptTemplate":
        """Add text to template (alias for + operator).

        Args
        ----
            text: Text to append to the template

        Returns
        -------
        PromptTemplate
            New PromptTemplate instance with enhanced content
        """
        return self + text


# ---------------------------------------------------------------------------
# Few-Shot Prompt Template
# ---------------------------------------------------------------------------


class FewShotPromptTemplate(PromptTemplate):
    r"""PromptTemplate with built-in support for few-shot examples.

    Simplified implementation that builds the template once during initialization.
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

        # Set default example formatter
        if format_example is None:

            def _default_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Input: {inp}\nOutput: {out}"

            format_example = _default_formatter

        self.format_example = format_example

        # Build the full template with examples
        full_template = self._build_template()

        # Initialize parent with the full template
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

    def __add__(self, other: str) -> "FewShotPromptTemplate":
        """Add text to template using + operator.

        Returns
        -------
        FewShotPromptTemplate
            New FewShotPromptTemplate instance with enhanced content
        """
        enhanced_base_template = self.base_template + other
        return FewShotPromptTemplate(
            enhanced_base_template,
            self.examples,
            format_example=self.format_example,
            example_separator=self.example_separator,
        )

    def add(self, text: str) -> "FewShotPromptTemplate":
        """Add text to template (alias for + operator).

        Returns
        -------
        FewShotPromptTemplate
            New FewShotPromptTemplate instance with enhanced content
        """
        return self + text


# ---------------------------------------------------------------------------
# Chat Prompt Template
# ---------------------------------------------------------------------------


class ChatPromptTemplate(PromptTemplate):
    """Enhanced prompt template supporting multi-message conversations like LangChain.

    Supports:
    - System messages
    - Conversation history from context
    - Multi-message templates
    - Role-based message construction
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

        Examples
        --------
            # Simple system + user message
            template = ChatPromptTemplate(
                system_message="You are an expert {{domain}} analyst.",
                human_message="Analyze this {{data_type}}: {{data}}"
            )

            # Multi-message template
            template = ChatPromptTemplate(messages=[
                {"role": "system", "content": "You are an expert analyst."},
                {"role": "user", "content": "Previous context: {{context}}"},
                {"role": "assistant", "content": "I understand the context."},
                {"role": "user", "content": "Now analyze: {{data}}"}
            ])
        """
        self.system_message = system_message
        self.human_message = human_message
        self.message_templates = messages or []

        # Build combined template for variable extraction
        if not self.message_templates:
            # Build from system + human messages
            if system_message and human_message:
                combined_template = f"{system_message}\n{human_message}"
            elif human_message:
                combined_template = human_message
            elif system_message:
                combined_template = system_message
            else:
                combined_template = ""
        else:
            # Extract content from message templates
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

        # Add system message if provided
        if system_prompt:
            system_content = self._render_template(system_prompt, **kwargs)
            messages.append({"role": "system", "content": system_content})
        elif self.system_message:
            system_content = self._render_template(self.system_message, **kwargs)
            messages.append({"role": "system", "content": system_content})

        # Add conversation history from context
        if context_history:
            messages.extend(context_history)

        # Add message templates or human message
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
    def from_messages(cls, messages: list[dict[str, str]]) -> "ChatPromptTemplate":
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

    def __add__(self, other: str) -> "ChatPromptTemplate":
        """Add text to system message using + operator.

        Returns
        -------
        ChatPromptTemplate
            New ChatPromptTemplate instance with enhanced system message
        """
        enhanced_system = (self.system_message or "") + other
        return ChatPromptTemplate(
            messages=self.message_templates,
            system_message=enhanced_system,
            human_message=self.human_message,
        )

    def add(self, text: str) -> "ChatPromptTemplate":
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
        # Store example-related attributes
        self.examples = examples or []
        self.example_separator = example_separator

        # Set default example formatter
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

        # Initialize parent with enhanced system message - EXACT same API otherwise
        super().__init__(
            messages=messages,
            system_message=enhanced_system_message,
            human_message=human_message,
            **kwargs,
        )

        # Store original system message for add_example functionality
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

    def to_messages(
        self,
        system_prompt: str | None = None,
        context_history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Convert to role-based messages with examples, system prompt, and conversation history.

        Args
        ----
            system_prompt: Optional additional system prompt (added to examples + system_message)
            context_history: Previous conversation messages
            **kwargs: Variable values for template substitution

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries with examples, system message, history, and current input
        """
        messages = []

        # 1. Build comprehensive system message
        system_parts = []

        # Start with the enhanced system message (examples + original system message)
        if self.system_message:
            rendered_system = self._render_template(self.system_message, **kwargs)
            system_parts.append(rendered_system)

        # Add additional system prompt if provided
        if system_prompt:
            rendered_additional = self._render_template(system_prompt, **kwargs)
            system_parts.append(rendered_additional)

        # Combine system parts
        if system_parts:
            system_content = self.example_separator.join(system_parts)
            messages.append({"role": "system", "content": system_content})

        # 2. Add conversation history (past messages)
        if context_history:
            messages.extend(context_history)

        # 3. Add message templates or human message
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

    def __str__(self) -> str:
        """Return string representation of the template.

        Returns
        -------
        str
            String representation showing template details
        """
        system_preview = f"{self.system_message[:50]}..." if self.system_message else "None"
        human_preview = f"{self.human_message[:50]}..." if self.human_message else "None"
        return (
            f"ChatFewShotTemplate(system='{system_preview}', "
            f"  human='{human_preview}', "
            f"examples={len(self.examples)})"
        )

    def __add__(self, other: str) -> "ChatFewShotTemplate":
        """Add text to original system message using + operator.

        Returns
        -------
        ChatFewShotTemplate
            New ChatFewShotTemplate instance with enhanced system message
        """
        enhanced_original_system = (self._original_system_message or "") + other
        return ChatFewShotTemplate(
            messages=self.message_templates,
            system_message=enhanced_original_system,
            human_message=self.human_message,
            examples=self.examples,
            example_separator=self.example_separator,
            format_example=self.format_example,
        )

    def add(self, text: str) -> "ChatFewShotTemplate":
        """Add text to original system message (alias for + operator).

        Returns
        -------
        ChatFewShotTemplate
            New ChatFewShotTemplate instance with enhanced system message
        """
        return self + text
