"""Base prompt template classes for building composable prompts.

These are the foundational prompt types that all builtin prompts inherit from.
They provide the core functionality for chat-style prompts, few-shot learning,
and combinations thereof.

Moved from core.orchestration.prompt.template to keep core minimal and make
these templates part of the builtin library.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from hexdag.kernel.orchestration.prompt.template import PromptTemplate, _extract_variables_cached


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
