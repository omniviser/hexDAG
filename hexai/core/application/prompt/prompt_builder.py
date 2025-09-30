"""Clean PromptBuilder that extends Messages with prompt-specific features."""

from collections.abc import Callable
from typing import Any

from hexai.core.application.prompt.messages import Messages
from hexai.core.application.prompt.template_utils import extract_variables


class PromptBuilder:
    """Enhances Messages with prompt-specific features.

    This builder provides:
    - Few-shot example management
    - Variable tracking from templates
    - Custom example formatting

    For simple message chains, use Messages directly:
        messages = Messages().add_system("Hi").add_human("Hello")

    Use PromptBuilder when you need examples or variable tracking:
        builder = PromptBuilder()
        builder.messages.add_system("You are {{role}}")  # Direct access
        builder.track_variables()                        # Track {{role}}
        builder.with_example("2+2?", "4")               # Add examples
    """

    def __init__(self, messages: Messages | None = None) -> None:
        """Initialize prompt builder.

        Args:
            messages: Optional Messages instance to enhance (creates new if None)
        """
        # Exposed Messages - users can access directly
        self.messages = messages or Messages()

        # Prompt-specific features
        self.examples: list[tuple[str, str]] = []
        self.variables: set[str] = set()
        self.example_formatter: Callable[[str, str], str] | None = None

    def track_variables(self) -> "PromptBuilder":
        """Extract and track all template variables from current messages.

        Returns:
            Self for method chaining

        Examples:
            >>> builder.messages.add_system("You are {{role}}")
            >>> builder.track_variables()  # Now tracks 'role'
            >>> print(builder.get_variables())  # ['role']
        """
        for msg in self.messages:
            self.variables.update(extract_variables(msg.content))
        return self

    def with_example(self, input_text: str, output_text: str) -> "PromptBuilder":
        """Add a few-shot example.

        Args:
            input_text: Example input
            output_text: Example output

        Returns:
            Self for method chaining

        Examples:
            >>> builder.with_example("What's 2+2?", "4")
        """
        self.examples.append((input_text, output_text))
        self.variables.update(extract_variables(input_text))
        self.variables.update(extract_variables(output_text))
        return self

    def with_examples(self, examples: list[dict[str, str]]) -> "PromptBuilder":
        """Add multiple few-shot examples.

        Args:
            examples: List of dicts with 'input' and 'output' keys

        Returns:
            Self for method chaining

        Examples:
            >>> examples = [
            ...     {"input": "What's 2+2?", "output": "4"},
            ...     {"input": "What's 3+3?", "output": "6"}
            ... ]
            >>> builder.with_examples(examples)
        """
        for ex in examples:
            self.with_example(ex["input"], ex["output"])
        return self

    def set_example_formatter(self, formatter: Callable[[str, str], str]) -> "PromptBuilder":
        """Set custom formatter for examples.

        Args:
            formatter: Function (input, output) -> formatted_string

        Returns:
            Self for method chaining

        Examples:
            >>> def formatter(inp, out):
            ...     return f"Q: {inp}\\nA: {out}"
            >>> builder.set_example_formatter(formatter)
        """
        self.example_formatter = formatter
        return self

    def build(self) -> Messages:
        """Build final Messages with examples included.

        Examples are added before regular messages.

        Returns:
            Complete Messages collection
        """
        result = Messages()

        # Add examples as messages
        for input_text, output_text in self.examples:
            if self.example_formatter:
                formatted = self.example_formatter(input_text, output_text)
                result.add_human(formatted)
            else:
                result.add_human(input_text)
                result.add_ai(output_text)

        # Add regular messages
        result.extend(self.messages)

        return result

    def to_chat_template(self) -> Any:
        """Convert to ChatPromptTemplate.

        Returns:
            ChatPromptTemplate with messages and examples
        """
        from hexai.core.application.prompt.template import ChatPromptTemplate

        return ChatPromptTemplate(messages=self.build())

    def get_variables(self) -> list[str]:
        """Get all tracked template variables.

        Returns:
            Sorted list of variable names
        """
        return sorted(self.variables)

    def clear_examples(self) -> "PromptBuilder":
        """Clear examples but keep messages.

        Returns:
            Self for method chaining
        """
        self.examples.clear()
        return self

    def clear_all(self) -> "PromptBuilder":
        """Clear everything.

        Returns:
            Self for method chaining
        """
        self.messages.clear()
        self.examples.clear()
        self.variables.clear()
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PromptBuilder("
            f"messages={len(self.messages)}, "
            f"examples={len(self.examples)}, "
            f"variables={self.get_variables()})"
        )
