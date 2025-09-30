"""PromptBuilder - Fluent API for constructing complex prompts with messages and examples.

This module provides a builder pattern for creating ChatPromptTemplate instances with:
- System messages and sections
- User/assistant message pairs
- Few-shot examples
- Tool documentation and instructions
"""

from hexai.core.application.prompt.messages import Messages
from hexai.core.application.prompt.template import ChatPromptTemplate, PromptTemplate


class PromptBuilder:
    """Fluent builder for constructing complex prompts with proper message structure.

    Key features:
    - Start from existing template or build fresh
    - Add system messages separately from user prompts
    - Add few-shot examples as proper message pairs
    - Method chaining for clean syntax
    - Builds to ChatPromptTemplate with proper message structure

    Examples
    --------
    Build from scratch:
        >>> builder = PromptBuilder()
        >>> builder.add_system("You are a helpful assistant")
        >>> builder.add_user("What is {{topic}}?")
        >>> template = builder.build()

    Start from existing template:
        >>> builder = PromptBuilder.from_template("Analyze: {{input}}")
        >>> builder.add_system("## Tools\\n- search()\\n- calculator()")
        >>> template = builder.build()

    Add examples:
        >>> builder = PromptBuilder()
        >>> builder.add_system("Classify sentiment")
        >>> builder.add_examples([
        ...     {"user": "I love this!", "assistant": "positive"},
        ...     {"user": "This is terrible", "assistant": "negative"}
        ... ])
        >>> builder.add_user("Classify: {{text}}")
        >>> template = builder.build()
    """

    def __init__(self) -> None:
        """Initialize empty PromptBuilder."""
        self._messages = Messages()

    @classmethod
    def from_template(cls, template: str | PromptTemplate | ChatPromptTemplate) -> "PromptBuilder":
        """Create builder from existing template.

        Parameters
        ----------
        template : str | PromptTemplate | ChatPromptTemplate
            Existing template to start from

        Returns
        -------
        PromptBuilder
            New builder initialized with template's messages

        Examples
        --------
        From string:
            >>> builder = PromptBuilder.from_template("Hello {{name}}")

        From PromptTemplate:
            >>> pt = PromptTemplate("Analyze {{data}}")
            >>> builder = PromptBuilder.from_template(pt)

        From ChatPromptTemplate:
            >>> ct = ChatPromptTemplate(system_message="You are helpful", human_message="{{query}}")
            >>> builder = PromptBuilder.from_template(ct)
        """
        builder = cls()

        if isinstance(template, str):
            # Simple string - add as user message
            builder._messages.add_human(template)
        elif isinstance(template, PromptTemplate) and not isinstance(template, ChatPromptTemplate):
            # PromptTemplate - extract template string
            builder._messages.add_human(template.template)
        elif isinstance(template, ChatPromptTemplate):
            # ChatPromptTemplate - copy its messages
            if hasattr(template, "_messages") and template._messages:
                # Copy messages from ChatPromptTemplate
                for msg in template._messages:
                    builder._messages._messages.append(msg)
            else:
                # Fallback: use legacy attributes
                if hasattr(template, "system_message") and template.system_message:
                    builder._messages.add_system(template.system_message)
                if hasattr(template, "human_message") and template.human_message:
                    builder._messages.add_human(template.human_message)

        return builder

    def add_system(self, content: str) -> "PromptBuilder":
        """Add a system message.

        System messages set context and instructions for the LLM.

        Parameters
        ----------
        content : str
            System message content

        Returns
        -------
        PromptBuilder
            Self for method chaining

        Examples
        --------
            >>> builder.add_system("You are a helpful coding assistant")
        """
        self._messages.add_system(content)
        return self

    def add_user(self, content: str) -> "PromptBuilder":
        """Add a user message.

        Parameters
        ----------
        content : str
            User message content (can contain {{variables}})

        Returns
        -------
        PromptBuilder
            Self for method chaining

        Examples
        --------
            >>> builder.add_user("Explain {{concept}} in simple terms")
        """
        self._messages.add_human(content)
        return self

    def add_assistant(self, content: str) -> "PromptBuilder":
        """Add an assistant message.

        Used for few-shot examples or conversation context.

        Parameters
        ----------
        content : str
            Assistant message content

        Returns
        -------
        PromptBuilder
            Self for method chaining

        Examples
        --------
            >>> builder.add_assistant("Sure, I can help with that!")
        """
        self._messages.add_ai(content)
        return self

    def add_examples(
        self, examples: list[dict[str, str]], input_key: str = "user", output_key: str = "assistant"
    ) -> "PromptBuilder":
        """Add few-shot examples as user/assistant message pairs.

        Parameters
        ----------
        examples : list[dict[str, str]]
            List of example dictionaries with input/output keys
        input_key : str
            Key for user message (default: "user")
        output_key : str
            Key for assistant message (default: "assistant")

        Returns
        -------
        PromptBuilder
            Self for method chaining

        Examples
        --------
        Standard format:
            >>> builder.add_examples([
            ...     {"user": "What is 2+2?", "assistant": "4"},
            ...     {"user": "What is 3+3?", "assistant": "6"}
            ... ])

        Custom keys:
            >>> builder.add_examples(
            ...     [{"input": "Hello", "output": "Hi there!"}],
            ...     input_key="input",
            ...     output_key="output"
            ... )
        """
        for example in examples:
            user_msg = example.get(input_key, "")
            assistant_msg = example.get(output_key, "")
            if user_msg and assistant_msg:
                self._messages.add_human(user_msg)
                self._messages.add_ai(assistant_msg)
        return self

    def build(self) -> ChatPromptTemplate:
        """Build final ChatPromptTemplate from accumulated messages.

        Returns
        -------
        ChatPromptTemplate
            Complete template ready for use with LLM nodes

        Examples
        --------
            >>> builder = PromptBuilder()
            >>> builder.add_system("Be helpful")
            >>> builder.add_user("{{query}}")
            >>> template = builder.build()
        """
        return ChatPromptTemplate(messages=self._messages)

    def get_messages(self) -> Messages:
        """Get the underlying Messages object.

        Useful for advanced manipulation or inspection.

        Returns
        -------
        Messages
            The Messages object being built
        """
        return self._messages
