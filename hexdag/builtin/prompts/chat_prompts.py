"""Registrable chat and few-shot prompt templates.

These are pre-configured, reusable prompt templates that can be
referenced in YAML pipelines or composed programmatically.
"""

from collections.abc import Callable
from typing import Any

from hexdag.builtin.prompts.base import (
    ChatFewShotTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
)


class ChatQAPrompt(ChatPromptTemplate):
    """Chat template for question-answering tasks.

    Provides a helpful assistant persona with structured Q&A format.

    Examples
    --------
    Direct usage::

        prompt = ChatQAPrompt()
        messages = prompt.to_messages(
            domain="AI",
            question="What is machine learning?"
        )

    Composition::

        from hexdag.builtin.prompts import ToolPrompt
        full_prompt = ChatQAPrompt() + ToolPrompt()

    YAML usage::

        nodes:
          - type: prompt_node
            name: qa_prompt
            params:
              template: core:chat_qa
              inputs:
                domain: "science"
                question: "{{user_question}}"
    """

    def __init__(self) -> None:
        """Initialize Q&A chat template."""
        super().__init__(
            system_message="You are a helpful expert in {{domain}}. "
            "Provide clear, accurate, and concise answers.",
            human_message="Question: {{question}}",
        )


class ChatAnalysisPrompt(ChatPromptTemplate):
    """Chat template for analytical tasks.

    Encourages step-by-step reasoning and structured output.

    Examples
    --------
        prompt = ChatAnalysisPrompt()
        messages = prompt.to_messages(
            task="sentiment analysis",
            data="Customer feedback: Great product!"
        )
    """

    def __init__(self) -> None:
        """Initialize analysis chat template."""
        super().__init__(
            system_message="You are an expert analyst. "
            "Analyze the given data thoroughly and provide structured insights. "
            "Think step-by-step and explain your reasoning.",
            human_message="Task: {{task}}\n\nData:\n{{data}}\n\nProvide your analysis:",
        )


class ChatConversationalPrompt(ChatPromptTemplate):
    """Conversational chat template with conversation history.

    Designed for multi-turn conversations with context.

    Examples
    --------
        prompt = ChatConversationalPrompt()
        messages = prompt.to_messages(
            bot_name="Assistant",
            user_message="Tell me about quantum computing",
            context_history=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ]
        )
    """

    def __init__(self) -> None:
        """Initialize conversational chat template."""
        super().__init__(
            system_message="You are {{bot_name}}, a friendly and helpful conversational AI. "
            "Maintain context from previous messages and provide engaging responses.",
            human_message="{{user_message}}",
        )


class FewShotClassificationPrompt(FewShotPromptTemplate):
    """Few-shot template for classification tasks.

    Provides examples to guide the model's classification behavior.

    Examples
    --------
    Direct usage::

        examples = [
            {"input": "I love this!", "output": "positive"},
            {"input": "Terrible experience", "output": "negative"},
            {"input": "It's okay", "output": "neutral"}
        ]

        prompt = FewShotClassificationPrompt(examples=examples)
        text = prompt.format(text="Amazing product!")

    YAML usage::

        nodes:
          - type: prompt_node
            name: classifier
            params:
              template: core:fewshot_classification
              examples:
                - input: "Great!"
                  output: "positive"
                - input: "Bad"
                  output: "negative"
              inputs:
                text: "{{review}}"
    """

    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        format_example: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize classification few-shot template.

        Args
        ----
            examples: List of example dicts with 'input' and 'output' keys
            format_example: Optional custom formatter for examples
        """
        if format_example is None:

            def _classification_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Text: {inp}\nClassification: {out}"

            format_example = _classification_formatter

        super().__init__(
            template="Classify the following text:\n\nText: {{text}}\nClassification:",
            examples=examples or [],
            format_example=format_example,
            example_separator="\n\n",
        )


class FewShotExtractionPrompt(FewShotPromptTemplate):
    """Few-shot template for extracting structured information.

    Examples
    --------
        examples = [
            {
                "input": "John Doe, age 30, lives in NYC",
                "output": '{"name": "John Doe", "age": 30, "city": "NYC"}'
            }
        ]

        prompt = FewShotExtractionPrompt(examples=examples)
        text = prompt.format(text="Jane Smith, 25, from LA")
    """

    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        format_example: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize extraction few-shot template."""
        if format_example is None:

            def _extraction_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Input: {inp}\nExtracted: {out}"

            format_example = _extraction_formatter

        super().__init__(
            template="Extract structured information from the following text:\n\nInput: {{text}}\nExtracted:",
            examples=examples or [],
            format_example=format_example,
            example_separator="\n\n",
        )


class ChatFewShotQAPrompt(ChatFewShotTemplate):
    """Chat template with few-shot examples for question answering.

    Combines chat-style interaction with example-based learning.

    Examples
    --------
        examples = [
            {
                "input": "What is Python?",
                "output": "Python is a high-level programming language known for simplicity."
            },
            {
                "input": "What is AI?",
                "output": "AI is the simulation of human intelligence by machines."
            }
        ]

        prompt = ChatFewShotQAPrompt(examples=examples)
        messages = prompt.to_messages(question="What is blockchain?")
    """

    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        format_example: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize chat few-shot Q&A template."""
        if format_example is None:

            def _qa_formatter(ex: dict[str, Any]) -> str:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                return f"Q: {inp}\nA: {out}"

            format_example = _qa_formatter

        super().__init__(
            system_message="You are a knowledgeable assistant. "
            "Answer questions clearly and concisely, following the example format.",
            human_message="Q: {{question}}\nA:",
            examples=examples or [],
            format_example=format_example,
            example_separator="\n\n",
        )


# Factory function for creating custom chat prompts
def create_chat_prompt(
    system_message: str,
    human_message: str,
) -> type[ChatPromptTemplate]:
    """Factory for creating custom chat prompts.

    Examples
    --------
        MyCustomPrompt = create_chat_prompt(
            system_message="You are a {{role}}",
            human_message="{{task}}",
        )
    """

    class CustomChatPrompt(ChatPromptTemplate):
        def __init__(self) -> None:
            super().__init__(system_message=system_message, human_message=human_message)

    return CustomChatPrompt


# Factory for creating custom few-shot prompts
def create_fewshot_prompt(
    template: str,
    examples: list[dict[str, Any]],
    format_example: Callable[[dict[str, Any]], str] | None = None,
) -> type[FewShotPromptTemplate]:
    """Factory for creating custom few-shot prompts.

    Examples
    --------
        MyFewShotPrompt = create_fewshot_prompt(
            template="Translate: {{text}}",
            examples=[
                {"input": "Hello", "output": "Hola"},
                {"input": "Goodbye", "output": "Adiós"}
            ],
        )
    """

    class CustomFewShotPrompt(FewShotPromptTemplate):
        def __init__(self) -> None:
            super().__init__(template=template, examples=examples, format_example=format_example)

    return CustomFewShotPrompt
