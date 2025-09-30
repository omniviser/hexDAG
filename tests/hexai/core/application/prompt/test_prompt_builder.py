"""Tests for PromptBuilder class.

Tests cover:
- Basic initialization
- Building from existing templates
- Adding system/user/assistant messages
- Adding few-shot examples
- Building ChatPromptTemplate
- Method chaining
"""

from hexai.core.application.prompt.messages import Messages
from hexai.core.application.prompt.prompt_builder import PromptBuilder
from hexai.core.application.prompt.template import ChatPromptTemplate, PromptTemplate


class TestPromptBuilderInit:
    """Test PromptBuilder initialization."""

    def test_init_empty(self):
        """Test creating empty PromptBuilder."""
        builder = PromptBuilder()
        assert isinstance(builder._messages, Messages)
        assert len(builder._messages) == 0

    def test_from_template_string(self):
        """Test creating PromptBuilder from string."""
        builder = PromptBuilder.from_template("Hello {{name}}")
        messages = builder.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello {{name}}"
        assert messages[0].role.value == "human"

    def test_from_template_prompt_template(self):
        """Test creating PromptBuilder from PromptTemplate."""
        template = PromptTemplate("Analyze {{data}}")
        builder = PromptBuilder.from_template(template)
        messages = builder.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Analyze {{data}}"

    def test_from_template_chat_template(self):
        """Test creating PromptBuilder from ChatPromptTemplate."""
        # Create a ChatPromptTemplate with messages
        original_builder = PromptBuilder()
        original_builder.add_system("You are helpful")
        original_builder.add_user("{{query}}")
        original_template = original_builder.build()

        # Build from it
        builder = PromptBuilder.from_template(original_template)
        messages = builder.get_messages()
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[1].role.value == "human"


class TestPromptBuilderAddMethods:
    """Test adding messages to PromptBuilder."""

    def test_add_system(self):
        """Test adding system message."""
        builder = PromptBuilder()
        builder.add_system("You are a helpful assistant")
        messages = builder.get_messages()
        assert len(messages) == 1
        assert messages[0].role.value == "system"
        assert messages[0].content == "You are a helpful assistant"

    def test_add_user(self):
        """Test adding user message."""
        builder = PromptBuilder()
        builder.add_user("What is {{topic}}?")
        messages = builder.get_messages()
        assert len(messages) == 1
        assert messages[0].role.value == "human"
        assert messages[0].content == "What is {{topic}}?"

    def test_add_assistant(self):
        """Test adding assistant message."""
        builder = PromptBuilder()
        builder.add_assistant("I can help with that!")
        messages = builder.get_messages()
        assert len(messages) == 1
        assert messages[0].role.value == "ai"
        assert messages[0].content == "I can help with that!"

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        builder = PromptBuilder()
        builder.add_system("System prompt")
        builder.add_user("User question")
        builder.add_assistant("Assistant response")
        messages = builder.get_messages()
        assert len(messages) == 3


class TestPromptBuilderExamples:
    """Test adding few-shot examples."""

    def test_add_examples_standard_format(self):
        """Test adding examples with standard user/assistant keys."""
        builder = PromptBuilder()
        examples = [
            {"user": "What is 2+2?", "assistant": "4"},
            {"user": "What is 3+3?", "assistant": "6"},
        ]
        builder.add_examples(examples)
        messages = builder.get_messages()
        assert len(messages) == 4  # 2 examples × 2 messages each
        assert messages[0].content == "What is 2+2?"
        assert messages[0].role.value == "human"
        assert messages[1].content == "4"
        assert messages[1].role.value == "ai"
        assert messages[2].content == "What is 3+3?"
        assert messages[3].content == "6"

    def test_add_examples_custom_keys(self):
        """Test adding examples with custom input/output keys."""
        builder = PromptBuilder()
        examples = [{"input": "Hello", "output": "Hi there!"}]
        builder.add_examples(examples, input_key="input", output_key="output")
        messages = builder.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    def test_add_examples_empty_values(self):
        """Test that empty example values are skipped."""
        builder = PromptBuilder()
        examples = [
            {"user": "Valid", "assistant": "Response"},
            {"user": "", "assistant": ""},  # Should be skipped
            {"user": "Another", "assistant": "One"},
        ]
        builder.add_examples(examples)
        messages = builder.get_messages()
        # Only 2 examples should be added (empty one skipped)
        assert len(messages) == 4


class TestPromptBuilderMethodChaining:
    """Test method chaining functionality."""

    def test_add_system_returns_self(self):
        """Test that add_system returns self for chaining."""
        builder = PromptBuilder()
        result = builder.add_system("System")
        assert result is builder

    def test_add_user_returns_self(self):
        """Test that add_user returns self for chaining."""
        builder = PromptBuilder()
        result = builder.add_user("User")
        assert result is builder

    def test_add_assistant_returns_self(self):
        """Test that add_assistant returns self for chaining."""
        builder = PromptBuilder()
        result = builder.add_assistant("Assistant")
        assert result is builder

    def test_add_examples_returns_self(self):
        """Test that add_examples returns self for chaining."""
        builder = PromptBuilder()
        result = builder.add_examples([{"user": "Q", "assistant": "A"}])
        assert result is builder

    def test_fluent_api_chaining(self):
        """Test full fluent API with method chaining."""
        builder = (
            PromptBuilder()
            .add_system("You are helpful")
            .add_examples([{"user": "Hi", "assistant": "Hello"}])
            .add_user("{{query}}")
        )
        messages = builder.get_messages()
        assert len(messages) == 4  # system + 2 example messages + user


class TestPromptBuilderBuild:
    """Test building ChatPromptTemplate."""

    def test_build_empty(self):
        """Test building with no messages."""
        builder = PromptBuilder()
        template = builder.build()
        assert isinstance(template, ChatPromptTemplate)
        # Should have empty messages
        assert len(template._messages) == 0

    def test_build_with_messages(self):
        """Test building with messages."""
        builder = PromptBuilder()
        builder.add_system("System")
        builder.add_user("User")
        template = builder.build()
        assert isinstance(template, ChatPromptTemplate)
        assert len(template._messages) == 2

    def test_build_with_examples(self):
        """Test building with few-shot examples."""
        builder = PromptBuilder()
        builder.add_system("Classify sentiment")
        builder.add_examples([{"user": "I love this!", "assistant": "positive"}])
        builder.add_user("Classify: {{text}}")
        template = builder.build()
        assert len(template._messages) == 4  # system + 2 example + user

    def test_build_multiple_times(self):
        """Test that build can be called multiple times."""
        builder = PromptBuilder()
        builder.add_user("Test")
        template1 = builder.build()
        template2 = builder.build()
        # Should create different instances
        assert template1 is not template2
        # But with same content
        assert len(template1._messages) == len(template2._messages)


class TestPromptBuilderIntegration:
    """Test integration scenarios."""

    def test_sentiment_classification_example(self):
        """Test real-world sentiment classification scenario."""
        builder = PromptBuilder()
        builder.add_system("You are a sentiment classifier")
        builder.add_examples([
            {"user": "I love this!", "assistant": "positive"},
            {"user": "I hate this", "assistant": "negative"},
        ])
        builder.add_user("Classify: {{text}}")
        template = builder.build()

        # Format the template
        messages = template.format_messages(text="This is great")

        # Should have proper ordering: system, examples, final user message
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "I love this!"  # Example 1 user
        assert messages[2]["content"] == "positive"  # Example 1 assistant
        assert messages[3]["content"] == "I hate this"  # Example 2 user
        assert messages[4]["content"] == "negative"  # Example 2 assistant
        assert messages[5]["content"] == "Classify: This is great"  # Final user msg

    def test_with_context_history(self):
        """Test that examples come before context history."""
        builder = PromptBuilder()
        builder.add_system("You are helpful")
        builder.add_examples([{"user": "Example", "assistant": "Response"}])
        builder.add_user("{{query}}")
        template = builder.build()

        # Add context history
        context = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = template.format_messages(context_history=context, query="Current question")

        # Order should be: system, examples, context, current
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Example"  # Example user
        assert messages[2]["content"] == "Response"  # Example assistant
        assert messages[3]["content"] == "Previous question"  # Context history
        assert messages[4]["content"] == "Previous answer"  # Context history
        assert messages[5]["content"] == "Current question"  # Final query

    def test_get_messages_access(self):
        """Test direct access to underlying Messages object."""
        builder = PromptBuilder()
        builder.add_system("Test")
        messages = builder.get_messages()
        assert isinstance(messages, Messages)
        assert len(messages) == 1
        assert messages[0].content == "Test"


class TestPromptBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_add_empty_strings(self):
        """Test adding empty string messages."""
        builder = PromptBuilder()
        builder.add_system("")
        builder.add_user("")
        messages = builder.get_messages()
        assert len(messages) == 2
        assert messages[0].content == ""
        assert messages[1].content == ""

    def test_multiline_content(self):
        """Test messages with multiline content."""
        builder = PromptBuilder()
        multiline = "Line 1\nLine 2\nLine 3"
        builder.add_system(multiline)
        messages = builder.get_messages()
        assert messages[0].content == multiline

    def test_special_characters(self):
        """Test messages with special characters."""
        builder = PromptBuilder()
        special = "Test {{var}} with {braces} and $pecial ch@rs!"
        builder.add_user(special)
        messages = builder.get_messages()
        assert messages[0].content == special

    def test_examples_missing_keys(self):
        """Test examples with missing keys are skipped."""
        builder = PromptBuilder()
        examples = [
            {"user": "Valid", "assistant": "Response"},
            {"user": "Missing output"},  # Missing assistant key
            {"assistant": "Missing input"},  # Missing user key
        ]
        builder.add_examples(examples)
        messages = builder.get_messages()
        # Only the first valid example should be added
        assert len(messages) == 2
