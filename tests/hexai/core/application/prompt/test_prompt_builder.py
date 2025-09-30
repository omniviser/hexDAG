"""Tests for PromptBuilder class.

Tests cover:
- Basic initialization and message handling
- Variable tracking functionality
- Few-shot example management
- Custom example formatting
- Building final Messages
- Integration with ChatPromptTemplate
- Clear operations
"""

from hexai.core.application.prompt.messages import Messages
from hexai.core.application.prompt.prompt_builder import PromptBuilder


class TestPromptBuilderInit:
    """Test PromptBuilder initialization."""

    def test_init_empty(self):
        """Test creating empty PromptBuilder."""
        builder = PromptBuilder()
        assert isinstance(builder.messages, Messages)
        assert len(builder.messages) == 0
        assert builder.examples == []
        assert builder.variables == set()
        assert builder.example_formatter is None

    def test_init_with_messages(self):
        """Test creating PromptBuilder with existing Messages."""
        messages = Messages().add_system("Hello").add_human("Hi")
        builder = PromptBuilder(messages)
        assert builder.messages is messages
        assert len(builder.messages) == 2


class TestPromptBuilderVariableTracking:
    """Test variable tracking functionality."""

    def test_track_variables_single_message(self):
        """Test tracking variables from a single message."""
        builder = PromptBuilder()
        builder.messages.add_system("You are a {{role}} assistant")
        builder.track_variables()

        assert "role" in builder.variables
        assert builder.get_variables() == ["role"]

    def test_track_variables_multiple_messages(self):
        """Test tracking variables from multiple messages."""
        builder = PromptBuilder()
        builder.messages.add_system("You are a {{role}} assistant")
        builder.messages.add_human("Analyze this {{data}}")
        builder.track_variables()

        assert builder.variables == {"role", "data"}
        assert builder.get_variables() == ["data", "role"]  # Sorted

    def test_track_variables_no_variables(self):
        """Test tracking when no variables present."""
        builder = PromptBuilder()
        builder.messages.add_system("You are helpful")
        builder.track_variables()

        assert builder.variables == set()
        assert builder.get_variables() == []

    def test_track_variables_duplicate_variables(self):
        """Test that duplicate variables are deduplicated."""
        builder = PromptBuilder()
        builder.messages.add_system("{{name}} is {{name}}")
        builder.track_variables()

        assert builder.variables == {"name"}
        assert builder.get_variables() == ["name"]

    def test_track_variables_method_chaining(self):
        """Test that track_variables returns self for chaining."""
        builder = PromptBuilder()
        builder.messages.add_system("{{x}}")
        result = builder.track_variables()
        assert result is builder


class TestPromptBuilderExamples:
    """Test few-shot example management."""

    def test_with_example_basic(self):
        """Test adding a single example."""
        builder = PromptBuilder()
        builder.with_example("What's 2+2?", "4")

        assert len(builder.examples) == 1
        assert builder.examples[0] == ("What's 2+2?", "4")

    def test_with_example_tracks_variables(self):
        """Test that adding examples tracks their variables."""
        builder = PromptBuilder()
        builder.with_example("Input: {{x}}", "Output: {{y}}")

        assert builder.variables == {"x", "y"}

    def test_with_examples_list(self):
        """Test adding multiple examples from list."""
        builder = PromptBuilder()
        examples = [
            {"input": "What's 2+2?", "output": "4"},
            {"input": "What's 3+3?", "output": "6"},
        ]
        builder.with_examples(examples)

        assert len(builder.examples) == 2
        assert builder.examples[0] == ("What's 2+2?", "4")
        assert builder.examples[1] == ("What's 3+3?", "6")

    def test_with_example_method_chaining(self):
        """Test that with_example returns self for chaining."""
        builder = PromptBuilder()
        result = builder.with_example("Q", "A")
        assert result is builder

    def test_with_examples_method_chaining(self):
        """Test that with_examples returns self for chaining."""
        builder = PromptBuilder()
        result = builder.with_examples([{"input": "Q", "output": "A"}])
        assert result is builder


class TestPromptBuilderFormatting:
    """Test example formatting functionality."""

    def test_set_example_formatter(self):
        """Test setting a custom formatter."""

        def formatter(inp: str, out: str) -> str:
            return f"Q: {inp}\nA: {out}"

        builder = PromptBuilder()
        builder.set_example_formatter(formatter)

        assert builder.example_formatter is formatter

    def test_set_example_formatter_method_chaining(self):
        """Test that set_example_formatter returns self for chaining."""
        builder = PromptBuilder()
        result = builder.set_example_formatter(lambda i, o: f"{i}={o}")
        assert result is builder

    def test_custom_formatter_in_build(self):
        """Test that custom formatter is used when building."""

        def formatter(inp: str, out: str) -> str:
            return f"Example: {inp} -> {out}"

        builder = PromptBuilder()
        builder.set_example_formatter(formatter)
        builder.with_example("2+2", "4")

        result = builder.build()
        assert len(result) == 1
        assert result[0].content == "Example: 2+2 -> 4"
        assert result[0].role.value == "human"


class TestPromptBuilderBuild:
    """Test building final Messages."""

    def test_build_empty(self):
        """Test building with no messages or examples."""
        builder = PromptBuilder()
        result = builder.build()

        assert isinstance(result, Messages)
        assert len(result) == 0

    def test_build_only_messages(self):
        """Test building with only messages."""
        builder = PromptBuilder()
        builder.messages.add_system("Hello")
        builder.messages.add_human("Hi")

        result = builder.build()
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi"

    def test_build_only_examples(self):
        """Test building with only examples."""
        builder = PromptBuilder()
        builder.with_example("Q1", "A1")
        builder.with_example("Q2", "A2")

        result = builder.build()
        assert len(result) == 4  # 2 examples × 2 messages each
        assert result[0].content == "Q1"
        assert result[0].role.value == "human"
        assert result[1].content == "A1"
        assert result[1].role.value == "ai"
        assert result[2].content == "Q2"
        assert result[3].content == "A2"

    def test_build_examples_before_messages(self):
        """Test that examples appear before regular messages."""
        builder = PromptBuilder()
        builder.messages.add_human("Main query")
        builder.with_example("Example Q", "Example A")

        result = builder.build()
        assert len(result) == 3
        assert result[0].content == "Example Q"  # Example first
        assert result[1].content == "Example A"
        assert result[2].content == "Main query"  # Then regular messages

    def test_build_with_custom_formatter(self):
        """Test building with custom example formatter."""

        def formatter(inp: str, out: str) -> str:
            return f"{inp} === {out}"

        builder = PromptBuilder()
        builder.set_example_formatter(formatter)
        builder.with_example("Input", "Output")
        builder.messages.add_human("Query")

        result = builder.build()
        assert len(result) == 2
        assert result[0].content == "Input === Output"
        assert result[0].role.value == "human"
        assert result[1].content == "Query"

    def test_build_does_not_modify_original(self):
        """Test that building doesn't modify the builder's messages."""
        builder = PromptBuilder()
        builder.messages.add_human("Test")
        builder.with_example("Q", "A")

        original_count = len(builder.messages)
        result = builder.build()

        assert len(result) == 3  # 2 example messages + 1 regular
        assert len(builder.messages) == original_count  # Original unchanged


class TestPromptBuilderClear:
    """Test clear operations."""

    def test_clear_examples(self):
        """Test clearing only examples."""
        builder = PromptBuilder()
        builder.messages.add_system("System")
        builder.with_example("Q", "A")
        builder.variables.add("test")

        builder.clear_examples()

        assert len(builder.examples) == 0
        assert len(builder.messages) == 1  # Messages preserved
        assert "test" in builder.variables  # Variables preserved

    def test_clear_examples_method_chaining(self):
        """Test that clear_examples returns self for chaining."""
        builder = PromptBuilder()
        result = builder.clear_examples()
        assert result is builder

    def test_clear_all(self):
        """Test clearing everything."""
        builder = PromptBuilder()
        builder.messages.add_system("System")
        builder.with_example("Q", "A")
        builder.variables.add("test")

        builder.clear_all()

        assert len(builder.examples) == 0
        assert len(builder.messages) == 0
        assert len(builder.variables) == 0

    def test_clear_all_method_chaining(self):
        """Test that clear_all returns self for chaining."""
        builder = PromptBuilder()
        result = builder.clear_all()
        assert result is builder


class TestPromptBuilderIntegration:
    """Test integration with other components."""

    def test_to_chat_template(self):
        """Test conversion to ChatPromptTemplate."""
        builder = PromptBuilder()
        builder.messages.add_system("You are helpful")
        builder.with_example("Hi", "Hello")

        template = builder.to_chat_template()

        # Should have examples + messages
        assert len(template._messages) == 3  # 2 example msgs + 1 system

    def test_direct_message_access(self):
        """Test that users can directly access and modify messages."""
        builder = PromptBuilder()
        builder.messages.add_system("Initial")

        # Direct access to messages
        assert len(builder.messages) == 1
        builder.messages.add_human("Added directly")
        assert len(builder.messages) == 2

        # Changes reflected in build
        result = builder.build()
        assert len(result) == 2

    def test_fluent_api_chaining(self):
        """Test fluent API with method chaining."""
        builder = (
            PromptBuilder().with_example("Q1", "A1").with_example("Q2", "A2").track_variables()
        )

        builder.messages.add_system("System").add_human("Query")

        result = builder.build()
        assert len(result) == 6  # 4 example messages + 2 regular


class TestPromptBuilderRepr:
    """Test string representation."""

    def test_repr_empty(self):
        """Test repr of empty builder."""
        builder = PromptBuilder()
        repr_str = repr(builder)

        assert "PromptBuilder" in repr_str
        assert "messages=0" in repr_str
        assert "examples=0" in repr_str

    def test_repr_with_content(self):
        """Test repr with messages and examples."""
        builder = PromptBuilder()
        builder.messages.add_system("{{role}}")
        builder.with_example("Q", "A")
        builder.track_variables()

        repr_str = repr(builder)

        assert "messages=1" in repr_str
        assert "examples=1" in repr_str
        assert "role" in repr_str


class TestPromptBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_example_strings(self):
        """Test adding examples with empty strings."""
        builder = PromptBuilder()
        builder.with_example("", "")

        assert len(builder.examples) == 1
        assert builder.examples[0] == ("", "")

    def test_multiline_examples(self):
        """Test examples with multiline strings."""
        builder = PromptBuilder()
        builder.with_example("Line 1\nLine 2", "Answer\nMultiline")

        result = builder.build()
        assert len(result) == 2
        assert "\n" in result[0].content
        assert "\n" in result[1].content

    def test_special_characters_in_examples(self):
        """Test examples with special characters."""
        builder = PromptBuilder()
        builder.with_example("What's {{x}}?", "{{x}} = 42")

        assert "x" in builder.variables

        result = builder.build()
        assert result[0].content == "What's {{x}}?"

    def test_multiple_track_variables_calls(self):
        """Test calling track_variables multiple times."""
        builder = PromptBuilder()
        builder.messages.add_system("{{a}}")
        builder.track_variables()

        builder.messages.add_human("{{b}}")
        builder.track_variables()

        assert builder.variables == {"a", "b"}

    def test_build_multiple_times(self):
        """Test that build can be called multiple times."""
        builder = PromptBuilder()
        builder.messages.add_system("Test")

        result1 = builder.build()
        result2 = builder.build()

        assert len(result1) == len(result2)
        assert result1 is not result2  # Different instances
        assert result1[0].content == result2[0].content
