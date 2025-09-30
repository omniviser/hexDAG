"""Tests for Message and Messages classes.

Tests cover:
- MessageRole enum and from_string conversion
- Message dataclass creation and conversion (to_dict, from_dict)
- Messages collection operations (add, extend, clear)
- Fluent API and method chaining
- Filtering and history retrieval
- Edge cases and error handling
"""

import pytest

from hexai.core.application.prompt.messages import Message, MessageRole, Messages


class TestMessageRole:
    """Test MessageRole enum."""

    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.HUMAN.value == "human"
        assert MessageRole.AI.value == "ai"
        assert MessageRole.TOOL.value == "tool"
        assert MessageRole.FUNCTION_CALL.value == "function_call"

    def test_from_string_direct_values(self):
        """Test from_string with direct enum values."""
        assert MessageRole.from_string("system") == MessageRole.SYSTEM
        assert MessageRole.from_string("human") == MessageRole.HUMAN
        assert MessageRole.from_string("ai") == MessageRole.AI
        assert MessageRole.from_string("tool") == MessageRole.TOOL
        assert MessageRole.from_string("function_call") == MessageRole.FUNCTION_CALL

    def test_from_string_aliases(self):
        """Test from_string with common aliases."""
        assert MessageRole.from_string("user") == MessageRole.HUMAN
        assert MessageRole.from_string("assistant") == MessageRole.AI
        assert MessageRole.from_string("function") == MessageRole.FUNCTION_CALL

    def test_from_string_case_insensitive(self):
        """Test from_string is case insensitive."""
        assert MessageRole.from_string("SYSTEM") == MessageRole.SYSTEM
        assert MessageRole.from_string("Human") == MessageRole.HUMAN
        assert MessageRole.from_string("USER") == MessageRole.HUMAN
        assert MessageRole.from_string("Assistant") == MessageRole.AI

    def test_from_string_invalid(self):
        """Test from_string raises ValueError for invalid roles."""
        with pytest.raises(ValueError) as exc_info:
            MessageRole.from_string("invalid")
        assert "Unknown message role" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)


class TestMessage:
    """Test Message dataclass."""

    def test_create_basic_message(self):
        """Test creating a basic message."""
        msg = Message(MessageRole.HUMAN, "Hello")
        assert msg.role == MessageRole.HUMAN
        assert msg.content == "Hello"
        assert msg.metadata == {}
        assert msg.name is None

    def test_create_message_with_metadata(self):
        """Test creating message with metadata."""
        metadata = {"key": "value", "number": 42}
        msg = Message(MessageRole.AI, "Hi", metadata)
        assert msg.metadata == metadata

    def test_create_message_with_name(self):
        """Test creating message with name."""
        msg = Message(MessageRole.HUMAN, "Hello", name="Alice")
        assert msg.name == "Alice"

    def test_to_dict_basic(self):
        """Test converting message to dict."""
        msg = Message(MessageRole.HUMAN, "Hello")
        result = msg.to_dict()
        assert result == {"role": "human", "content": "Hello"}

    def test_to_dict_with_name(self):
        """Test to_dict includes name."""
        msg = Message(MessageRole.HUMAN, "Hello", name="Alice")
        result = msg.to_dict()
        assert result["name"] == "Alice"

    def test_to_dict_use_aliases(self):
        """Test to_dict with use_aliases=True."""
        msg_human = Message(MessageRole.HUMAN, "Hi")
        msg_ai = Message(MessageRole.AI, "Hello")

        assert msg_human.to_dict(use_aliases=True)["role"] == "user"
        assert msg_ai.to_dict(use_aliases=True)["role"] == "assistant"

    def test_to_dict_tool_message(self):
        """Test to_dict for tool message includes tool_call_id."""
        msg = Message(MessageRole.TOOL, "Result", {"tool_id": "123"})
        result = msg.to_dict()
        assert result["tool_call_id"] == "123"

    def test_to_dict_function_call(self):
        """Test to_dict for function call includes function_call."""
        metadata = {"function_name": "test_func", "arguments": {"arg": "value"}}
        msg = Message(MessageRole.FUNCTION_CALL, "", metadata)
        result = msg.to_dict()

        assert "function_call" in result
        assert result["function_call"]["name"] == "test_func"
        assert result["function_call"]["arguments"] == {"arg": "value"}

    def test_from_dict_basic(self):
        """Test creating message from dict."""
        data = {"role": "human", "content": "Hello"}
        msg = Message.from_dict(data)

        assert msg.role == MessageRole.HUMAN
        assert msg.content == "Hello"

    def test_from_dict_with_name(self):
        """Test from_dict with name field."""
        data = {"role": "human", "content": "Hello", "name": "Alice"}
        msg = Message.from_dict(data)
        assert msg.name == "Alice"

    def test_from_dict_tool_message(self):
        """Test from_dict extracts tool_call_id to metadata."""
        data = {"role": "tool", "content": "Result", "tool_call_id": "123"}
        msg = Message.from_dict(data)

        assert msg.role == MessageRole.TOOL
        assert msg.metadata["tool_id"] == "123"

    def test_from_dict_function_call(self):
        """Test from_dict extracts function_call to metadata."""
        data = {
            "role": "function_call",
            "content": "",
            "function_call": {"name": "test_func", "arguments": {"arg": "value"}},
        }
        msg = Message.from_dict(data)

        assert msg.role == MessageRole.FUNCTION_CALL
        assert msg.metadata["function_name"] == "test_func"
        assert msg.metadata["arguments"] == {"arg": "value"}

    def test_from_dict_with_aliases(self):
        """Test from_dict handles role aliases."""
        data = {"role": "user", "content": "Hi"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.HUMAN

        data = {"role": "assistant", "content": "Hello"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.AI

    def test_round_trip_conversion(self):
        """Test message can be converted to dict and back."""
        original = Message(MessageRole.HUMAN, "Test message", {"key": "value"}, "Alice")
        as_dict = original.to_dict()
        reconstructed = Message.from_dict(as_dict)

        assert reconstructed.role == original.role
        assert reconstructed.content == original.content
        assert reconstructed.name == original.name


class TestMessagesInit:
    """Test Messages initialization."""

    def test_init_empty(self):
        """Test creating empty Messages collection."""
        messages = Messages()
        assert len(messages) == 0
        assert not messages

    def test_init_with_messages(self):
        """Test creating Messages with initial messages."""
        msg_list = [
            Message(MessageRole.SYSTEM, "System"),
            Message(MessageRole.HUMAN, "Hello"),
        ]
        messages = Messages(msg_list)
        assert len(messages) == 2


class TestMessagesAdd:
    """Test adding messages to Messages collection."""

    def test_add_message(self):
        """Test adding a message directly."""
        messages = Messages()
        msg = Message(MessageRole.HUMAN, "Hello")
        result = messages.add(msg)

        assert len(messages) == 1
        assert messages[0] == msg
        assert result is messages  # Method chaining

    def test_add_system(self):
        """Test add_system convenience method."""
        messages = Messages()
        result = messages.add_system("System message")

        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == "System message"
        assert result is messages

    def test_add_system_with_metadata(self):
        """Test add_system with metadata."""
        messages = Messages()
        messages.add_system("System", key="value")

        assert messages[0].metadata == {"key": "value"}

    def test_add_human(self):
        """Test add_human convenience method."""
        messages = Messages()
        result = messages.add_human("User message")

        assert len(messages) == 1
        assert messages[0].role == MessageRole.HUMAN
        assert messages[0].content == "User message"
        assert result is messages

    def test_add_human_with_name(self):
        """Test add_human with name parameter."""
        messages = Messages()
        messages.add_human("Hello", name="Alice")

        assert messages[0].name == "Alice"

    def test_add_ai(self):
        """Test add_ai convenience method."""
        messages = Messages()
        result = messages.add_ai("AI response")

        assert len(messages) == 1
        assert messages[0].role == MessageRole.AI
        assert messages[0].content == "AI response"
        assert result is messages

    def test_add_ai_with_name(self):
        """Test add_ai with name parameter."""
        messages = Messages()
        messages.add_ai("Response", name="Assistant")

        assert messages[0].name == "Assistant"

    def test_add_tool(self):
        """Test add_tool convenience method."""
        messages = Messages()
        result = messages.add_tool("Tool result", tool_id="123")

        assert len(messages) == 1
        assert messages[0].role == MessageRole.TOOL
        assert messages[0].content == "Tool result"
        assert messages[0].metadata["tool_id"] == "123"
        assert result is messages

    def test_add_tool_with_name(self):
        """Test add_tool with tool_name parameter."""
        messages = Messages()
        messages.add_tool("Result", tool_id="123", tool_name="calculator")

        assert messages[0].metadata["tool_name"] == "calculator"

    def test_add_function_call(self):
        """Test add_function_call convenience method."""
        messages = Messages()
        args = {"x": 1, "y": 2}
        result = messages.add_function_call("add", args, "Adding numbers")

        assert len(messages) == 1
        assert messages[0].role == MessageRole.FUNCTION_CALL
        assert messages[0].content == "Adding numbers"
        assert messages[0].metadata["function_name"] == "add"
        assert messages[0].metadata["arguments"] == args
        assert result is messages

    def test_add_function_call_no_arguments(self):
        """Test add_function_call without arguments."""
        messages = Messages()
        messages.add_function_call("test_func")

        assert messages[0].metadata["arguments"] == {}


class TestMessagesFluentAPI:
    """Test fluent API and method chaining."""

    def test_chaining_multiple_adds(self):
        """Test chaining multiple add operations."""
        messages = Messages().add_system("Be helpful").add_human("Hello").add_ai("Hi there!")

        assert len(messages) == 3
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.HUMAN
        assert messages[2].role == MessageRole.AI

    def test_chaining_mixed_operations(self):
        """Test chaining mixed operations."""
        msg = Message(MessageRole.TOOL, "Result", {"tool_id": "123"})
        messages = Messages().add_system("System").add(msg).add_human("Query")

        assert len(messages) == 3


class TestMessagesExtend:
    """Test extending Messages collection."""

    def test_extend_with_list(self):
        """Test extend with list of messages."""
        messages = Messages()
        messages.add_system("System")

        new_msgs = [
            Message(MessageRole.HUMAN, "Hello"),
            Message(MessageRole.AI, "Hi"),
        ]
        result = messages.extend(new_msgs)

        assert len(messages) == 3
        assert result is messages

    def test_extend_with_messages(self):
        """Test extend with another Messages collection."""
        messages1 = Messages().add_system("System")
        messages2 = Messages().add_human("Hello").add_ai("Hi")

        messages1.extend(messages2)

        assert len(messages1) == 3
        assert len(messages2) == 2  # Original unchanged

    def test_extend_empty(self):
        """Test extend with empty collection."""
        messages = Messages().add_system("System")
        messages.extend([])

        assert len(messages) == 1


class TestMessagesQuery:
    """Test querying and filtering messages."""

    def test_filter_by_role(self):
        """Test filtering messages by role."""
        messages = (
            Messages()
            .add_system("System")
            .add_human("Q1")
            .add_ai("A1")
            .add_human("Q2")
            .add_ai("A2")
        )

        human_msgs = messages.filter_by_role(MessageRole.HUMAN)
        assert len(human_msgs) == 2
        assert all(msg.role == MessageRole.HUMAN for msg in human_msgs)

        ai_msgs = messages.filter_by_role(MessageRole.AI)
        assert len(ai_msgs) == 2

    def test_filter_by_role_no_matches(self):
        """Test filter_by_role with no matches."""
        messages = Messages().add_system("System")
        tool_msgs = messages.filter_by_role(MessageRole.TOOL)
        assert len(tool_msgs) == 0

    def test_get_history_all(self):
        """Test get_history without limit."""
        messages = Messages().add_system("S").add_human("H").add_ai("A")
        history = messages.get_history()

        assert len(history) == 3
        assert history is not messages._messages  # Returns copy

    def test_get_history_with_limit(self):
        """Test get_history with max_messages limit."""
        messages = (
            Messages().add_system("S1").add_human("H1").add_ai("A1").add_human("H2").add_ai("A2")
        )

        recent = messages.get_history(max_messages=2)
        assert len(recent) == 2
        assert recent[0].content == "H2"  # Most recent 2
        assert recent[1].content == "A2"

    def test_get_history_limit_exceeds_size(self):
        """Test get_history when limit exceeds collection size."""
        messages = Messages().add_system("S")
        history = messages.get_history(max_messages=10)

        assert len(history) == 1

    def test_get_history_zero_limit(self):
        """Test get_history with zero limit."""
        messages = Messages().add_system("S")
        history = messages.get_history(max_messages=0)

        assert len(history) == 0


class TestMessagesConversion:
    """Test conversion methods."""

    def test_to_list_basic(self):
        """Test converting messages to list of dicts."""
        messages = Messages().add_system("S").add_human("H")
        result = messages.to_list()

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "S"}
        assert result[1] == {"role": "human", "content": "H"}

    def test_to_list_with_aliases(self):
        """Test to_list with use_aliases=True."""
        messages = Messages().add_human("H").add_ai("A")
        result = messages.to_list(use_aliases=True)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_from_list(self):
        """Test creating Messages from list of dicts."""
        data = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        messages = Messages.from_list(data)

        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.HUMAN

    def test_round_trip_conversion(self):
        """Test Messages can be converted to list and back."""
        original = Messages().add_system("S").add_human("H").add_ai("A")
        as_list = original.to_list()
        reconstructed = Messages.from_list(as_list)

        assert len(reconstructed) == len(original)
        for i in range(len(original)):
            assert reconstructed[i].role == original[i].role
            assert reconstructed[i].content == original[i].content


class TestMessagesClear:
    """Test clearing messages."""

    def test_clear(self):
        """Test clearing all messages."""
        messages = Messages().add_system("S").add_human("H")
        result = messages.clear()

        assert len(messages) == 0
        assert not messages
        assert result is messages  # Method chaining

    def test_clear_empty(self):
        """Test clearing already empty collection."""
        messages = Messages()
        messages.clear()
        assert len(messages) == 0


class TestMessagesDunderMethods:
    """Test dunder methods."""

    def test_len(self):
        """Test __len__ method."""
        messages = Messages()
        assert len(messages) == 0

        messages.add_system("S")
        assert len(messages) == 1

        messages.add_human("H").add_ai("A")
        assert len(messages) == 3

    def test_bool(self):
        """Test __bool__ method."""
        messages = Messages()
        assert not messages

        messages.add_system("S")
        assert messages

    def test_iter(self):
        """Test __iter__ method."""
        messages = Messages().add_system("S").add_human("H")

        msg_list = list(messages)
        assert len(msg_list) == 2
        assert msg_list[0].role == MessageRole.SYSTEM

    def test_getitem(self):
        """Test __getitem__ method."""
        messages = Messages().add_system("S").add_human("H")

        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.HUMAN
        assert messages[-1].role == MessageRole.HUMAN

    def test_getitem_out_of_range(self):
        """Test __getitem__ raises IndexError."""
        messages = Messages()

        with pytest.raises(IndexError):
            _ = messages[0]

    def test_repr(self):
        """Test __repr__ method."""
        messages = Messages().add_system("S").add_human("H")
        repr_str = repr(messages)

        assert "Messages" in repr_str
        assert "count=2" in repr_str
        assert "system" in repr_str


class TestMessagesEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self):
        """Test messages with empty content."""
        messages = Messages().add_system("").add_human("")
        assert len(messages) == 2
        assert messages[0].content == ""

    def test_multiline_content(self):
        """Test messages with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        messages = Messages().add_human(content)
        assert messages[0].content == content

    def test_special_characters(self):
        """Test messages with special characters."""
        content = "Test {{var}} with $pecial ch@rs!"
        messages = Messages().add_human(content)
        assert messages[0].content == content

    def test_large_metadata(self):
        """Test message with large metadata."""
        metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        messages = Messages().add_system("Test", **metadata)
        assert len(messages[0].metadata) == 100

    def test_iterate_and_modify(self):
        """Test iterating while modifying is safe."""
        messages = Messages().add_system("S").add_human("H")

        # Iterate and collect (don't modify during iteration)
        roles = [msg.role for msg in messages]

        # Now modify
        messages.add_ai("A")

        assert len(roles) == 2
        assert len(messages) == 3

    def test_nested_metadata(self):
        """Test message with nested metadata structures."""
        metadata = {"level1": {"level2": {"level3": ["a", "b", "c"]}}}
        messages = Messages().add_system("Test", **metadata)
        assert messages[0].metadata["level1"]["level2"]["level3"] == ["a", "b", "c"]
