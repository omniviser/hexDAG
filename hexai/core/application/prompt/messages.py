"""Message classes for structured conversation handling in prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(Enum):
    """Enumeration of possible message roles in a conversation."""

    SYSTEM = "system"
    HUMAN = "human"  # User messages
    AI = "ai"  # Assistant/AI responses
    TOOL = "tool"  # Tool execution results
    FUNCTION_CALL = "function_call"  # Function invocation requests

    @classmethod
    def from_string(cls, role: str) -> MessageRole:
        """Create MessageRole from string, with aliases support.

        Args:
            role: Role string (e.g., "user", "human", "assistant", "ai")

        Returns:
            Corresponding MessageRole enum value

        Raises:
            ValueError: If role string is not recognized
        """
        # Handle common aliases
        aliases = {
            "user": cls.HUMAN,
            "assistant": cls.AI,
            "human": cls.HUMAN,
            "ai": cls.AI,
            "system": cls.SYSTEM,
            "tool": cls.TOOL,
            "function_call": cls.FUNCTION_CALL,
            "function": cls.FUNCTION_CALL,
        }

        role_lower = role.lower()
        if role_lower in aliases:
            return aliases[role_lower]

        # Try direct enum value
        try:
            return cls(role_lower)
        except ValueError:
            raise ValueError(
                f"Unknown message role: '{role}'. Valid roles: {', '.join(aliases.keys())}"
            ) from None


@dataclass
class Message:
    """Single message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The message content (can be template with variables)
        metadata: Optional metadata (tool_id, function_name, etc.)
        name: Optional name identifier for the message
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    def to_dict(self, use_aliases: bool = False) -> dict[str, Any]:
        """Convert message to dictionary format.

        Args:
            use_aliases: If True, use common aliases (user/assistant instead of human/ai)

        Returns:
            Dictionary representation of the message
        """
        role_value = self.role.value

        # Use common aliases if requested
        if use_aliases:
            if self.role == MessageRole.HUMAN:
                role_value = "user"
            elif self.role == MessageRole.AI:
                role_value = "assistant"

        result: dict[str, Any] = {"role": role_value, "content": self.content}

        # Add optional fields if present
        if self.name:
            result["name"] = self.name

        # Add relevant metadata fields
        if self.metadata:
            # For tool messages, include tool_id
            if self.role == MessageRole.TOOL and "tool_id" in self.metadata:
                result["tool_call_id"] = self.metadata["tool_id"]
            # For function calls, include function details
            elif self.role == MessageRole.FUNCTION_CALL and "function_name" in self.metadata:
                result["function_call"] = {
                    "name": self.metadata["function_name"],
                    "arguments": self.metadata.get("arguments", {}),
                }

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create Message from dictionary.

        Args:
            data: Dictionary with role, content, and optional metadata

        Returns:
            Message instance
        """
        role = MessageRole.from_string(data["role"])
        content = data["content"]
        metadata = {}
        name = data.get("name")

        # Extract metadata from special fields
        if "tool_call_id" in data:
            metadata["tool_id"] = data["tool_call_id"]

        if "function_call" in data:
            fc = data["function_call"]
            metadata["function_name"] = fc.get("name", "")
            metadata["arguments"] = fc.get("arguments", {})

        # Include any additional metadata
        for key in data:
            if key not in ["role", "content", "name", "tool_call_id", "function_call"]:
                metadata[key] = data[key]

        return cls(role=role, content=content, metadata=metadata or {}, name=name)


class Messages:
    """Collection of messages forming a conversation.

    Provides a fluent API for building conversations and managing message history.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        """Initialize Messages collection.

        Args:
            messages: Optional initial list of messages
        """
        self._messages: list[Message] = messages or []

    def add(self, message: Message) -> Messages:
        """Add a message to the collection.

        Args:
            message: Message to add

        Returns:
            Self for method chaining
        """
        self._messages.append(message)
        return self

    def add_system(self, content: str, **metadata: Any) -> Messages:
        """Add a system message.

        Args:
            content: System message content
            **metadata: Optional metadata for the message

        Returns:
            Self for method chaining
        """
        return self.add(Message(MessageRole.SYSTEM, content, metadata))

    def add_human(self, content: str, name: str | None = None, **metadata: Any) -> Messages:
        """Add a human/user message.

        Args:
            content: Human message content
            name: Optional name for the human
            **metadata: Optional metadata for the message

        Returns:
            Self for method chaining
        """
        return self.add(Message(MessageRole.HUMAN, content, metadata, name))

    def add_ai(self, content: str, name: str | None = None, **metadata: Any) -> Messages:
        """Add an AI/assistant message.

        Args:
            content: AI message content
            name: Optional name for the AI
            **metadata: Optional metadata for the message

        Returns:
            Self for method chaining
        """
        return self.add(Message(MessageRole.AI, content, metadata, name))

    def add_tool(self, content: str, tool_id: str, tool_name: str | None = None) -> Messages:
        """Add a tool result message.

        Args:
            content: Tool execution result
            tool_id: Unique identifier for the tool call
            tool_name: Optional name of the tool

        Returns:
            Self for method chaining
        """
        metadata = {"tool_id": tool_id}
        if tool_name:
            metadata["tool_name"] = tool_name

        return self.add(Message(MessageRole.TOOL, content, metadata))

    def add_function_call(
        self, function_name: str, arguments: dict[str, Any] | None = None, content: str = ""
    ) -> Messages:
        """Add a function call message.

        Args:
            function_name: Name of the function to call
            arguments: Function arguments
            content: Optional content/description

        Returns:
            Self for method chaining
        """
        metadata = {"function_name": function_name, "arguments": arguments or {}}

        return self.add(Message(MessageRole.FUNCTION_CALL, content, metadata))

    def extend(self, messages: list[Message] | Messages) -> Messages:
        """Extend with multiple messages.

        Args:
            messages: List of messages or another Messages collection

        Returns:
            Self for method chaining
        """
        if isinstance(messages, Messages):
            self._messages.extend(messages._messages)
        else:
            self._messages.extend(messages)
        return self

    def filter_by_role(self, role: MessageRole) -> list[Message]:
        """Get all messages with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of messages with the specified role
        """
        return [msg for msg in self._messages if msg.role == role]

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """Get conversation history.

        Args:
            max_messages: Maximum number of messages to return (from most recent)

        Returns:
            List of messages, optionally limited to max_messages
        """
        if max_messages is None:
            return self._messages.copy()

        return self._messages[-max_messages:] if max_messages > 0 else []

    def to_list(self, use_aliases: bool = False) -> list[dict[str, Any]]:
        """Convert all messages to list of dictionaries.

        Args:
            use_aliases: If True, use common aliases (user/assistant instead of human/ai)

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict(use_aliases=use_aliases) for msg in self._messages]

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> Messages:
        """Create Messages collection from list of dictionaries.

        Args:
            messages: List of message dictionaries

        Returns:
            Messages collection
        """
        return cls([Message.from_dict(msg) for msg in messages])

    def clear(self) -> Messages:
        """Clear all messages.

        Returns:
            Self for method chaining
        """
        self._messages.clear()
        return self

    def __len__(self) -> int:
        """Get number of messages."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Check if collection has messages."""
        return bool(self._messages)

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Iterate over messages."""
        return iter(self._messages)

    def __getitem__(self, index: int) -> Message:
        """Get message by index."""
        return self._messages[index]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Messages(count={len(self._messages)}, \
            roles={[m.role.value for m in self._messages[:3]]}...)"
