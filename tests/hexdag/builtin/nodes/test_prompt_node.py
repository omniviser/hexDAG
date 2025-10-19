"""Tests for PromptNode - composable prompt building."""

import pytest
from pydantic import BaseModel

from hexdag.builtin.nodes.prompt_node import PromptNode
from hexdag.builtin.prompts.base import ChatPromptTemplate, PromptTemplate


class TestPromptNode:
    """Test PromptNode functionality."""

    def test_simple_string_template(self):
        """Test prompt building with simple string template."""
        prompt_node = PromptNode()
        spec = prompt_node(name="greeter", template="Hello {{name}}!", output_format="string")

        assert spec.name == "greeter"
        assert spec.fn is not None

    @pytest.mark.asyncio
    async def test_render_simple_prompt(self):
        """Test rendering a simple prompt."""
        prompt_node = PromptNode()
        spec = prompt_node(name="greeter", template="Hello {{name}}!", output_format="string")

        result = await spec.fn({"name": "Alice"})

        assert result["text"] == "Hello Alice!"

    @pytest.mark.asyncio
    async def test_render_prompt_as_messages(self):
        """Test rendering prompt in message format."""
        prompt_node = PromptNode()
        spec = prompt_node(name="greeter", template="Hello {{name}}!", output_format="messages")

        result = await spec.fn({"name": "Bob"})

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello Bob!"

    @pytest.mark.asyncio
    async def test_chat_template(self):
        """Test chat-style template with system and user messages."""
        template = ChatPromptTemplate(
            system_message="You are a helpful assistant named {{bot_name}}",
            human_message="Answer this question: {{question}}",
        )

        prompt_node = PromptNode()
        spec = prompt_node(name="qa_bot", template=template, output_format="messages")

        result = await spec.fn({"bot_name": "Claude", "question": "What is 2+2?"})

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "Claude" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert "What is 2+2?" in result["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_composed_template(self):
        """Test builder pattern composition."""
        base = PromptTemplate("Analyze {{data}}")
        tools = PromptTemplate("\n\nAvailable tools: {{tools}}")
        composed = base + tools

        prompt_node = PromptNode()
        spec = prompt_node(name="analyzer", template=composed, output_format="string")

        result = await spec.fn({"data": "sales report", "tools": "search, calculator"})

        assert "Analyze sales report" in result["text"]
        assert "Available tools: search, calculator" in result["text"]

    @pytest.mark.asyncio
    async def test_with_system_prompt(self):
        """Test adding system prompt to simple template."""
        prompt_node = PromptNode()
        spec = prompt_node(
            name="helper",
            template="User question: {{question}}",
            output_format="messages",
            system_prompt="You are an expert assistant",
        )

        result = await spec.fn({"question": "Help me"})

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "expert assistant" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"
        assert "Help me" in result["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_with_pydantic_input(self):
        """Test prompt rendering with Pydantic input model."""

        class UserInput(BaseModel):
            name: str
            age: int

        prompt_node = PromptNode()
        spec = prompt_node(
            name="greeter",
            template="Hello {{name}}, you are {{age}} years old!",
            output_format="string",
        )

        input_data = UserInput(name="Alice", age=30)
        result = await spec.fn(input_data)

        assert result["text"] == "Hello Alice, you are 30 years old!"
