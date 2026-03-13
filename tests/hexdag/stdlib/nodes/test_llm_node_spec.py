"""Tests for the new LLMNode unified YAML spec.

Tests cover:
- New fields: human_message, system_message, examples, conversation, output_schema
- Backward compatibility: prompt_template, system_prompt, parse_json, template
- Deprecation warnings
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

from hexdag.stdlib.nodes.llm_node import LLMNode


class TestNewSpec:
    """Test the new unified YAML spec fields."""

    def test_human_message_creates_node(self) -> None:
        llm = LLMNode()
        spec = llm(name="test", human_message="Hello {{name}}")
        assert spec.name == "test"

    def test_system_message(self) -> None:
        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Analyze {{text}}",
            system_message="You are an analyst.",
        )
        assert spec.name == "test"

    def test_output_schema_implies_structured(self) -> None:
        """Presence of output_schema should enable structured output."""
        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Analyze {{text}}",
            output_schema={"sentiment": str, "score": float},
        )
        assert spec.name == "test"

    def test_examples_accepted(self) -> None:
        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Classify: {{text}}",
            system_message="You are a classifier.",
            examples=[
                {"input": "Great!", "output": "positive"},
                {"input": "Terrible.", "output": "negative"},
            ],
        )
        assert spec.name == "test"

    def test_conversation_string_accepted(self) -> None:
        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Answer: {{question}}",
            conversation="{{history_node}}",
        )
        assert spec.name == "test"

    def test_conversation_list_accepted(self) -> None:
        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Answer: {{question}}",
            conversation=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert spec.name == "test"

    def test_missing_human_message_raises(self) -> None:
        llm = LLMNode()
        with pytest.raises(ValueError, match="human_message"):
            llm(name="test")

    def test_output_schema_with_pydantic_model(self) -> None:
        class MyOutput(BaseModel):
            result: str

        llm = LLMNode()
        spec = llm(
            name="test",
            human_message="Do it",
            output_schema=MyOutput,
        )
        assert spec.name == "test"


class TestBackwardCompat:
    """Test backward compatibility with deprecated fields."""

    def test_prompt_template_string_works(self) -> None:
        llm = LLMNode()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = llm(name="test", prompt_template="Hello {{name}}")
            assert spec.name == "test"
            assert any("prompt_template" in str(warning.message) for warning in w)

    def test_system_prompt_works(self) -> None:
        llm = LLMNode()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = llm(
                name="test",
                human_message="Hi",
                system_prompt="Be helpful",
            )
            assert spec.name == "test"
            assert any("system_prompt" in str(warning.message) for warning in w)

    def test_template_alias_works(self) -> None:
        llm = LLMNode()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = llm(name="test", template="Hello {{name}}")
            assert spec.name == "test"
            assert any("template" in str(warning.message) for warning in w)

    def test_parse_json_warns(self) -> None:
        llm = LLMNode()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = llm(
                name="test",
                human_message="Analyze {{text}}",
                parse_json=True,
                output_schema={"result": str},
            )
            assert spec.name == "test"
            assert any("parse_json" in str(warning.message) for warning in w)

    def test_human_message_takes_precedence_over_prompt_template(self) -> None:
        """human_message should be used when both are provided."""
        llm = LLMNode()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec = llm(
                name="test",
                human_message="New way {{x}}",
                prompt_template="Old way {{y}}",
            )
            assert spec.name == "test"

    def test_from_template_legacy_method(self) -> None:
        """Legacy from_template class method should still work."""
        spec = LLMNode.from_template(
            name="test",
            template="Hello {{name}}",
        )
        assert spec.name == "test"
