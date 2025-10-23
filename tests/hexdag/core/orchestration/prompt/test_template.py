"""Tests for PromptTemplate class.

Tests cover:
- Basic template rendering
- Variable extraction (manual and automatic)
- Missing variable error handling
- Deep nested variable replacement
- Complex template scenarios
- Edge cases and error conditions
"""

import pytest

from hexdag.builtin.prompts.base import (
    FewShotPromptTemplate,
    PromptTemplate,
)
from hexdag.core.orchestration.prompt.template import (
    MissingVariableError,
)

# No initialization needed - framework is ready to use


class TestPromptTemplate:
    """Test cases for PromptTemplate class."""

    def test_basic_template_rendering(self):
        """Test basic template creation and rendering."""
        template = PromptTemplate("Hello {{name}}!")
        result = template.render(name="Alice")
        assert result == "Hello Alice!"

    def test_auto_variable_extraction(self):
        """Test automatic variable extraction from template."""
        template = PromptTemplate("Hello {{name}}, you are {{age}} years old!")
        assert "name" in template.input_vars
        assert "age" in template.input_vars
        assert len(template.input_vars) == 2

    def test_manual_variable_specification(self):
        """Test manually specifying input variables."""
        template = PromptTemplate("Hello {{name}}!", ["name", "optional_var"])

        assert template.input_vars == ["name", "optional_var"]

        with pytest.raises(MissingVariableError) as exc_info:
            template.render(name="Alice")

        assert "optional_var" in str(exc_info.value)

    def test_deep_nested_variable_replacement(self):
        """Test deep nested object access in templates."""
        template = PromptTemplate(
            "User: {{user.name}} ({{user.profile.role}})\n"
            "Company: {{user.company.name}}\n"
            "Contact: {{user.contact.email}}"
        )

        assert template.input_vars == ["user"]

        user_data = {
            "name": "Alice Johnson",
            "profile": {"role": "Senior Developer"},
            "company": {"name": "Tech Corp"},
            "contact": {"email": "alice@techcorp.com"},
        }

        result = template.render(user=user_data)
        expected = (
            "User: Alice Johnson (Senior Developer)\n"
            "Company: Tech Corp\n"
            "Contact: alice@techcorp.com"
        )
        assert result == expected

    def test_missing_variable_error(self):
        """Test error handling for missing variables."""
        template = PromptTemplate("Hello {{name}}!")
        with pytest.raises(MissingVariableError):
            template.render()

    def test_complex_template_with_conditionals(self):
        """Test template with nested variable access (simplified from Jinja2 version)."""
        # Simplified template without Jinja2 control structures
        template_str = "User: {{user.name}}\nRole: {{user.role}}\nDepartment: {{user.department}}"
        template = PromptTemplate(template_str)

        assert template.input_vars == ["user"]

        user_data = {
            "name": "Alice Smith",
            "role": "Admin",
            "department": "Engineering",
        }

        result = template.render(user=user_data)
        assert result == "User: Alice Smith\nRole: Admin\nDepartment: Engineering"

    def test_whitespace_handling(self):
        """Test template with various whitespace scenarios."""
        template = PromptTemplate("{{ name }}\n{{ role }}\n{{ department }}")

        assert set(template.input_vars) == {"name", "role", "department"}

        result = template.render(name="Eve", role="Engineer", department="AI")
        assert result == "Eve\nEngineer\nAI"

    def test_special_characters_in_variables(self):
        """Test templates with special characters and escaping."""
        template = PromptTemplate("Message: {{message}}\nQuote: '{{quote}}'")

        result = template.render(message='Hello "World"!', quote="She said 'Hello'")
        assert result == "Message: Hello \"World\"!\nQuote: 'She said 'Hello''"

    def test_empty_template(self):
        """Test edge case with empty template."""
        template = PromptTemplate("")

        assert template.input_vars == []
        assert template.render() == ""

    def test_template_without_variables(self):
        """Test template with no variables."""
        template = PromptTemplate("This is a static template.")

        assert template.input_vars == []
        result = template.render()
        assert result == "This is a static template."

    def test_variable_extraction_edge_cases(self):
        """Test variable extraction with edge cases."""
        template = PromptTemplate("{{var_1}} {{var_2}} {{user_name_2}}")
        assert set(template.input_vars) == {"var_1", "var_2", "user_name_2"}

        template = PromptTemplate("{{a.b.c}} {{x.y}} {{simple}}")
        assert set(template.input_vars) == {"a", "x", "simple"}

        template = PromptTemplate("{{name}} and {{name}} again")
        assert template.input_vars == ["name"]

    def test_string_representations(self):
        """Test string representation methods."""
        template = PromptTemplate("Hello {{name}}!")

        str_repr = str(template)
        assert "PromptTemplate" in str_repr
        assert "name" in str_repr

        repr_str = repr(template)
        assert "PromptTemplate" in repr_str
        assert "Hello" in repr_str
        assert "name" in repr_str

    def test_simple_undefined_error_handling(self):
        """Test handling of missing nested variables."""
        template = PromptTemplate("{{user.profile.name}}", ["user"])

        with pytest.raises(MissingVariableError) as exc_info:
            template.render(user={"name": "Alice"})

        error_msg = str(exc_info.value)
        assert "Cannot access variable" in error_msg
        assert "user.profile.name" in error_msg

    def test_multiple_missing_variables_detailed_error(self):
        """Test detailed error message for multiple missing variables."""
        template = PromptTemplate("{{a}} {{b}} {{c}}", ["a", "b", "c"])

        with pytest.raises(MissingVariableError) as exc_info:
            template.render(a="value_a")

        error_msg = str(exc_info.value)
        assert "Missing required template variables: ['b', 'c']" in error_msg
        assert "Required variables: ['a', 'b', 'c']" in error_msg
        assert "Provided variables: ['a']" in error_msg

    def test_extra_variables_allowed(self):
        """Test that providing extra variables doesn't cause errors."""
        template = PromptTemplate("Hello {{name}}!")

        result = template.render(name="Alice", extra_var="ignored", another="also_ignored")
        assert result == "Hello Alice!"

    def test_simple_nested_data_structures(self):
        """Test with nested data structures (simplified from Jinja2 version)."""
        template_str = (
            "Project: {{project.name}}\n"
            "Team Lead: {{project.team.lead.name}}\n"
            "Status: {{project.status.current}}"
        )
        template = PromptTemplate(template_str)

        project_data = {
            "name": "AI Platform",
            "team": {
                "lead": {"name": "Alice"},
                "members": [{"name": "Bob"}, {"name": "Charlie"}, {"name": "Diana"}],
            },
            "status": {"current": "In Progress"},
        }

        result = template.render(project=project_data)
        expected = "Project: AI Platform\nTeam Lead: Alice\nStatus: In Progress"
        assert result == expected


class TestFewShotPromptTemplate:
    """Test cases for FewShotPromptTemplate class."""

    def test_basic_few_shot_creation(self):
        """Test basic few-shot template creation."""
        examples = [
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+3?", "output": "6"},
        ]
        template = FewShotPromptTemplate("Calculate: {{question}}", examples=examples)

        # Check that examples are included in the template
        assert "Input: What is 2+2?" in template.template
        assert "Output: 4" in template.template
        assert "Calculate: {{question}}" in template.template

    def test_few_shot_rendering(self):
        """Test rendering with few-shot examples."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "Goodbye", "output": "See you later!"},
        ]
        template = FewShotPromptTemplate("Respond to: {{greeting}}", examples=examples)

        result = template.render(greeting="Good morning")

        # Should contain both examples and the rendered question
        assert "Input: Hello" in result
        assert "Output: Hi there!" in result
        assert "Input: Goodbye" in result
        assert "Output: See you later!" in result
        assert "Respond to: Good morning" in result

    def test_custom_example_separator(self):
        """Test custom example separator."""
        examples = [
            {"input": "Test1", "output": "Result1"},
            {"input": "Test2", "output": "Result2"},
        ]
        template = FewShotPromptTemplate(
            "Task: {{task}}", examples=examples, example_separator="\n---\n"
        )

        result = template.render(task="New task")
        assert "\n---\n" in result

    def test_custom_example_formatter(self):
        """Test custom example formatting function."""

        def custom_formatter(example):
            return f"Q: {example['input']}\nA: {example['output']}"

        examples = [{"input": "What's your name?", "output": "I'm an AI"}]
        template = FewShotPromptTemplate(
            "Question: {{question}}", examples=examples, format_example=custom_formatter
        )

        result = template.render(question="How are you?")
        assert "Q: What's your name?" in result
        assert "A: I'm an AI" in result

    def test_empty_examples(self):
        """Test few-shot template with no examples."""
        template = FewShotPromptTemplate("Task: {{task}}", examples=[])
        result = template.render(task="Do something")
        assert result == "Task: Do something"

    def test_add_example_method(self):
        """Test adding examples dynamically."""
        template = FewShotPromptTemplate("Task: {{task}}", examples=[])

        # Add an example
        template.add_example({"input": "Test", "output": "Success"})

        result = template.render(task="New task")
        assert "Input: Test" in result
        assert "Output: Success" in result
        assert "Task: New task" in result

    def test_variable_extraction_with_examples(self):
        """Test that variable extraction works with examples."""
        examples = [{"input": "Hello {{name}}", "output": "Hi {{name}}!"}]
        template = FewShotPromptTemplate("Greet: {{person}}", examples=examples)

        # Should extract 'person' from main template
        # Note: variables inside examples are not extracted
        assert "person" in template.input_vars

    def test_examples_with_missing_keys(self):
        """Test handling of examples with missing input/output keys."""
        examples = [
            {"input": "Test", "output": "Result"},
            {"question": "What?", "answer": "This!"},  # Different keys
        ]
        template = FewShotPromptTemplate("Task: {{task}}", examples=examples)

        result = template.render(task="Do it")
        # Should handle missing keys gracefully (empty strings)
        assert "Input: Test" in result
        assert "Output: Result" in result
        assert "Input: " in result  # Empty input from second example
        assert "Output: " in result  # Empty output from second example

    def test_tool_usage_examples(self):
        """Test few-shot examples with tool usage patterns."""
        examples = [
            {
                "input": "Search for Python tutorials",
                "output": (
                    'INVOKE_TOOL: search_tool {"query": "Python tutorials", "max_results": 5}'
                ),
            },
            {
                "input": "Analyze sentiment of 'I love this!'",
                "output": (
                    "INVOKE_TOOL: analyze_tool "
                    '{"content": "I love this!", "analysis_type": "sentiment"}'
                ),
            },
        ]

        template = FewShotPromptTemplate("Task: {{goal}}\nUse tools as needed.", examples=examples)

        result = template.render(goal="Find information about AI")

        # Should contain tool usage examples
        assert "INVOKE_TOOL: search_tool" in result
        assert "INVOKE_TOOL: analyze_tool" in result
        assert "Task: Find information about AI" in result
