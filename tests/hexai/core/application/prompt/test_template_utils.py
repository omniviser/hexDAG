"""Tests for template utility functions."""

import pytest

from hexai.core.application.prompt.template_utils import (
    MissingVariableError,
    extract_variables,
    format_template_error,
    get_nested_value,
    merge_template_variables,
    render_template,
    validate_template_syntax,
)


class TestExtractVariables:
    """Tests for extract_variables function."""

    def test_extract_simple_variables(self):
        """Test extracting simple variables."""
        template = "Hello {{name}}, you are {{age}} years old"
        variables = extract_variables(template)
        assert variables == ["name", "age"]

    def test_extract_nested_variables(self):
        """Test extracting nested variables."""
        template = "User: {{user.name}}, Email: {{user.email}}, Company: {{company.name}}"
        variables = extract_variables(template)
        assert variables == ["user", "company"]

    def test_extract_no_variables(self):
        """Test template with no variables."""
        template = "Hello world!"
        variables = extract_variables(template)
        assert variables == []

    def test_extract_duplicate_variables(self):
        """Test deduplication of variables."""
        template = "{{name}} is {{name}} and {{age}} years old"
        variables = extract_variables(template)
        assert variables == ["name", "age"]

    def test_extract_deeply_nested_variables(self):
        """Test extracting deeply nested variables."""
        template = "{{user.profile.settings.theme}} and {{user.name}}"
        variables = extract_variables(template)
        assert variables == ["user"]


class TestGetNestedValue:
    """Tests for get_nested_value function."""

    def test_get_simple_value(self):
        """Test getting simple value from dict."""
        data = {"name": "Alice", "age": 30}
        assert get_nested_value(data, "name") == "Alice"
        assert get_nested_value(data, "age") == 30

    def test_get_nested_dict_value(self):
        """Test getting nested value from dict."""
        data = {"user": {"name": "Bob", "profile": {"age": 25}}}
        assert get_nested_value(data, "user.name") == "Bob"
        assert get_nested_value(data, "user.profile.age") == 25

    def test_get_object_attribute(self):
        """Test getting attribute from object."""

        class User:
            def __init__(self):
                self.name = "Charlie"
                self.age = 35

        user = User()
        data = {"user": user}
        assert get_nested_value(data, "user.name") == "Charlie"
        assert get_nested_value(data, "user.age") == 35

    def test_missing_key_error(self):
        """Test KeyError for missing key."""
        data = {"name": "Alice"}
        with pytest.raises(KeyError):
            get_nested_value(data, "age")

    def test_missing_nested_key_error(self):
        """Test KeyError for missing nested key."""
        data = {"user": {"name": "Bob"}}
        with pytest.raises(KeyError):
            get_nested_value(data, "user.age")


class TestRenderTemplate:
    """Tests for render_template function."""

    def test_render_simple_template(self):
        """Test rendering simple template."""
        template = "Hello {{name}}!"
        variables = {"name": "Alice"}
        result = render_template(template, variables)
        assert result == "Hello Alice!"

    def test_render_nested_template(self):
        """Test rendering template with nested variables."""
        template = "User: {{user.name}}, Age: {{user.age}}"
        variables = {"user": {"name": "Bob", "age": 25}}
        result = render_template(template, variables)
        assert result == "User: Bob, Age: 25"

    def test_render_with_validation_missing_var(self):
        """Test rendering with validation when variable is missing."""
        template = "Hello {{name}}!"
        variables = {}
        with pytest.raises(MissingVariableError) as exc_info:
            render_template(template, variables, validate_vars=True)
        assert "name" in str(exc_info.value)

    def test_render_without_validation(self):
        """Test rendering without validation keeps placeholders."""
        template = "Hello {{name}}!"
        variables = {}
        # This will raise because we changed the behavior to always raise on missing nested access
        with pytest.raises(MissingVariableError):
            render_template(template, variables, validate_vars=False)

    def test_render_type_conversion(self):
        """Test automatic type conversion to string."""
        template = "Count: {{count}}, Pi: {{pi}}"
        variables = {"count": 42, "pi": 3.14159}
        result = render_template(template, variables)
        assert result == "Count: 42, Pi: 3.14159"


class TestValidateTemplateSyntax:
    """Tests for validate_template_syntax function."""

    def test_valid_template(self):
        """Test valid template syntax."""
        is_valid, error = validate_template_syntax("Hello {{name}}!")
        assert is_valid is True
        assert error is None

    def test_unbalanced_braces(self):
        """Test unbalanced braces detection."""
        is_valid, error = validate_template_syntax("Hello {{name}!")
        assert is_valid is False
        assert "Unbalanced braces" in error

    def test_empty_variable(self):
        """Test empty variable name detection."""
        is_valid, error = validate_template_syntax("Hello {{}}!")
        assert is_valid is False
        assert "Empty variable" in error

    def test_nested_braces(self):
        """Test nested braces detection."""
        is_valid, error = validate_template_syntax("Hello {{na{me}}!")
        assert is_valid is False
        assert "Nested braces" in error

    def test_invalid_variable_name(self):
        """Test invalid variable name detection."""
        is_valid, error = validate_template_syntax("Hello {{123name}}!")
        assert is_valid is False
        assert "Invalid variable name" in error

    def test_valid_nested_variable(self):
        """Test valid nested variable syntax."""
        is_valid, error = validate_template_syntax("Hello {{user.name.first}}!")
        assert is_valid is True
        assert error is None


class TestMergeTemplateVariables:
    """Tests for merge_template_variables function."""

    def test_merge_simple_dicts(self):
        """Test merging simple dictionaries."""
        base = {"name": "Alice", "age": 30}
        override = {"age": 31, "city": "NYC"}
        result = merge_template_variables(base, override)
        assert result == {"name": "Alice", "age": 31, "city": "NYC"}

    def test_merge_empty_base(self):
        """Test merging with empty base."""
        base = {}
        override = {"name": "Bob"}
        result = merge_template_variables(base, override)
        assert result == {"name": "Bob"}

    def test_merge_empty_override(self):
        """Test merging with empty override."""
        base = {"name": "Alice"}
        override = {}
        result = merge_template_variables(base, override)
        assert result == {"name": "Alice"}


class TestFormatTemplateError:
    """Tests for format_template_error function."""

    def test_format_simple_error(self):
        """Test formatting simple error message."""
        template = "Hello {{name}}!"
        error = Exception("Variable not found")
        formatted = format_template_error(template, error)
        assert "Template Error: Variable not found" in formatted
        assert "Hello {{name}}!" in formatted

    def test_format_multiline_template_error(self):
        """Test formatting error for multiline template."""
        template = "Line 1\nLine 2 {{invalid}}\nLine 3"
        error = MissingVariableError("Variable 'invalid' not found")
        formatted = format_template_error(template, error, context_lines=1)
        assert "Template Error" in formatted
        # Should show some context
        assert "Line" in formatted
