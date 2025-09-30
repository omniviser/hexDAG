"""Utility functions for template processing and variable handling."""

import re
from typing import Any


class TemplateError(Exception):
    """Base exception for template utilities."""


class MissingVariableError(TemplateError):
    """Raised when required template variables are missing."""


def extract_variables(template: str) -> list[str]:
    """Extract variable names from template string.

    Finds all {{variable}} patterns and extracts the root variable names.
    Handles nested variables like {{user.name}} by extracting 'user'.

    Args
    ----
        template: Template string with {{variable}} placeholders

    Returns
    -------
        List of unique root variable names found in template

    Examples
    --------
        >>> extract_variables("Hello {{name}}, your score is {{score}}")
        ['name', 'score']

        >>> extract_variables("User: {{user.name}}, Email: {{user.email}}")
        ['user']
    """
    # Pattern matches {{variable}} or {{object.property}}
    pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"
    matches = re.findall(pattern, template)

    # Extract root variable names (before any dots) and deduplicate
    root_vars = []
    for match in matches:
        root_var = match.split(".")[0]
        if root_var not in root_vars:
            root_vars.append(root_var)

    return root_vars


def get_nested_value(data: dict[str, Any], key: str) -> Any:
    """Get value from nested dictionary using dot notation.

    Supports both dictionary access and object attribute access.

    Args
    ----
        data: Dictionary or object to extract value from
        key: Key path (e.g., "user.name" -> data["user"]["name"])

    Returns
    -------
        The value at the specified path

    Raises
    ------
        KeyError: If dictionary key doesn't exist

    Examples
    --------
        >>> data = {"user": {"name": "Alice", "age": 30}}
        >>> get_nested_value(data, "user.name")
        'Alice'

        >>> get_nested_value(data, "user.age")
        30
    """
    parts = key.split(".")
    value = data
    for part in parts:
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    return value


def render_template(template: str, variables: dict[str, Any], validate_vars: bool = True) -> str:
    """Render a template string with provided variables.

    Replaces all {{variable}} placeholders with their corresponding values.
    Supports nested variable access like {{user.name}}.

    Args
    ----
        template: Template string with {{variable}} placeholders
        variables: Dictionary of variable values
        validate_vars: If True, check that all variables are provided

    Returns
    -------
        Rendered template string

    Raises
    ------
        MissingVariableError: If required variables are missing and validate_vars is True

    Examples
    --------
        >>> render_template("Hello {{name}}!", {"name": "Alice"})
        'Hello Alice!'

        >>> template = "User: {{user.name}}, Age: {{user.age}}"
        >>> variables = {"user": {"name": "Bob", "age": 25}}
        >>> render_template(template, variables)
        'User: Bob, Age: 25'
    """
    if validate_vars:
        required_vars = extract_variables(template)
        missing_vars = [var for var in required_vars if var not in variables]

        if missing_vars:
            raise MissingVariableError(
                f"Missing required template variables: {missing_vars}. "
                f"Required variables: {required_vars}. "
                f"Provided variables: {list(variables.keys())}"
            )

    # Pattern for variable substitution
    pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}"

    def replace_var(match: re.Match[str]) -> str:
        """Replace a single variable match."""
        var_path = match.group(1).strip()
        try:
            value = get_nested_value(variables, var_path)
            return str(value)
        except (KeyError, AttributeError) as e:
            # Always raise if we can't access a variable that exists in the template
            raise MissingVariableError(f"Cannot access variable '{var_path}': {e}") from e

    return re.sub(pattern, replace_var, template)


def validate_template_syntax(template: str) -> tuple[bool, str | None]:
    """Validate template syntax for common errors.

    Checks for:
    - Balanced braces
    - Valid variable names
    - No nested templates

    Args
    ----
        template: Template string to validate

    Returns
    -------
        Tuple of (is_valid, error_message)
        - is_valid: True if template syntax is valid
        - error_message: Description of error if invalid, None if valid

    Examples
    --------
        >>> validate_template_syntax("Hello {{name}}!")
        (True, None)

        >>> validate_template_syntax("Hello {{}}!")
        (False, 'Empty variable name in template')

        >>> validate_template_syntax("Hello {{na{me}}!")
        (False, 'Nested braces not allowed in template variables')
    """
    # Check for balanced braces
    if template.count("{{") != template.count("}}"):
        return False, "Unbalanced braces in template"

    # Find all potential variable patterns
    pattern = r"\{\{([^}]*)\}\}"
    matches = re.findall(pattern, template)

    for match in matches:
        # Check for empty variable
        stripped = match.strip()
        if not stripped:
            return False, "Empty variable name in template"

        # Check for nested braces
        if "{" in stripped or "}" in stripped:
            return False, "Nested braces not allowed in template variables"

        # Check for valid variable name (allows dots for nested access)
        var_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
        if not re.match(var_pattern, stripped):
            return False, f"Invalid variable name: '{stripped}'"

    return True, None


def merge_template_variables(
    base_vars: dict[str, Any], override_vars: dict[str, Any]
) -> dict[str, Any]:
    """Merge two variable dictionaries with override semantics.

    The override_vars take precedence over base_vars. Performs a shallow merge.

    Args
    ----
        base_vars: Base variable dictionary
        override_vars: Variables to override/add

    Returns
    -------
        Merged dictionary with override_vars taking precedence

    Examples
    --------
        >>> base = {"name": "Alice", "age": 30}
        >>> override = {"age": 31, "city": "NYC"}
        >>> merge_template_variables(base, override)
        {'name': 'Alice', 'age': 31, 'city': 'NYC'}
    """
    return {**base_vars, **override_vars}


def format_template_error(template: str, error: Exception, context_lines: int = 2) -> str:
    r"""Format a template error with context for better debugging.

    Args
    ----
        template: The template that caused the error
        error: The exception that occurred
        context_lines: Number of lines to show around the error

    Returns
    -------
        Formatted error message with template context

    Examples
    --------
        >>> template = "Line 1\\nLine 2 {{invalid}}\\nLine 3"
        >>> error = MissingVariableError("Variable 'invalid' not found")
        >>> print(format_template_error(template, error, context_lines=1))
        Template Error: Variable 'invalid' not found
        Template context:
        Line 2 {{invalid}}
    """
    lines = template.split("\n")
    error_msg = f"Template Error: {error}\n"

    # Try to extract variable name from error message
    # For MissingVariableError, the message often contains the variable name
    error_str = str(error)
    var_name = None

    # Try to extract variable name from error message patterns
    # Common patterns: "variable 'name'", "Cannot access variable 'name'"
    match = re.search(r"variable\s+['\"]?(\w+)['\"]?", error_str, re.IGNORECASE)
    if match:
        var_name = match.group(1)

    if var_name and len(lines) > 1:
        # Find lines containing the variable (for multi-line templates)
        for i, line in enumerate(lines):
            if var_name in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context = "\n".join(lines[start:end])
                error_msg += f"Template context:\n{context}"
                return error_msg

    # Default: show the entire template for single-line or when variable not found
    if len(lines) <= 5:
        error_msg += template
    else:
        # For longer templates, show first few lines
        context = "\n".join(lines[:5])
        error_msg += f"Template beginning:\n{context}..."

    return error_msg
