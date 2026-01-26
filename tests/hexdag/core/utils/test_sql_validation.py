"""Tests for the SQL validation module.

This module tests SQL identifier validation to prevent injection attacks.
"""

from __future__ import annotations

import pytest

from hexdag.core.utils.sql_validation import validate_sql_identifier


class TestValidateSqlIdentifier:
    """Tests for validate_sql_identifier function."""

    def test_valid_simple_identifier(self) -> None:
        """Test that simple identifiers are valid."""
        assert validate_sql_identifier("users") is True
        assert validate_sql_identifier("table") is True
        assert validate_sql_identifier("column") is True

    def test_valid_identifier_with_underscore(self) -> None:
        """Test that identifiers with underscores are valid."""
        assert validate_sql_identifier("user_data") is True
        assert validate_sql_identifier("my_table_name") is True
        assert validate_sql_identifier("_private") is True

    def test_valid_identifier_starting_with_underscore(self) -> None:
        """Test that identifiers starting with underscore are valid."""
        assert validate_sql_identifier("_hidden") is True
        assert validate_sql_identifier("__dunder") is True

    def test_valid_identifier_with_numbers(self) -> None:
        """Test that identifiers with numbers are valid."""
        assert validate_sql_identifier("table123") is True
        assert validate_sql_identifier("user_v2") is True
        assert validate_sql_identifier("data2024") is True

    def test_valid_mixed_case(self) -> None:
        """Test that mixed case identifiers are valid."""
        assert validate_sql_identifier("UserTable") is True
        assert validate_sql_identifier("myColumn") is True
        assert validate_sql_identifier("TABLE_NAME") is True

    def test_invalid_starts_with_number(self) -> None:
        """Test that identifiers starting with numbers are invalid."""
        assert validate_sql_identifier("123table") is False
        assert validate_sql_identifier("1user") is False
        assert validate_sql_identifier("0_data") is False

    def test_invalid_contains_dash(self) -> None:
        """Test that identifiers with dashes are invalid."""
        assert validate_sql_identifier("user-data") is False
        assert validate_sql_identifier("my-table") is False

    def test_invalid_contains_dot(self) -> None:
        """Test that identifiers with dots are invalid."""
        assert validate_sql_identifier("user.table") is False
        assert validate_sql_identifier("schema.name") is False

    def test_invalid_contains_space(self) -> None:
        """Test that identifiers with spaces are invalid."""
        assert validate_sql_identifier("user data") is False
        assert validate_sql_identifier("my table") is False

    def test_invalid_contains_special_chars(self) -> None:
        """Test that identifiers with special characters are invalid."""
        assert validate_sql_identifier("user@data") is False
        assert validate_sql_identifier("table$name") is False
        assert validate_sql_identifier("col#1") is False
        assert validate_sql_identifier("name;drop") is False

    def test_raise_on_invalid_true(self) -> None:
        """Test that raise_on_invalid=True raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("user-data", raise_on_invalid=True)
        assert "Invalid identifier" in str(exc_info.value)
        assert "user-data" in str(exc_info.value)

    def test_raise_on_invalid_with_custom_type(self) -> None:
        """Test error message with custom identifier type."""
        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("bad.table", "table", raise_on_invalid=True)
        assert "Invalid table" in str(exc_info.value)
        assert "bad.table" in str(exc_info.value)

    def test_raise_on_invalid_false_returns_false(self) -> None:
        """Test that raise_on_invalid=False returns False for invalid."""
        # Should not raise, just return False
        result = validate_sql_identifier("invalid-name", raise_on_invalid=False)
        assert result is False

    def test_custom_identifier_type_in_warning(self) -> None:
        """Test that custom identifier type is used in messages."""
        # This tests the logging path - just verify it doesn't raise
        result = validate_sql_identifier("bad.column", "column")
        assert result is False
