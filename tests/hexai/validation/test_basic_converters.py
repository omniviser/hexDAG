"""Unit tests for basic type converters."""

import pytest

from hexai.validation.basic_converters import BooleanConverter, StringToNumericConverter
from hexai.validation.converters import ConversionError


class TestStringToNumericConverter:
    """Test cases for StringToNumericConverter."""

    def setup_method(self):
        """Setup converter for each test."""
        self.converter = StringToNumericConverter()

    def test_can_convert_str_to_int(self):
        """Test conversion detection for string to int."""
        assert self.converter.can_convert(str, int) is True
        assert self.converter.can_convert(str, float) is True
        assert self.converter.can_convert(int, str) is False
        assert self.converter.can_convert(str, bool) is False

    def test_convert_string_to_int(self):
        """Test string to int conversion."""
        assert self.converter.convert("123", int) == 123
        assert self.converter.convert("-456", int) == -456
        assert self.converter.convert("0", int) == 0

    def test_convert_float_string_to_int(self):
        """Test float string to int conversion."""
        # Should work for integers disguised as floats
        assert self.converter.convert("123.0", int) == 123
        assert self.converter.convert("-456.0", int) == -456

    def test_convert_float_string_to_int_precision_loss(self):
        """Test float string to int conversion with precision loss."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert("123.45", int)
        assert "precision" in str(exc_info.value)

    def test_convert_string_to_float(self):
        """Test string to float conversion."""
        assert self.converter.convert("123", float) == 123.0
        assert self.converter.convert("123.45", float) == 123.45
        assert self.converter.convert("-456.78", float) == -456.78

    def test_convert_empty_string_fails(self):
        """Test empty string conversion fails."""
        with pytest.raises(ConversionError):
            self.converter.convert("", int)
        with pytest.raises(ConversionError):
            self.converter.convert("   ", int)

    def test_convert_invalid_string_fails(self):
        """Test invalid string conversion fails."""
        with pytest.raises(ConversionError):
            self.converter.convert("abc", int)
        with pytest.raises(ConversionError):
            self.converter.convert("12.34.56", float)

    def test_convert_non_string_fails(self):
        """Test non-string input fails."""
        with pytest.raises(ConversionError):
            self.converter.convert(123, int)


class TestBooleanConverter:
    """Test cases for BooleanConverter."""

    def setup_method(self):
        """Setup converter for each test."""
        self.converter = BooleanConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(str, bool) is True
        assert self.converter.can_convert(int, bool) is True
        assert self.converter.can_convert(float, bool) is True
        assert self.converter.can_convert(bool, str) is False
        assert self.converter.can_convert(list, bool) is False

    def test_convert_true_strings(self):
        """Test conversion of true string values."""
        true_values = ["true", "True", "TRUE", "yes", "y", "1", "on", "enabled", "active", "ok"]
        for value in true_values:
            assert self.converter.convert(value, bool) is True
            # Test with whitespace
            assert self.converter.convert(f"  {value}  ", bool) is True

    def test_convert_false_strings(self):
        """Test conversion of false string values."""
        false_values = [
            "false",
            "False",
            "FALSE",
            "no",
            "n",
            "0",
            "off",
            "disabled",
            "inactive",
            "none",
            "null",
        ]
        for value in false_values:
            assert self.converter.convert(value, bool) is False
            # Test with whitespace
            assert self.converter.convert(f"  {value}  ", bool) is False

    def test_convert_invalid_string_fails(self):
        """Test invalid string conversion fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert("maybe", bool)
        assert "Cannot convert string 'maybe' to boolean" in str(exc_info.value)

    def test_convert_numeric_values(self):
        """Test numeric to boolean conversion."""
        assert self.converter.convert(0, bool) is False
        assert self.converter.convert(1, bool) is True
        assert self.converter.convert(0.0, bool) is False
        assert self.converter.convert(1.0, bool) is True

    def test_convert_invalid_numeric_fails(self):
        """Test invalid numeric conversion fails."""
        with pytest.raises(ConversionError):
            self.converter.convert(2, bool)
        with pytest.raises(ConversionError):
            self.converter.convert(-1, bool)
        with pytest.raises(ConversionError):
            self.converter.convert(0.5, bool)

    def test_convert_wrong_target_type_fails(self):
        """Test wrong target type fails."""
        with pytest.raises(ConversionError):
            self.converter.convert("true", str)
