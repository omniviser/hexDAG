"""Unit tests for ValidationStrategy enum."""

from hexai.validation.strategies import ValidationStrategy


class TestValidationStrategy:
    """Test cases for ValidationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have correct string values."""
        assert ValidationStrategy.STRICT.value == "strict"
        assert ValidationStrategy.COERCE.value == "coerce"
        assert ValidationStrategy.PASSTHROUGH.value == "passthrough"

    def test_string_representation(self):
        """Test string representation of strategies."""
        assert str(ValidationStrategy.STRICT) == "strict"
        assert str(ValidationStrategy.COERCE) == "coerce"
        assert str(ValidationStrategy.PASSTHROUGH) == "passthrough"

    def test_enum_membership(self):
        """Test that all expected strategies are defined."""
        expected_strategies = {"STRICT", "COERCE", "PASSTHROUGH"}
        actual_strategies = {strategy.name for strategy in ValidationStrategy}
        assert actual_strategies == expected_strategies

    def test_enum_equality(self):
        """Test equality comparison of strategies."""
        assert ValidationStrategy.STRICT == ValidationStrategy.STRICT
        assert ValidationStrategy.COERCE == ValidationStrategy.COERCE
        assert ValidationStrategy.PASSTHROUGH == ValidationStrategy.PASSTHROUGH

        assert ValidationStrategy.STRICT != ValidationStrategy.COERCE
        assert ValidationStrategy.COERCE != ValidationStrategy.PASSTHROUGH

    def test_strategy_count(self):
        """Test that we have exactly 3 strategies."""
        assert len(list(ValidationStrategy)) == 3
