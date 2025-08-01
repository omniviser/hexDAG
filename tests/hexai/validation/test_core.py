"""Unit tests for core validation functionality."""

from hexai.validation.core import BaseValidator
from hexai.validation.strategies import ValidationStrategy
from hexai.validation.types import ValidationContext
from pydantic import BaseModel


class TestModel(BaseModel):
    """Test Pydantic model for validation tests."""

    name: str
    value: int


class TestBaseValidator:
    """Test cases for BaseValidator."""

    def test_initialization(self):
        """Test BaseValidator initialization."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        assert validator.strategy == ValidationStrategy.STRICT

        validator_coerce = BaseValidator(ValidationStrategy.COERCE)
        assert validator_coerce.strategy == ValidationStrategy.COERCE

    def test_is_pydantic_model(self):
        """Test _is_pydantic_model helper method."""
        validator = BaseValidator(ValidationStrategy.STRICT)

        assert validator._is_pydantic_model(TestModel) is True
        assert validator._is_pydantic_model(BaseModel) is True
        assert validator._is_pydantic_model(str) is False
        assert validator._is_pydantic_model(int) is False
        assert validator._is_pydantic_model(None) is False
        assert validator._is_pydantic_model(list) is False

    def test_is_basic_type(self):
        """Test _is_basic_type helper method."""
        validator = BaseValidator(ValidationStrategy.STRICT)

        assert validator._is_basic_type(str) is True
        assert validator._is_basic_type(int) is True
        assert validator._is_basic_type(float) is True
        assert validator._is_basic_type(bool) is True
        assert validator._is_basic_type(list) is True
        assert validator._is_basic_type(dict) is True

        assert validator._is_basic_type(TestModel) is False
        assert validator._is_basic_type(BaseModel) is False
        assert validator._is_basic_type(None) is False


class TestValidateInput:
    """Test cases for validate_input method."""

    def test_validate_input_no_type_specified(self):
        """Test validation when no expected type is specified."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        result = validator.validate_input("any_data")

        assert result.is_valid is True
        assert result.data == "any_data"
        assert result.errors == []

    def test_validate_input_passthrough_strategy(self):
        """Test validation with passthrough strategy."""
        validator = BaseValidator(ValidationStrategy.PASSTHROUGH)
        result = validator.validate_input("wrong_type", int)

        assert result.is_valid is True
        assert result.data == "wrong_type"
        assert result.errors == []

    def test_validate_input_string_type_strict_success(self):
        """Test string validation with strict strategy - success case."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        result = validator.validate_input("test_string", str)

        assert result.is_valid is True
        assert result.data == "test_string"
        assert result.errors == []

    def test_validate_input_string_type_strict_failure(self):
        """Test string validation with strict strategy - failure case."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        result = validator.validate_input(123, str)

        assert result.is_valid is False
        assert result.data == 123
        assert len(result.errors) == 1
        assert "Expected str, got int" in result.errors[0]

    def test_validate_input_string_type_coerce_success(self):
        """Test string validation with coerce strategy - conversion case."""
        validator = BaseValidator(ValidationStrategy.COERCE)
        result = validator.validate_input(123, str)

        assert result.is_valid is True
        assert result.data == "123"
        assert len(result.warnings) == 1
        assert "Converted int to str" in result.warnings[0]

    def test_validate_input_int_type_coerce(self):
        """Test int validation with coerce strategy."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        # String to int conversion
        result = validator.validate_input("123", int)
        assert result.is_valid is True
        assert result.data == 123
        assert len(result.warnings) == 1

        # Float to int conversion
        result = validator.validate_input(123.7, int)
        assert result.is_valid is True
        assert result.data == 123

    def test_validate_input_float_type_coerce(self):
        """Test float validation with coerce strategy."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        result = validator.validate_input("123.45", float)
        assert result.is_valid is True
        assert result.data == 123.45
        assert len(result.warnings) == 1

    def test_validate_input_bool_type_coerce(self):
        """Test bool validation with coerce strategy."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        result = validator.validate_input("true", bool)
        assert result.is_valid is True
        assert result.data is True  # Valid boolean string

    def test_validate_input_list_type_coerce(self):
        """Test list validation with coerce strategy."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        # Convert iterable to list
        result = validator.validate_input((1, 2, 3), list)
        assert result.is_valid is True
        assert result.data == [1, 2, 3]

        # Wrap non-iterable in list
        result = validator.validate_input("not_iterable", list)
        assert result.is_valid is True
        assert result.data == ["not_iterable"]

    def test_validate_input_dict_type_coerce(self):
        """Test dict validation with coerce strategy."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        # Convert Pydantic model to dict
        test_model = TestModel(name="test", value=42)
        result = validator.validate_input(test_model, dict)
        assert result.is_valid is True
        assert result.data == {"name": "test", "value": 42}

        # Convert object with __dict__ to dict
        class SimpleObj:
            def __init__(self):
                self.attr = "value"

        obj = SimpleObj()
        result = validator.validate_input(obj, dict)
        assert result.is_valid is True
        assert result.data == {"attr": "value"}

        # Wrap other types in dict
        result = validator.validate_input("string", dict)
        assert result.is_valid is True
        assert result.data == {"value": "string"}

    def test_validate_input_pydantic_model_success(self):
        """Test Pydantic model validation - success case."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        data = {"name": "test", "value": 42}
        result = validator.validate_input(data, TestModel)

        assert result.is_valid is True
        assert isinstance(result.data, TestModel)
        assert result.data.name == "test"
        assert result.data.value == 42

    def test_validate_input_pydantic_model_failure_strict(self):
        """Test Pydantic model validation - failure with strict strategy."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        data = {"name": "test", "value": "not_an_int"}
        result = validator.validate_input(data, TestModel)

        assert result.is_valid is False
        assert result.data == data
        assert len(result.errors) == 1
        assert "Pydantic validation failed" in result.errors[0]

    def test_validate_input_pydantic_model_failure_passthrough(self):
        """Test Pydantic model validation - failure with passthrough strategy."""
        validator = BaseValidator(ValidationStrategy.PASSTHROUGH)
        data = {"name": "test", "value": "not_an_int"}
        result = validator.validate_input(data, TestModel)

        assert result.is_valid is True  # Passthrough allows invalid data
        assert result.data == data
        assert len(result.warnings) == 1
        assert "Pydantic validation failed" in result.warnings[0]

    def test_validate_input_with_context(self):
        """Test validation with context information."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        context = ValidationContext(node_name="test_node", pipeline_name="test_pipeline")

        result = validator.validate_input(123, str, context)

        assert result.is_valid is False
        assert "in node 'test_node'" in result.errors[0]

    def test_validate_input_conversion_failure(self):
        """Test validation when type conversion fails."""
        validator = BaseValidator(ValidationStrategy.COERCE)

        # Try to convert invalid string to int
        result = validator.validate_input("not_a_number", int)

        assert result.is_valid is False
        assert result.data == "not_a_number"
        assert len(result.errors) == 1
        assert "Type conversion failed" in result.errors[0]


class TestValidateOutput:
    """Test cases for validate_output method."""

    def test_validate_output_delegates_to_input(self):
        """Test that validate_output delegates to validate_input."""
        validator = BaseValidator(ValidationStrategy.STRICT)

        # Test successful validation
        result = validator.validate_output("test", str)
        assert result.is_valid is True
        assert result.data == "test"

        # Test failed validation
        result = validator.validate_output(123, str)
        assert result.is_valid is False
        assert result.data == 123

    def test_validate_output_context_stage(self):
        """Test that validate_output sets correct context stage."""
        validator = BaseValidator(ValidationStrategy.STRICT)
        context = ValidationContext(node_name="test_node", validation_stage="input")

        result = validator.validate_output(123, str, context)

        # The error should indicate it's in the context, but we can't directly
        # test the stage was changed since it's internal to the method
        assert result.is_valid is False
        assert "in node 'test_node'" in result.errors[0]
