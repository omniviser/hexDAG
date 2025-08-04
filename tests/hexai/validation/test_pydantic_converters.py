"""Unit tests for Pydantic model converters."""

import json

import pytest
from pydantic import BaseModel, Field

from hexai.validation.converters import ConversionError
from hexai.validation.pydantic_converters import (
    DictToPydanticConverter,
    JsonStringToPydanticConverter,
    NestedPydanticConverter,
    PydanticToDictConverter,
    PydanticToJsonConverter,
)


# Test models
class UserModel(BaseModel):
    """Test user model."""

    name: str
    age: int
    email: str | None = None


class ProfileModel(BaseModel):
    """Test profile model."""

    username: str
    display_name: str
    age: int = Field(ge=0)


class CompatibleUserModel(BaseModel):
    """Test user model compatible with ProfileModel."""

    username: str
    display_name: str
    age: int
    email: str | None = None


class AddressModel(BaseModel):
    """Test address model."""

    street: str
    city: str
    country: str = "USA"


class PersonModel(BaseModel):
    """Test person model with nested address."""

    name: str
    address: AddressModel


class TestDictToPydanticConverter:
    """Test cases for DictToPydanticConverter."""

    def setup_method(self):
        """Set up converter for each test."""
        self.converter = DictToPydanticConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(dict, UserModel) is True
        assert self.converter.can_convert(dict, str) is False
        assert self.converter.can_convert(str, UserModel) is False
        assert self.converter.can_convert(list, UserModel) is False

    def test_convert_valid_dict(self):
        """Test valid dictionary conversion."""
        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}
        result = self.converter.convert(data, UserModel)

        assert isinstance(result, UserModel)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_convert_dict_with_optional_fields(self):
        """Test dictionary conversion with optional fields."""
        data = {"name": "Jane Doe", "age": 25}
        result = self.converter.convert(data, UserModel)

        assert isinstance(result, UserModel)
        assert result.name == "Jane Doe"
        assert result.age == 25
        assert result.email is None

    def test_convert_dict_with_extra_fields(self):
        """Test dictionary conversion with extra fields (should be ignored)."""
        data = {"name": "Bob Smith", "age": 35, "extra_field": "ignored"}
        result = self.converter.convert(data, UserModel)

        assert isinstance(result, UserModel)
        assert result.name == "Bob Smith"
        assert result.age == 35

    def test_convert_invalid_dict_missing_required_field(self):
        """Test conversion fails with missing required field."""
        data = {"name": "Alice"}  # Missing age

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(data, UserModel)
        assert "Pydantic validation failed" in str(exc_info.value)
        assert "age" in str(exc_info.value)

    def test_convert_invalid_dict_wrong_type(self):
        """Test conversion fails with wrong field type."""
        data = {"name": "Charlie", "age": "thirty"}  # age should be int

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(data, UserModel)
        assert "Pydantic validation failed" in str(exc_info.value)

    def test_convert_non_dict_fails(self):
        """Test non-dict input fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert("not a dict", UserModel)
        assert "Value must be a dictionary" in str(exc_info.value)

    def test_convert_non_pydantic_target_fails(self):
        """Test non-Pydantic target type fails."""
        data = {"name": "Test"}
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(data, str)
        assert "is not a Pydantic model" in str(exc_info.value)


class TestPydanticToDictConverter:
    """Test cases for PydanticToDictConverter."""

    def setup_method(self):
        """Set up converter for each test."""
        self.converter = PydanticToDictConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(UserModel, dict) is True
        assert self.converter.can_convert(dict, dict) is False
        assert self.converter.can_convert(UserModel, str) is False
        assert self.converter.can_convert(str, dict) is False

    def test_convert_valid_model(self):
        """Test valid Pydantic model conversion."""
        user = UserModel(name="John Doe", age=30, email="john@example.com")
        result = self.converter.convert(user, dict)

        assert isinstance(result, dict)
        assert result == {"name": "John Doe", "age": 30, "email": "john@example.com"}

    def test_convert_model_with_default_values(self):
        """Test model conversion with default values."""
        user = UserModel(name="Jane Doe", age=25)
        result = self.converter.convert(user, dict)

        assert isinstance(result, dict)
        assert result == {"name": "Jane Doe", "age": 25, "email": None}

    def test_convert_nested_model(self):
        """Test nested model conversion."""
        address = AddressModel(street="123 Main St", city="New York")
        person = PersonModel(name="Alice", address=address)
        result = self.converter.convert(person, dict)

        assert isinstance(result, dict)
        expected = {
            "name": "Alice",
            "address": {"street": "123 Main St", "city": "New York", "country": "USA"},
        }
        assert result == expected

    def test_convert_non_model_fails(self):
        """Test non-Pydantic model input fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert({"not": "a model"}, dict)
        assert "Value must be a Pydantic model instance" in str(exc_info.value)

    def test_convert_wrong_target_type_fails(self):
        """Test wrong target type fails."""
        user = UserModel(name="Test", age=30)
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(user, str)
        assert "Target type must be dict" in str(exc_info.value)


class TestJsonStringToPydanticConverter:
    """Test cases for JsonStringToPydanticConverter."""

    def setup_method(self):
        """Set up converter for each test."""
        self.converter = JsonStringToPydanticConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(str, UserModel) is True
        assert self.converter.can_convert(dict, UserModel) is False
        assert self.converter.can_convert(str, dict) is False
        assert self.converter.can_convert(int, UserModel) is False

    def test_convert_valid_json(self):
        """Test valid JSON string conversion."""
        json_str = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        result = self.converter.convert(json_str, UserModel)

        assert isinstance(result, UserModel)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_convert_json_with_optional_fields(self):
        """Test JSON conversion with optional fields."""
        json_str = '{"name": "Jane Doe", "age": 25}'
        result = self.converter.convert(json_str, UserModel)

        assert isinstance(result, UserModel)
        assert result.name == "Jane Doe"
        assert result.age == 25
        assert result.email is None

    def test_convert_nested_json(self):
        """Test nested JSON conversion."""
        json_str = '{"name": "Alice", "address": {"street": "123 Main St", "city": "New York"}}'
        result = self.converter.convert(json_str, PersonModel)

        assert isinstance(result, PersonModel)
        assert result.name == "Alice"
        assert isinstance(result.address, AddressModel)
        assert result.address.street == "123 Main St"
        assert result.address.city == "New York"
        assert result.address.country == "USA"  # Default value

    def test_convert_invalid_json_fails(self):
        """Test invalid JSON string fails."""
        invalid_json = '{"name": "John", "age": 30'  # Missing closing brace

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(invalid_json, UserModel)
        assert "Invalid JSON string" in str(exc_info.value)

    def test_convert_json_array_fails(self):
        """Test JSON array (not object) fails."""
        json_array = '["item1", "item2"]'

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(json_array, UserModel)
        assert "JSON must represent an object" in str(exc_info.value)

    def test_convert_json_validation_fails(self):
        """Test JSON with validation errors fails."""
        json_str = '{"name": "John"}'  # Missing required age field

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(json_str, UserModel)
        assert "Pydantic validation failed" in str(exc_info.value)

    def test_convert_non_string_fails(self):
        """Test non-string input fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert({"name": "Test"}, UserModel)
        assert "Value must be a string" in str(exc_info.value)


class TestPydanticToJsonConverter:
    """Test cases for PydanticToJsonConverter."""

    def setup_method(self):
        """Set up converter for each test."""
        self.converter = PydanticToJsonConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(UserModel, str) is True
        assert self.converter.can_convert(dict, str) is False
        assert self.converter.can_convert(UserModel, dict) is False

    def test_convert_valid_model(self):
        """Test valid model conversion."""
        user = UserModel(name="John Doe", age=30, email="john@example.com")
        result = self.converter.convert(user, str)

        assert isinstance(result, str)
        # Parse back to verify
        data = json.loads(result)
        assert data == {"name": "John Doe", "age": 30, "email": "john@example.com"}

    def test_convert_nested_model(self):
        """Test nested model conversion."""
        address = AddressModel(street="123 Main St", city="New York")
        person = PersonModel(name="Alice", address=address)
        result = self.converter.convert(person, str)

        assert isinstance(result, str)
        data = json.loads(result)
        expected = {
            "name": "Alice",
            "address": {"street": "123 Main St", "city": "New York", "country": "USA"},
        }
        assert data == expected

    def test_convert_non_model_fails(self):
        """Test non-model input fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert({"not": "a model"}, str)
        assert "Value must be a Pydantic model instance" in str(exc_info.value)


class TestNestedPydanticConverter:
    """Test cases for NestedPydanticConverter."""

    def setup_method(self):
        """Set up converter for each test."""
        self.converter = NestedPydanticConverter()

    def test_can_convert(self):
        """Test conversion detection."""
        assert self.converter.can_convert(CompatibleUserModel, ProfileModel) is True
        assert self.converter.can_convert(UserModel, ProfileModel) is False  # Incompatible fields
        assert self.converter.can_convert(UserModel, UserModel) is False  # Same type
        assert self.converter.can_convert(dict, UserModel) is False
        assert self.converter.can_convert(UserModel, dict) is False

    def test_convert_compatible_models(self):
        """Test conversion between compatible models."""
        user = CompatibleUserModel(username="johndoe", display_name="John Doe", age=30)
        result = self.converter.convert(user, ProfileModel)

        assert isinstance(result, ProfileModel)
        assert result.username == "johndoe"
        assert result.display_name == "John Doe"
        assert result.age == 30

    def test_convert_incompatible_models_fails(self):
        """Test conversion between incompatible models fails."""
        user = UserModel(name="John Doe", age=30)

        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert(user, ProfileModel)
        assert "Failed to convert between Pydantic models" in str(exc_info.value)

    def test_convert_non_model_fails(self):
        """Test non-model input fails."""
        with pytest.raises(ConversionError) as exc_info:
            self.converter.convert({"not": "a model"}, UserModel)
        assert "Value must be a Pydantic model instance" in str(exc_info.value)
