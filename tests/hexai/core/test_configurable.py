"""Tests for the configurable component system."""

from pydantic import BaseModel, Field


class TestConfigurableComponent:
    """Test the ConfigurableComponent protocol."""

    def test_protocol_implementation(self) -> None:
        """Test that classes can implement the ConfigurableComponent protocol."""

        class MyConfig(BaseModel):
            """Configuration for test component."""

            api_key: str = Field(default="test-key", description="API key")
            timeout: int = Field(default=30, description="Timeout in seconds")

        class TestAdapter:
            """Test adapter that implements ConfigurableComponent."""

            @classmethod
            def get_config_class(cls) -> type[BaseModel]:
                """Return configuration class."""
                return MyConfig

        # Check that the adapter implements the protocol
        adapter = TestAdapter()
        assert hasattr(adapter, "get_config_class")

        # Check the methods work
        assert adapter.get_config_class() == MyConfig

    def test_config_class_fields(self) -> None:
        """Test that config class fields are accessible."""

        class TestConfig(BaseModel):
            """Test configuration."""

            field1: str = Field(default="value1", description="First field")
            field2: int = Field(default=42, description="Second field")
            field3: bool = Field(default=True, description="Third field")

        class TestComponent:
            """Test component."""

            @classmethod
            def get_config_class(cls) -> type[BaseModel]:
                return TestConfig

        config_class = TestComponent.get_config_class()
        fields = config_class.model_fields

        assert "field1" in fields
        assert "field2" in fields
        assert "field3" in fields

        assert fields["field1"].description == "First field"
        assert fields["field2"].description == "Second field"
        assert fields["field3"].description == "Third field"
