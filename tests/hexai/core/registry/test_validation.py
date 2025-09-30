"""Tests for RegistryValidator class."""

import pytest

from hexai.core.registry.exceptions import InvalidComponentError
from hexai.core.registry.models import (
    COMPONENT_VALUE_ATTR,
    IMPLEMENTS_PORT_ATTR,
    REQUIRED_PORTS_ATTR,
    ClassComponent,
    ComponentType,
    FunctionComponent,
    InstanceComponent,
    Namespace,
)
from hexai.core.registry.validation import RegistryValidator


class TestComponentAttributeExtraction:
    """Tests for component attribute extraction methods."""

    def test_unwrap_component_with_wrapper(self):
        """Test unwrapping a wrapped component."""

        class MockComponent:
            pass

        wrapped = ClassComponent(value=MockComponent)
        unwrapped = RegistryValidator.unwrap_component(wrapped)

        assert unwrapped is MockComponent

    def test_unwrap_component_without_wrapper(self):
        """Test unwrapping a component that's not wrapped."""

        class MockComponent:
            pass

        unwrapped = RegistryValidator.unwrap_component(MockComponent)
        assert unwrapped is MockComponent

    def test_get_implements_port_present(self):
        """Test extracting port when present."""

        class MockAdapter:
            pass

        setattr(MockAdapter, IMPLEMENTS_PORT_ATTR, "test_port")

        port = RegistryValidator.get_implements_port(MockAdapter)
        assert port == "test_port"

    def test_get_implements_port_absent(self):
        """Test extracting port when not present."""

        class MockAdapter:
            pass

        port = RegistryValidator.get_implements_port(MockAdapter)
        assert port is None

    def test_get_implements_port_from_wrapped(self):
        """Test extracting port from wrapped component."""

        class MockAdapter:
            pass

        setattr(MockAdapter, IMPLEMENTS_PORT_ATTR, "test_port")
        wrapped = ClassComponent(value=MockAdapter)

        port = RegistryValidator.get_implements_port(wrapped)
        assert port == "test_port"

    def test_get_required_ports_present(self):
        """Test extracting required ports when present."""

        class MockNode:
            pass

        setattr(MockNode, REQUIRED_PORTS_ATTR, ["llm", "database"])

        ports = RegistryValidator.get_required_ports(MockNode)
        assert ports == ["llm", "database"]

    def test_get_required_ports_absent(self):
        """Test extracting required ports when not present."""

        class MockNode:
            pass

        ports = RegistryValidator.get_required_ports(MockNode)
        assert ports == []

    def test_get_required_ports_from_wrapped(self):
        """Test extracting required ports from wrapped component."""

        class MockNode:
            pass

        setattr(MockNode, REQUIRED_PORTS_ATTR, ["llm"])
        wrapped = ClassComponent(value=MockNode)

        ports = RegistryValidator.get_required_ports(wrapped)
        assert ports == ["llm"]

    def test_attribute_constants_match(self):
        """Verify attribute constants are correct."""
        assert COMPONENT_VALUE_ATTR == "value"
        assert IMPLEMENTS_PORT_ATTR == "_hexdag_implements_port"
        assert REQUIRED_PORTS_ATTR == "_hexdag_required_ports"


class TestComponentValidation:
    """Tests for component validation methods."""

    def test_validate_component_type_valid(self):
        """Test validating valid component types."""
        assert RegistryValidator.validate_component_type("node") == ComponentType.NODE
        assert RegistryValidator.validate_component_type("adapter") == ComponentType.ADAPTER
        assert RegistryValidator.validate_component_type("tool") == ComponentType.TOOL
        assert RegistryValidator.validate_component_type("port") == ComponentType.PORT

    def test_validate_component_type_invalid(self):
        """Test validating invalid component type."""
        with pytest.raises(InvalidComponentError) as exc_info:
            RegistryValidator.validate_component_type("invalid_type")

        assert "invalid_type" in str(exc_info.value)
        assert "Invalid component type" in str(exc_info.value)

    def test_validate_component_name_valid(self):
        """Test validating valid component names."""
        RegistryValidator.validate_component_name("my_component")
        RegistryValidator.validate_component_name("Component123")
        RegistryValidator.validate_component_name("test_node_2")
        # Should not raise

    def test_validate_component_name_empty(self):
        """Test validating empty component name."""
        with pytest.raises(InvalidComponentError) as exc_info:
            RegistryValidator.validate_component_name("")

        assert "non-empty" in str(exc_info.value)

    def test_validate_component_name_invalid_chars(self):
        """Test validating name with invalid characters."""
        with pytest.raises(InvalidComponentError) as exc_info:
            RegistryValidator.validate_component_name("my-component")

        assert "alphanumeric" in str(exc_info.value)

        with pytest.raises(InvalidComponentError):
            RegistryValidator.validate_component_name("my.component")

        with pytest.raises(InvalidComponentError):
            RegistryValidator.validate_component_name("my component")

    def test_validate_namespace_none_defaults_to_user(self):
        """Test that None namespace defaults to 'user'."""
        assert RegistryValidator.validate_namespace(None) == Namespace.USER

    def test_validate_namespace_empty_defaults_to_user(self):
        """Test that empty string defaults to 'user'."""
        assert RegistryValidator.validate_namespace("") == Namespace.USER

    def test_validate_namespace_valid(self):
        """Test validating valid namespaces."""
        assert RegistryValidator.validate_namespace("core") == "core"
        assert RegistryValidator.validate_namespace("plugin") == "plugin"
        assert RegistryValidator.validate_namespace("MyNamespace") == "mynamespace"  # Lowercased

    def test_validate_namespace_lowercase_normalization(self):
        """Test that namespaces are normalized to lowercase."""
        assert RegistryValidator.validate_namespace("UPPERCASE") == "uppercase"
        assert RegistryValidator.validate_namespace("MixedCase") == "mixedcase"

    def test_validate_namespace_invalid_chars(self):
        """Test validating namespace with invalid characters."""
        with pytest.raises(InvalidComponentError) as exc_info:
            RegistryValidator.validate_namespace("my-namespace")

        assert "alphanumeric" in str(exc_info.value)

        with pytest.raises(InvalidComponentError):
            RegistryValidator.validate_namespace("my.namespace")

    def test_wrap_component_class(self):
        """Test wrapping a class component."""

        class MyClass:
            pass

        wrapped = RegistryValidator.wrap_component(MyClass)
        assert isinstance(wrapped, ClassComponent)
        assert wrapped.value is MyClass

    def test_wrap_component_function(self):
        """Test wrapping a function component."""

        def my_function():
            pass

        wrapped = RegistryValidator.wrap_component(my_function)
        assert isinstance(wrapped, FunctionComponent)
        assert wrapped.value is my_function

    def test_wrap_component_method(self):
        """Test wrapping a method component."""

        class MyClass:
            def my_method(self):
                pass

        wrapped = RegistryValidator.wrap_component(MyClass.my_method)
        assert isinstance(wrapped, FunctionComponent)
        assert wrapped.value is MyClass.my_method

    def test_wrap_component_instance(self):
        """Test wrapping an instance component."""

        class MyClass:
            pass

        instance = MyClass()
        wrapped = RegistryValidator.wrap_component(instance)
        assert isinstance(wrapped, InstanceComponent)
        assert wrapped.value is instance


class TestNamespaceProtection:
    """Tests for namespace protection checking."""

    def test_is_protected_namespace_core(self):
        """Test that 'core' namespace is protected."""
        assert RegistryValidator.is_protected_namespace(Namespace.CORE) is True
        assert RegistryValidator.is_protected_namespace("core") is True

    def test_is_protected_namespace_user(self):
        """Test that 'user' namespace is not protected."""
        assert RegistryValidator.is_protected_namespace(Namespace.USER) is False
        assert RegistryValidator.is_protected_namespace("user") is False

    def test_is_protected_namespace_plugin(self):
        """Test that 'plugin' namespace is not protected."""
        assert RegistryValidator.is_protected_namespace(Namespace.PLUGIN) is False
        assert RegistryValidator.is_protected_namespace("plugin") is False

    def test_is_protected_namespace_custom(self):
        """Test that custom namespaces are not protected."""
        assert RegistryValidator.is_protected_namespace("custom") is False
        assert RegistryValidator.is_protected_namespace("myapp") is False

    def test_protected_namespaces_constant(self):
        """Test that PROTECTED_NAMESPACES contains expected values."""
        assert Namespace.CORE in RegistryValidator.PROTECTED_NAMESPACES
        assert len(RegistryValidator.PROTECTED_NAMESPACES) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unwrap_component_with_none_value(self):
        """Test unwrapping component that returns None."""

        class WeirdComponent:
            value = None

        result = RegistryValidator.unwrap_component(WeirdComponent)
        # Should return None (the value attribute)
        assert result is None

    def test_get_implements_port_with_none_value(self):
        """Test getting implements port when attribute is None."""

        class MockAdapter:
            pass

        setattr(MockAdapter, IMPLEMENTS_PORT_ATTR, None)

        # Should still extract the attribute even if it's None
        port = RegistryValidator.get_implements_port(MockAdapter)
        assert port is None

    def test_validate_component_name_with_numbers(self):
        """Test that names with numbers are valid."""
        RegistryValidator.validate_component_name("component123")
        RegistryValidator.validate_component_name("123component")
        RegistryValidator.validate_component_name("comp123onent")
        # Should not raise

    def test_validate_component_name_only_numbers(self):
        """Test that names with only numbers are valid."""
        RegistryValidator.validate_component_name("123456")
        # Should not raise

    def test_validate_component_name_only_underscores(self):
        """Test that names with only underscores are valid."""
        RegistryValidator.validate_component_name("___")
        # Should not raise

    def test_wrap_component_with_callable_instance(self):
        """Test wrapping an instance with __call__ method."""

        class CallableClass:
            def __call__(self):
                pass

        instance = CallableClass()
        wrapped = RegistryValidator.wrap_component(instance)
        # Should still be InstanceComponent, not FunctionComponent
        assert isinstance(wrapped, InstanceComponent)
        assert wrapped.value is instance
