"""Tests for the simplified metadata module."""

import pytest

from hexai.core.registry.metadata import ComponentMetadata, InstanceFactory
from hexai.core.registry.types import ComponentType, NodeSubtype


class TestComponentMetadata:
    """Test the ComponentMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic metadata creation."""

        class TestComponent:
            pass

        meta = ComponentMetadata(
            name="test_component",
            component_type=ComponentType.NODE,
            component=TestComponent,
            namespace="test",
        )

        assert meta.name == "test_component"
        assert meta.component_type == ComponentType.NODE
        assert meta.component is TestComponent
        assert meta.namespace == "test"
        assert meta.description == ""
        assert meta.subtype is None

    def test_with_description(self):
        """Test metadata with description."""

        def test_function():
            pass

        meta = ComponentMetadata(
            name="test_function",
            component_type=ComponentType.TOOL,
            component=test_function,
            namespace="plugins",
            description="A test function",
        )

        assert meta.name == "test_function"
        assert meta.component_type == ComponentType.TOOL
        assert meta.component is test_function
        assert meta.namespace == "plugins"
        assert meta.description == "A test function"

    def test_with_subtype(self):
        """Test metadata with node subtype."""

        class TestNode:
            pass

        meta = ComponentMetadata(
            name="test_node",
            component_type=ComponentType.NODE,
            component=TestNode,
            namespace="test",
            subtype=NodeSubtype.FUNCTION,
        )

        assert meta.name == "test_node"
        assert meta.component_type == ComponentType.NODE
        assert meta.subtype == NodeSubtype.FUNCTION

    def test_is_core_property(self):
        """Test is_core property."""

        class TestComponent:
            pass

        # Core component
        core_meta = ComponentMetadata(
            name="core_component",
            component_type=ComponentType.NODE,
            component=TestComponent,
            namespace="core",
        )
        assert core_meta.is_core is True

        # Non-core component
        user_meta = ComponentMetadata(
            name="user_component",
            component_type=ComponentType.NODE,
            component=TestComponent,
            namespace="user",
        )
        assert user_meta.is_core is False


class TestLazyLoading:
    """Test lazy loading functionality."""

    def test_lazy_component_metadata(self):
        """Test creating lazy component metadata."""
        meta = ComponentMetadata(
            name="lazy_component",
            component_type=ComponentType.NODE,
            component=None,  # Lazy - no component yet
            namespace="test",
            is_lazy=True,
            import_path="test.module",
            attribute_name="MyComponent",
        )

        assert meta.is_lazy is True
        assert meta.component is None
        assert meta.import_path == "test.module"
        assert meta.attribute_name == "MyComponent"

    def test_resolve_lazy_component_import_error(self):
        """Test lazy component resolution with import error."""
        meta = ComponentMetadata(
            name="bad_lazy",
            component_type=ComponentType.NODE,
            component=None,
            namespace="test",
            is_lazy=True,
            import_path="nonexistent.module",
            attribute_name="Component",
        )

        with pytest.raises(ImportError):
            meta.resolve_lazy_component()

    def test_resolve_lazy_component_attribute_error(self):
        """Test lazy component resolution with missing attribute."""
        import sys
        from types import ModuleType

        mock_module = ModuleType("test_module_no_attr")
        sys.modules["test_module_no_attr"] = mock_module

        try:
            meta = ComponentMetadata(
                name="lazy_no_attr",
                component_type=ComponentType.NODE,
                component=None,
                namespace="test",
                is_lazy=True,
                import_path="test_module_no_attr",
                attribute_name="NonExistentComponent",
            )

            with pytest.raises(AttributeError):
                meta.resolve_lazy_component()
        finally:
            del sys.modules["test_module_no_attr"]

    def test_non_lazy_component(self):
        """Test that non-lazy components don't have lazy attributes."""

        class TestComponent:
            pass

        meta = ComponentMetadata(
            name="regular",
            component_type=ComponentType.NODE,
            component=TestComponent,
            namespace="test",
        )

        assert meta.is_lazy is False
        assert meta.import_path is None
        assert meta.attribute_name is None
        assert meta.component is TestComponent


class TestInstanceFactory:
    """Test the InstanceFactory utility."""

    def test_create_class_instance(self):
        """Test creating instance from a class."""

        class TestClass:
            def __init__(self, value=42):
                self.value = value

        # Default arguments
        instance = InstanceFactory.create_instance(TestClass)
        assert isinstance(instance, TestClass)
        assert instance.value == 42

        # Custom arguments
        instance = InstanceFactory.create_instance(TestClass, value=100)
        assert instance.value == 100

    def test_create_function_instance(self):
        """Test that functions are returned as-is, not called."""

        def test_function(value=10):
            return f"test-{value}"

        # Functions should be returned as-is, not called
        func = InstanceFactory.create_instance(test_function)
        assert func is test_function
        assert callable(func)

        # kwargs should be ignored for functions
        func = InstanceFactory.create_instance(test_function, value=20)
        assert func is test_function

        # Can call the function normally after getting it
        assert func() == "test-10"
        assert func(value=20) == "test-20"

    def test_create_instance_with_args(self):
        """Test creating instance with keyword args only (no positional)."""

        class TestClass:
            def __init__(self, a=1, b=2, c=3):
                self.a = a
                self.b = b
                self.c = c

        # InstanceFactory only supports kwargs, not positional args
        instance = InstanceFactory.create_instance(TestClass, a=10, b=20, c=30)
        assert instance.a == 10
        assert instance.b == 20
        assert instance.c == 30

    def test_create_singleton_instance(self):
        """Test that non-class objects are returned as-is."""
        obj = object()
        instance = InstanceFactory.create_instance(obj)
        assert instance is obj

    def test_create_instance_error_handling(self):
        """Test error handling in instance creation."""

        class BadClass:
            def __init__(self):
                raise ValueError("Cannot create instance")

        with pytest.raises(ValueError):
            InstanceFactory.create_instance(BadClass)
