"""Tests for AdapterValidator."""

import pytest

from hexai.core.registry.adapter_validator import AdapterValidator
from hexai.core.registry.component_store import ComponentStore
from hexai.core.registry.exceptions import InvalidComponentError
from hexai.core.registry.models import (
    IMPLEMENTS_PORT_ATTR,
    ClassComponent,
    ComponentMetadata,
    ComponentType,
)


class TestAdapterValidator:
    """Test AdapterValidator functionality."""

    @pytest.fixture
    def store(self):
        """Create a component store."""
        return ComponentStore()

    @pytest.fixture
    def validator(self, store):
        """Create an adapter validator."""
        return AdapterValidator(store)

    def test_extract_implements_port_present(self, validator):
        """Test extracting implements_port when present."""

        class MyAdapter:
            pass

        setattr(MyAdapter, IMPLEMENTS_PORT_ATTR, "llm")

        port = validator._extract_implements_port(MyAdapter)
        assert port == "llm"

    def test_extract_implements_port_absent(self, validator):
        """Test extracting implements_port when absent."""

        class MyAdapter:
            pass

        port = validator._extract_implements_port(MyAdapter)
        assert port is None

    def test_extract_implements_port_from_wrapped(self, validator):
        """Test extracting from wrapped component."""

        class MyAdapter:
            pass

        setattr(MyAdapter, IMPLEMENTS_PORT_ATTR, "database")
        wrapped = ClassComponent(value=MyAdapter)

        port = validator._extract_implements_port(wrapped)
        assert port == "database"

    def test_build_search_attempts_qualified(self, validator):
        """Test building search attempts for qualified port name."""
        attempts = validator._build_search_attempts("core:llm", "user")
        assert attempts == ["core:llm"]

    def test_build_search_attempts_unqualified(self, validator):
        """Test building search attempts for unqualified port name."""
        attempts = validator._build_search_attempts("llm", "plugin")
        assert attempts == [
            "core:llm",  # Core first
            "plugin:llm",  # Same namespace as adapter
            "llm",  # As declared
        ]

    def test_validate_adapter_no_port_declared(self, validator, store):
        """Test validating adapter with no port declaration."""

        class MyAdapter:
            pass

        # Should not raise - no port declared
        validator.validate_adapter_registration("my_adapter", MyAdapter, "user")

    def test_validate_adapter_port_not_found(self, validator, store):
        """Test validating adapter when port doesn't exist."""

        class MyAdapter:
            pass

        setattr(MyAdapter, IMPLEMENTS_PORT_ATTR, "nonexistent_port")

        with pytest.raises(InvalidComponentError) as exc_info:
            validator.validate_adapter_registration("my_adapter", MyAdapter, "user")

        assert "does not exist in registry" in str(exc_info.value)

    def test_find_port_exact_match(self, validator, store):
        """Test finding port with exact name match."""
        # Register a port
        from typing import Protocol

        class LLMPort(Protocol):
            def generate(self, prompt: str) -> str: ...

        port_meta = ComponentMetadata(
            name="llm",
            component_type=ComponentType.PORT,
            component=ClassComponent(value=LLMPort),
            namespace="core",
        )
        store.register(port_meta, "core", is_protected=True)

        # Find it
        found = validator._find_port("llm", "user", "test_adapter")
        assert found is not None
        assert found.name == "llm"

    def test_find_port_qualified_match(self, validator, store):
        """Test finding port with qualified name."""
        from typing import Protocol

        class DatabasePort(Protocol):
            def query(self, sql: str) -> list: ...

        port_meta = ComponentMetadata(
            name="database",
            component_type=ComponentType.PORT,
            component=ClassComponent(value=DatabasePort),
            namespace="core",
        )
        store.register(port_meta, "core", is_protected=True)

        # Find with qualified name
        found = validator._find_port("core:database", "user", "test_adapter")
        assert found is not None
        assert found.name == "database"

    def test_find_port_not_found(self, validator, store):
        """Test finding non-existent port."""
        found = validator._find_port("nonexistent", "user", "test_adapter")
        assert found is None
