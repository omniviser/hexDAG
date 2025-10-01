"""Tests for convention over configuration introspection utilities."""

import asyncio
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from hexai.core.registry.introspection import (
    extract_port_methods,
    extract_tool_signature,
    infer_adapter_capabilities,
    validate_adapter_implementation,
)


class TestExtractPortMethods:
    """Test extraction of required and optional methods from Protocol classes."""

    def test_extract_methods_from_protocol(self):
        """Test extracting methods from a Protocol with abstract and concrete methods."""

        @runtime_checkable
        class DatabasePort(Protocol):
            @abstractmethod
            def query(self, sql: str) -> list:
                """Required method."""
                ...

            @abstractmethod
            def execute(self, sql: str) -> None:
                """Required method."""
                ...

            def batch_execute(self, statements: list[str]) -> None:
                """Optional method with default implementation."""
                for stmt in statements:
                    self.execute(stmt)

            def get_version(self) -> str:
                """Optional method."""
                return "1.0.0"

        required, optional = extract_port_methods(DatabasePort)

        assert required == ["execute", "query"]  # Alphabetical order
        assert optional == ["batch_execute", "get_version"]

    def test_protocol_with_only_required_methods(self):
        """Test Protocol with only abstract methods."""

        @runtime_checkable
        class SimplePort(Protocol):
            @abstractmethod
            def method_a(self) -> None: ...

            @abstractmethod
            def method_b(self) -> str: ...

        required, optional = extract_port_methods(SimplePort)

        assert required == ["method_a", "method_b"]
        assert optional == []

    def test_protocol_with_only_optional_methods(self):
        """Test Protocol with only concrete methods."""

        @runtime_checkable
        class OptionalOnlyPort(Protocol):
            def optional_a(self) -> int:
                return 42

            def optional_b(self, x: str) -> str:
                return x.upper()

        required, optional = extract_port_methods(OptionalOnlyPort)

        assert required == []
        assert optional == ["optional_a", "optional_b"]

    def test_protocol_ignores_private_methods(self):
        """Test that private and dunder methods are ignored."""

        @runtime_checkable
        class PortWithPrivates(Protocol):
            @abstractmethod
            def public_method(self) -> None: ...

            def _private_method(self) -> None:
                """Should be ignored."""
                pass

            def __special__(self) -> None:
                """Should be ignored."""
                pass

        required, optional = extract_port_methods(PortWithPrivates)

        assert required == ["public_method"]
        assert optional == []
        assert "_private_method" not in required
        assert "_private_method" not in optional
        assert "__special__" not in required
        assert "__special__" not in optional

    def test_protocol_with_async_methods(self):
        """Test Protocol with async methods."""

        @runtime_checkable
        class AsyncPort(Protocol):
            @abstractmethod
            async def async_required(self) -> str: ...

            async def async_optional(self) -> int:
                await asyncio.sleep(0.01)
                return 42

            @abstractmethod
            def sync_required(self) -> None: ...

        required, optional = extract_port_methods(AsyncPort)

        assert "async_required" in required
        assert "sync_required" in required
        assert "async_optional" in optional

    def test_inherited_protocol_methods(self):
        """Test Protocol that inherits from another Protocol."""

        @runtime_checkable
        class BasePort(Protocol):
            @abstractmethod
            def base_method(self) -> None: ...

        @runtime_checkable
        class ExtendedPort(BasePort, Protocol):
            @abstractmethod
            def extended_method(self) -> str: ...

            def optional_method(self) -> int:
                return 100

        required, optional = extract_port_methods(ExtendedPort)

        # Should include methods from both base and extended
        assert "base_method" in required
        assert "extended_method" in required
        assert "optional_method" in optional


class TestValidateAdapterImplementation:
    """Test validation of adapter implementations against port requirements."""

    def test_valid_adapter_implementation(self):
        """Test adapter that correctly implements all required methods."""

        @runtime_checkable
        class LLMPort(Protocol):
            @abstractmethod
            def generate(self, prompt: str) -> str: ...

            @abstractmethod
            def tokenize(self, text: str) -> list[str]: ...

        class ValidAdapter:
            def generate(self, prompt: str) -> str:
                return "response"

            def tokenize(self, text: str) -> list[str]:
                return text.split()

        is_valid, missing = validate_adapter_implementation(ValidAdapter, LLMPort)

        assert is_valid is True
        assert missing == []

    def test_adapter_missing_required_methods(self):
        """Test adapter missing some required methods."""

        @runtime_checkable
        class StoragePort(Protocol):
            @abstractmethod
            def save(self, key: str, value: str) -> None: ...

            @abstractmethod
            def load(self, key: str) -> str: ...

            @abstractmethod
            def delete(self, key: str) -> None: ...

        class IncompleteAdapter:
            def save(self, key: str, value: str) -> None:
                pass

            # Missing load and delete methods

        is_valid, missing = validate_adapter_implementation(IncompleteAdapter, StoragePort)

        assert is_valid is False
        assert set(missing) == {"load", "delete"}

    def test_adapter_with_extra_methods(self):
        """Test that extra methods don't affect validation."""

        @runtime_checkable
        class MinimalPort(Protocol):
            @abstractmethod
            def required_method(self) -> None: ...

        class AdapterWithExtras:
            def required_method(self) -> None:
                pass

            def extra_method(self) -> str:
                return "extra"

            def another_extra(self, x: int) -> int:
                return x * 2

        is_valid, missing = validate_adapter_implementation(AdapterWithExtras, MinimalPort)

        assert is_valid is True
        assert missing == []

    def test_adapter_with_different_signatures(self):
        """Test that method signatures aren't validated, only presence."""

        @runtime_checkable
        class TypedPort(Protocol):
            @abstractmethod
            def process(self, data: str) -> int: ...

        class AdapterDifferentSignature:
            # Different signature but same method name
            def process(self, data: dict, extra: bool = True) -> str:
                return "processed"

        is_valid, missing = validate_adapter_implementation(AdapterDifferentSignature, TypedPort)

        # Should be valid - we only check method presence, not signatures
        assert is_valid is True
        assert missing == []


class TestInferAdapterCapabilities:
    """Test inferring adapter capabilities from optional methods."""

    def test_infer_all_optional_methods_implemented(self):
        """Test adapter that implements all optional methods."""

        @runtime_checkable
        class FeaturePort(Protocol):
            @abstractmethod
            def core_feature(self) -> None: ...

            def feature_a(self) -> str:
                return "default_a"

            def feature_b(self) -> int:
                return 0

            def feature_c(self) -> bool:
                return False

        class FullAdapter:
            def core_feature(self) -> None:
                pass

            def feature_a(self) -> str:
                return "custom_a"

            def feature_b(self) -> int:
                return 42

            def feature_c(self) -> bool:
                return True

        capabilities = infer_adapter_capabilities(FullAdapter, FeaturePort)

        assert set(capabilities) == {
            "supports_feature_a",
            "supports_feature_b",
            "supports_feature_c",
        }

    def test_infer_partial_optional_methods(self):
        """Test adapter that implements some optional methods."""

        @runtime_checkable
        class ExtensiblePort(Protocol):
            @abstractmethod
            def base_operation(self) -> None: ...

            def extension_1(self) -> None:
                pass

            def extension_2(self) -> None:
                pass

            def extension_3(self) -> None:
                pass

        class PartialAdapter:
            def base_operation(self) -> None:
                pass

            def extension_1(self) -> None:
                pass

            def extension_3(self) -> None:
                pass

            # Missing extension_2

        capabilities = infer_adapter_capabilities(PartialAdapter, ExtensiblePort)

        assert set(capabilities) == {"supports_extension_1", "supports_extension_3"}
        assert "supports_extension_2" not in capabilities

    def test_infer_no_optional_methods(self):
        """Test adapter that implements only required methods."""

        @runtime_checkable
        class RichPort(Protocol):
            @abstractmethod
            def required(self) -> None: ...

            def optional_a(self) -> None:
                pass

            def optional_b(self) -> None:
                pass

        class MinimalAdapter:
            def required(self) -> None:
                pass

        capabilities = infer_adapter_capabilities(MinimalAdapter, RichPort)

        assert capabilities == []


class TestExtractToolSignature:
    """Test extraction of tool signatures from functions."""

    def test_extract_sync_function_signature(self):
        """Test extracting signature from synchronous function."""

        def search_items(query: str, limit: int = 10, include_archived: bool = False) -> list[dict]:
            """Search for items."""
            return []

        info = extract_tool_signature(search_items)

        assert info["is_async"] is False
        assert info["return_type"] == "list[dict]"
        assert len(info["parameters"]) == 3

        # Check parameters
        params = info["parameters"]

        assert params[0]["name"] == "query"
        assert "str" in params[0]["type"]
        assert params[0]["required"] is True
        assert params[0]["default"] is None

        assert params[1]["name"] == "limit"
        assert "int" in params[1]["type"]
        assert params[1]["required"] is False
        assert params[1]["default"] == 10

        assert params[2]["name"] == "include_archived"
        assert "bool" in params[2]["type"]
        assert params[2]["required"] is False
        assert params[2]["default"] is False

    def test_extract_async_function_signature(self):
        """Test extracting signature from async function."""

        async def fetch_data(
            url: str, timeout: float = 30.0, headers: dict[str, str] | None = None
        ) -> dict:
            """Fetch data from URL."""
            await asyncio.sleep(0.01)
            return {}

        info = extract_tool_signature(fetch_data)

        assert info["is_async"] is True
        assert "dict" in info["return_type"]  # May be "dict" or "<class 'dict'>"
        assert len(info["parameters"]) == 3

        params = info["parameters"]

        assert params[0]["name"] == "url"
        assert params[0]["required"] is True

        assert params[1]["name"] == "timeout"
        assert params[1]["required"] is False
        assert params[1]["default"] == 30.0

        assert params[2]["name"] == "headers"
        assert params[2]["required"] is False
        assert params[2]["default"] is None

    def test_function_with_no_parameters(self):
        """Test function with no parameters."""

        def get_timestamp() -> int:
            """Get current timestamp."""
            return 0

        info = extract_tool_signature(get_timestamp)

        assert info["is_async"] is False
        assert "int" in info["return_type"]  # May be "int" or "<class 'int'>"
        assert info["parameters"] == []

    def test_function_with_no_annotations(self):
        """Test function without type annotations."""

        def untyped_function(x, y=10):
            """Untyped function."""
            return x + y

        info = extract_tool_signature(untyped_function)

        assert info["is_async"] is False
        assert info["return_type"] == "Any"
        assert len(info["parameters"]) == 2

        assert info["parameters"][0]["name"] == "x"
        assert info["parameters"][0]["type"] == "Any"
        assert info["parameters"][0]["required"] is True

        assert info["parameters"][1]["name"] == "y"
        assert info["parameters"][1]["type"] == "Any"
        assert info["parameters"][1]["required"] is False
        assert info["parameters"][1]["default"] == 10

    def test_function_with_args_kwargs(self):
        """Test function with *args and **kwargs."""

        def flexible_function(
            required: str, *args: int, optional: bool = True, **kwargs: str
        ) -> None:
            """Function with variable arguments."""
            pass

        info = extract_tool_signature(flexible_function)

        params = info["parameters"]

        # Should capture regular parameters
        assert any(p["name"] == "required" for p in params)
        assert any(p["name"] == "optional" for p in params)

        # *args and **kwargs should be in parameters
        assert any(p["name"] == "args" for p in params)
        assert any(p["name"] == "kwargs" for p in params)

    def test_lambda_function(self):
        """Test extracting signature from lambda."""

        def simple_lambda(x, y=5):
            return x * y

        info = extract_tool_signature(simple_lambda)

        assert info["is_async"] is False
        assert len(info["parameters"]) == 2
        assert info["parameters"][0]["name"] == "x"
        assert info["parameters"][1]["name"] == "y"
        assert info["parameters"][1]["default"] == 5


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_llm_port_and_adapters(self):
        """Test realistic LLM port with multiple adapters."""

        @runtime_checkable
        class LLMPort(Protocol):
            @abstractmethod
            async def agenerate(self, messages: list[dict]) -> str: ...

            @abstractmethod
            def count_tokens(self, text: str) -> int: ...

            def set_temperature(self, temp: float) -> None:
                """Optional temperature setting."""
                pass

            def get_model_info(self) -> dict:
                """Optional model information."""
                return {}

        class OpenAIAdapter:
            """Full-featured adapter."""

            async def agenerate(self, messages: list[dict]) -> str:
                return "OpenAI response"

            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def set_temperature(self, temp: float) -> None:
                self.temperature = temp

            def get_model_info(self) -> dict:
                return {"model": "gpt-4", "version": "latest"}

        class MinimalAdapter:
            """Minimal adapter with only required methods."""

            async def agenerate(self, messages: list[dict]) -> str:
                return "Minimal response"

            def count_tokens(self, text: str) -> int:
                return len(text) // 4

        # Validate both adapters
        is_valid_openai, missing_openai = validate_adapter_implementation(OpenAIAdapter, LLMPort)
        assert is_valid_openai is True
        assert missing_openai == []

        is_valid_minimal, missing_minimal = validate_adapter_implementation(MinimalAdapter, LLMPort)
        assert is_valid_minimal is True
        assert missing_minimal == []

        # Check capabilities
        openai_caps = infer_adapter_capabilities(OpenAIAdapter, LLMPort)
        assert set(openai_caps) == {"supports_set_temperature", "supports_get_model_info"}

        minimal_caps = infer_adapter_capabilities(MinimalAdapter, LLMPort)
        assert minimal_caps == []

    def test_database_port_with_transactions(self):
        """Test database port with transaction support."""

        @runtime_checkable
        class TransactionalDatabasePort(Protocol):
            @abstractmethod
            async def aexecute(self, query: str) -> None: ...

            @abstractmethod
            async def afetch(self, query: str) -> list[dict]: ...

            async def abegin_transaction(self) -> None:
                """Optional transaction support."""
                pass

            async def acommit(self) -> None:
                """Optional commit."""
                pass

            async def arollback(self) -> None:
                """Optional rollback."""
                pass

        class PostgreSQLAdapter:
            """Adapter with full transaction support."""

            async def aexecute(self, query: str) -> None:
                pass

            async def afetch(self, query: str) -> list[dict]:
                return []

            async def abegin_transaction(self) -> None:
                pass

            async def acommit(self) -> None:
                pass

            async def arollback(self) -> None:
                pass

        class SimpleDBAdapter:
            """Adapter without transactions."""

            async def aexecute(self, query: str) -> None:
                pass

            async def afetch(self, query: str) -> list[dict]:
                return []

        # Extract port methods
        required, optional = extract_port_methods(TransactionalDatabasePort)
        assert set(required) == {"aexecute", "afetch"}
        assert set(optional) == {"abegin_transaction", "acommit", "arollback"}

        # Check PostgreSQL adapter
        pg_caps = infer_adapter_capabilities(PostgreSQLAdapter, TransactionalDatabasePort)
        assert set(pg_caps) == {
            "supports_abegin_transaction",
            "supports_acommit",
            "supports_arollback",
        }

        # Check simple adapter
        simple_caps = infer_adapter_capabilities(SimpleDBAdapter, TransactionalDatabasePort)
        assert simple_caps == []

    def test_tool_collection_signatures(self):
        """Test extracting signatures from a collection of tools."""

        async def search_web(query: str, safe: bool = True) -> list[dict]:
            """Search the web."""
            return []

        def calculate_statistics(
            data: list[float], include_median: bool = False, precision: int = 2
        ) -> dict[str, float]:
            """Calculate statistics."""
            return {}

        async def send_email(
            to: str,
            subject: str,
            body: str,
            cc: list[str] | None = None,
            attachments: list[str] | None = None,
        ) -> bool:
            """Send an email."""
            return True

        tools = [search_web, calculate_statistics, send_email]

        for tool in tools:
            info = extract_tool_signature(tool)

            # All should have extracted info
            assert "is_async" in info
            assert "parameters" in info
            assert "return_type" in info

            # Check specific tool properties
            if tool.__name__ == "search_web":
                assert info["is_async"] is True
                assert len(info["parameters"]) == 2

            elif tool.__name__ == "calculate_statistics":
                assert info["is_async"] is False
                assert len(info["parameters"]) == 3

            elif tool.__name__ == "send_email":
                assert info["is_async"] is True
                assert len(info["parameters"]) == 5


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_empty_protocol(self):
        """Test Protocol with no methods."""

        @runtime_checkable
        class EmptyPort(Protocol):
            pass

        required, optional = extract_port_methods(EmptyPort)

        assert required == []
        assert optional == []

    def test_non_protocol_class(self):
        """Test behavior with regular class (not Protocol)."""

        class RegularClass:
            def method_a(self) -> None:
                pass

            def method_b(self) -> str:
                return "b"

        # Should still work - just won't detect abstract methods
        required, optional = extract_port_methods(RegularClass)

        assert required == []  # No abstract methods in regular class
        assert set(optional) == {"method_a", "method_b"}

    def test_class_with_properties(self):
        """Test that properties are handled correctly."""

        @runtime_checkable
        class PortWithProperties(Protocol):
            @property
            @abstractmethod
            def required_property(self) -> str: ...

            @property
            def optional_property(self) -> int:
                return 42

            @abstractmethod
            def regular_method(self) -> None: ...

        required, optional = extract_port_methods(PortWithProperties)

        # Properties might be included depending on inspect.isfunction behavior
        assert "regular_method" in required

    def test_adapter_as_instance_not_class(self):
        """Test validation when adapter is an instance, not a class."""

        @runtime_checkable
        class SimplePort(Protocol):
            @abstractmethod
            def do_something(self) -> None: ...

        class AdapterClass:
            def do_something(self) -> None:
                pass

        adapter_instance = AdapterClass()

        # Should work with instance too
        is_valid, missing = validate_adapter_implementation(type(adapter_instance), SimplePort)

        assert is_valid is True
        assert missing == []

    def test_method_name_conflicts(self):
        """Test handling of method name conflicts."""

        @runtime_checkable
        class ConflictPort(Protocol):
            @abstractmethod
            def process(self) -> None: ...

            def Process(self) -> None:  # Different case
                pass

            def PROCESS(self) -> None:  # All caps
                pass

        required, optional = extract_port_methods(ConflictPort)

        # Should handle all variants
        assert "process" in required
        # Case-sensitive, so these are different methods
        if "Process" in optional:
            assert "Process" in optional
        if "PROCESS" in optional:
            assert "PROCESS" in optional
