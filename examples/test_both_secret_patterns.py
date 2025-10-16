"""Test both secret resolution patterns: decorator-based and signature-based."""

import os

from hexdag.core.registry import adapter
from hexdag.core.secrets import secret

# Set environment variables for testing
os.environ["TEST_API_KEY"] = "sk-decorator-test-key"
os.environ["TEST_API_KEY_2"] = "sk-signature-test-key"


# Pattern 1: Decorator-based (PREFERRED)
@adapter("llm", name="decorator_adapter", secrets={"api_key": "TEST_API_KEY"})
class DecoratorAdapter:
    """Adapter using decorator-based secret declaration."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model


# Pattern 2: Signature-based (BACKWARD COMPATIBLE)
@adapter("llm", name="signature_adapter")
class SignatureAdapter:
    """Adapter using signature-based secret declaration."""

    def __init__(self, api_key: str = secret(env="TEST_API_KEY_2"), model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model


# Pattern 3: Multiple secrets in decorator
@adapter(
    "database", name="multi_secret_adapter", secrets={"username": "DB_USER", "password": "DB_PASS"}
)
class MultiSecretAdapter:
    """Adapter with multiple secrets."""

    def __init__(self, username: str, password: str, host: str = "localhost"):
        self.username = username
        self.password = password
        self.host = host


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Secret Resolution Patterns")
    print("=" * 60)

    # Test 1: Decorator-based resolution
    print("\n[Test 1] Decorator-based secret resolution")
    adapter1 = DecoratorAdapter()
    assert adapter1.api_key == "sk-decorator-test-key", "Decorator secret resolution failed"
    assert adapter1.model == "gpt-4", "Default parameter failed"
    print("✅ Decorator-based: api_key resolved from TEST_API_KEY")
    print(f"   api_key = {adapter1.api_key[:15]}...")

    # Test 2: Signature-based resolution (backward compatibility)
    print("\n[Test 2] Signature-based secret resolution (backward compatible)")
    adapter2 = SignatureAdapter()
    assert adapter2.api_key == "sk-signature-test-key", "Signature secret resolution failed"
    assert adapter2.model == "gpt-4", "Default parameter failed"
    print("✅ Signature-based: api_key resolved from TEST_API_KEY_2")
    print(f"   api_key = {adapter2.api_key[:15]}...")

    # Test 3: Explicit parameter (highest priority)
    print("\n[Test 3] Explicit parameter overrides environment")
    adapter3 = DecoratorAdapter(api_key="explicit-key")
    assert adapter3.api_key == "explicit-key", "Explicit override failed"
    print("✅ Explicit parameter: api_key = 'explicit-key'")

    # Test 4: Multiple secrets
    print("\n[Test 4] Multiple secrets in decorator")
    os.environ["DB_USER"] = "admin"
    os.environ["DB_PASS"] = "secret123"
    adapter4 = MultiSecretAdapter()
    assert adapter4.username == "admin", "Username secret failed"
    assert adapter4.password == "secret123", "Password secret failed"
    assert adapter4.host == "localhost", "Default host failed"
    print("✅ Multiple secrets: username and password resolved")
    print(f"   username = {adapter4.username}")
    print("   password = ***")

    # Test 5: Decorator metadata introspection
    print("\n[Test 5] Decorator metadata introspection")
    assert hasattr(DecoratorAdapter, "_hexdag_secrets"), "Missing _hexdag_secrets attribute"
    assert DecoratorAdapter._hexdag_secrets == {"api_key": "TEST_API_KEY"}
    print("✅ Decorator metadata available for CLI introspection")
    print(f"   _hexdag_secrets = {DecoratorAdapter._hexdag_secrets}")

    # Test 6: Missing required secret (should raise error)
    print("\n[Test 6] Missing required secret raises error")
    os.environ.pop("TEST_API_KEY", None)  # Remove env var
    try:
        adapter5 = DecoratorAdapter()
        print("❌ Should have raised ValueError for missing secret")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Required secret 'api_key' not found" in str(e)
        print("✅ ValueError raised for missing required secret")
        print(f"   Error: {e}")
    finally:
        os.environ["TEST_API_KEY"] = "sk-decorator-test-key"  # Restore

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nConclusions:")
    print("1. Decorator-based pattern works perfectly")
    print("2. Signature-based pattern still works (backward compatible)")
    print("3. Explicit parameters have highest priority")
    print("4. Multiple secrets are supported")
    print("5. CLI can introspect decorator metadata")
    print("6. Missing required secrets raise clear errors")
