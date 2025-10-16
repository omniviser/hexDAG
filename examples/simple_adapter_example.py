"""Example: Simple adapter with automatic secret resolution.

This example shows the new simplified pattern - no Config classes needed!
Secrets are declared in the @adapter decorator for explicit, clean configuration.
"""

import os

from hexdag.core.registry import adapter, policy


# Example 1: Adapter with secret (DECORATOR-BASED - PREFERRED)
@adapter("llm", name="simple_openai", secrets={"api_key": "OPENAI_API_KEY"})
class SimpleOpenAIAdapter:
    """OpenAI adapter - no Config class needed!

    Secrets are declared in the @adapter decorator.
    The decorator automatically resolves them from:
    1. Environment variables (OPENAI_API_KEY)
    2. Memory port (if provided)
    3. Explicit kwargs (highest priority)
    """

    def __init__(
        self,
        api_key: str,  # ← Clean signature! Secret defined in decorator
        model: str = "gpt-4",
        temperature: float = 0.7,
    ):
        """Initialize OpenAI adapter.

        Parameters
        ----------
        api_key : str
            API key (auto-resolved from OPENAI_API_KEY env var by decorator)
        model : str, default="gpt-4"
            Model to use
        temperature : float, default=0.7
            Sampling temperature
        """
        self.api_key = api_key  # Already resolved by decorator!
        self.model = model
        self.temperature = temperature
        print(f"✅ Initialized with model={model}, temp={temperature}")
        print(f"✅ API key resolved: {api_key[:10]}..." if api_key else "❌ No API key")

    async def aresponse(self, messages):
        """Generate response (mock implementation)."""
        return f"Mock response using {self.model}"


# Example 2: Adapter without secrets
@adapter("cache", name="simple_cache")
class SimpleCacheAdapter:
    """Cache adapter - no secrets, no Config class!"""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize cache.

        Parameters
        ----------
        max_size : int, default=100
            Maximum cache size
        ttl : int, default=3600
            Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        print(f"✅ Cache initialized: max_size={max_size}, ttl={ttl}")

    async def aget(self, key: str):
        """Get value from cache."""
        return

    async def aset(self, key: str, value):
        """Set value in cache."""
        pass


@policy(name="simple_retry")
class SimpleRetryPolicy:
    """Retry policy - no ConfigurablePolicy needed!"""

    def __init__(self, max_retries: int = 3):
        """Initialize retry policy.

        Parameters
        ----------
        max_retries : int, default=3
            Maximum number of retries
        """
        self.max_retries = max_retries

    async def evaluate(self, context):
        """Evaluate retry logic."""
        from hexdag.core.policies.models import PolicyResponse, PolicySignal

        if context.error and context.attempt <= self.max_retries:
            return PolicyResponse(signal=PolicySignal.RETRY)
        return PolicyResponse(signal=PolicySignal.PROCEED)


if __name__ == "__main__":
    # Test 1: With environment variable
    os.environ["OPENAI_API_KEY"] = "sk-test-1234567890abcdef"

    print("=" * 60)
    print("Test 1: Creating adapter with env var")
    print("=" * 60)
    adapter1 = SimpleOpenAIAdapter()

    print("\n" + "=" * 60)
    print("Test 2: Creating adapter with custom params")
    print("=" * 60)
    adapter2 = SimpleOpenAIAdapter(model="gpt-3.5-turbo", temperature=0.5)

    print("\n" + "=" * 60)
    print("Test 3: Creating adapter without secrets")
    print("=" * 60)
    cache = SimpleCacheAdapter(max_size=200)

    print("\n" + "=" * 60)
    print("Test 4: Creating policy")
    print("=" * 60)
    policy = SimpleRetryPolicy(max_retries=5)
    print(f"✅ Policy created: max_retries={policy.max_retries}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. No Config classes needed!")
    print("2. Secrets declared in @adapter decorator - explicit & clean")
    print("3. Auto-resolved from environment by decorator")
    print("4. __init__ signatures stay clean and simple")
    print("5. CLI can introspect decorator metadata for schemas")
    print("6. Backward compatible with secret() helper pattern")
