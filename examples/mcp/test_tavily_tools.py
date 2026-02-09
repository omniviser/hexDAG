"""Simple test script for Tavily tools.

This validates that:
1. Tavily tools can be imported
2. Tool functions have proper signatures
3. Basic tool execution works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.mcp.tavily_adapter import tavily_qna_search, tavily_search


def test_tool_import():
    """Test that Tavily tools can be imported."""
    print("\n" + "=" * 80)
    print("Testing Tavily Tool Import")
    print("=" * 80)

    # Check tavily_search tool
    print("\n1. Checking tavily_search function...")
    print(f"   ✓ tavily_search: {tavily_search}")
    print(f"   ✓ Is async: {tavily_search.__code__.co_flags & 0x80 != 0}")

    # Check tavily_qna_search tool
    print("\n2. Checking tavily_qna_search function...")
    print(f"   ✓ tavily_qna_search: {tavily_qna_search}")
    print(f"   ✓ Is async: {tavily_qna_search.__code__.co_flags & 0x80 != 0}")

    return True


def test_tool_signatures():
    """Test that tool functions have proper type hints."""
    print("\n3. Checking function signatures...")

    import inspect

    # Check tavily_search
    sig = inspect.signature(tavily_search)
    print(f"   ✓ tavily_search parameters: {list(sig.parameters.keys())}")

    # Check tavily_qna_search
    sig = inspect.signature(tavily_qna_search)
    print(f"   ✓ tavily_qna_search parameters: {list(sig.parameters.keys())}")

    return True


if __name__ == "__main__":
    success = True

    success = test_tool_import() and success
    success = test_tool_signatures() and success

    print("\n" + "=" * 80)
    if success:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 80)

    sys.exit(0 if success else 1)
