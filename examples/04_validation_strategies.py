"""
Example 04: Validation Strategies

This example demonstrates the different validation strategies available in hexAI:
- STRICT: No type conversion, fails on any mismatch
- COERCE: Attempts type conversion when possible
- PASSTHROUGH: No validation, passes data through as-is
"""

import asyncio
from typing import Any

from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph, NodeSpec
from hexai.validation import coerce_validator, passthrough_validator, strict_validator


async def process_strict(input_data: str) -> dict:
    """Process data with strict validation."""
    return {
        "input_type": type(input_data).__name__,
        "value": input_data,
        "strategy": "strict",
        "status": "processed",
    }


async def process_coerce(input_data: Any) -> dict:
    """Process data with coercion validation."""
    return {
        "input_type": type(input_data).__name__,
        "value": input_data,
        "strategy": "coerce",
        "status": "processed",
    }


async def process_passthrough(input_data: Any) -> dict:
    """Process data with passthrough validation."""
    return {
        "input_type": type(input_data).__name__,
        "value": input_data,
        "strategy": "passthrough",
        "status": "processed",
    }


async def compare_results(
    strict_result: dict, coerce_result: dict, passthrough_result: dict
) -> dict:
    """Compare results from different validation strategies."""
    return {
        "comparison": {
            "strict": {
                "type": strict_result.get("input_type"),
                "value": strict_result.get("value"),
                "status": strict_result.get("status"),
            },
            "coerce": {
                "type": coerce_result.get("input_type"),
                "value": coerce_result.get("value"),
                "status": coerce_result.get("status"),
            },
            "passthrough": {
                "type": passthrough_result.get("input_type"),
                "value": passthrough_result.get("value"),
                "status": passthrough_result.get("status"),
            },
        },
        "analysis": "Validation strategy comparison complete",
    }


async def main():
    """Demonstrate different validation strategies."""

    print("🔍 Example 04: Validation Strategies")
    print("=" * 40)

    print("\n🎯 This example demonstrates:")
    print("   • STRICT validation - No type conversion")
    print("   • COERCE validation - Attempts type conversion")
    print("   • PASSTHROUGH validation - No validation")
    print("   • Strategy comparison and analysis")

    # Test data
    test_cases = [
        ("string_number", "42"),
        ("integer", 42),
        ("float_number", 42.0),
        ("boolean_string", "true"),
        ("complex_data", {"nested": "value"}),
    ]

    for test_name, test_data in test_cases:
        print(f"\n🧪 Test Case: {test_name} = {test_data} (type: {type(test_data).__name__})")
        print("-" * 60)

        # Create orchestrators with different strategies
        strict_orchestrator = Orchestrator(validator=strict_validator())
        coerce_orchestrator = Orchestrator(validator=coerce_validator())
        passthrough_orchestrator = Orchestrator(validator=passthrough_validator())

        # Create graphs for each strategy
        strict_graph = DirectedGraph()
        coerce_graph = DirectedGraph()
        passthrough_graph = DirectedGraph()

        # Add nodes
        strict_node = NodeSpec("process", process_strict, in_type=str)
        coerce_node = NodeSpec("process", process_coerce, in_type=int)
        passthrough_node = NodeSpec("process", process_passthrough)

        strict_graph.add(strict_node)
        coerce_graph.add(coerce_node)
        passthrough_graph.add(passthrough_node)

        # Execute with different strategies
        results = {}

        # STRICT Strategy
        try:
            strict_results = await strict_orchestrator.run(strict_graph, test_data)
            results["strict"] = strict_results["process"]
            print("   ✅ STRICT: Success")
        except Exception as e:
            results["strict"] = {"error": str(e)}
            print(f"   ❌ STRICT: Failed - {e}")

        # COERCE Strategy
        try:
            coerce_results = await coerce_orchestrator.run(coerce_graph, test_data)
            results["coerce"] = coerce_results["process"]
            print("   ✅ COERCE: Success")
        except Exception as e:
            results["coerce"] = {"error": str(e)}
            print(f"   ❌ COERCE: Failed - {e}")

        # PASSTHROUGH Strategy
        try:
            passthrough_results = await passthrough_orchestrator.run(passthrough_graph, test_data)
            results["passthrough"] = passthrough_results["process"]
            print("   ✅ PASSTHROUGH: Success")
        except Exception as e:
            results["passthrough"] = {"error": str(e)}
            print(f"   ❌ PASSTHROUGH: Failed - {e}")

        # Show detailed results
        print(f"\n📊 Results for '{test_name}':")
        for strategy, result in results.items():
            if "error" in result:
                print(f"   {strategy.upper()}: ❌ {result['error']}")
            else:
                print(f"   {strategy.upper()}: {result['value']} ({result['input_type']})")

    # Demonstrate strategy comparison
    print(f"\n🔬 Strategy Comparison Analysis:")
    print("   • STRICT: Best for type safety, fails on any mismatch")
    print("   • COERCE: Best for flexibility, converts when possible")
    print("   • PASSTHROUGH: Best for performance, no validation overhead")

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Validation Strategies - Choose based on your needs")
    print("   ✅ Type Safety - STRICT ensures exact type matching")
    print("   ✅ Flexibility - COERCE handles type conversions")
    print("   ✅ Performance - PASSTHROUGH skips validation")
    print("   ✅ Error Handling - Different strategies fail differently")

    print(f"\n🔗 Next: Run example 05 to learn about the event system!")


if __name__ == "__main__":
    asyncio.run(main())
