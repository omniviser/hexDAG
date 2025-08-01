#!/usr/bin/env python3
"""
⚠️ Example 16: Common Validation Errors and Solutions.

This example teaches:
- Common validation failure patterns
- How to debug validation errors
- Validation strategy selection
- Error recovery patterns

Run: python examples/16_validation_errors.py
"""

import asyncio
from typing import Any

from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph, NodeSpec
from hexai.validation import coerce_validator, passthrough_validator, strict_validator
from pydantic import BaseModel, Field, ValidationError


class StrictUserModel(BaseModel):
    """Strict user model with validation rules."""

    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    score: float = Field(..., ge=0.0, le=1.0)


class FlexibleUserModel(BaseModel):
    """More flexible user model."""

    name: str = Field(default="Unknown", min_length=1)
    age: int = Field(default=0, ge=0, le=200)
    email: str = Field(default="noemail@example.com")
    score: float = Field(default=0.0, ge=0.0, le=1.0)


async def strict_processor(input_data: Any) -> StrictUserModel:
    """Process with strict validation."""
    if isinstance(input_data, dict):
        return StrictUserModel(**input_data)
    else:
        raise ValueError(f"Expected dict, got {type(input_data)}")


async def flexible_processor(input_data: Any) -> FlexibleUserModel:
    """Process with flexible validation."""
    if isinstance(input_data, dict):
        return FlexibleUserModel(**input_data)
    elif isinstance(input_data, str):
        # Try to parse string input
        parts = input_data.split(",")
        if len(parts) >= 2:
            return FlexibleUserModel(
                name=parts[0].strip(),
                age=int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                email=parts[2].strip() if len(parts) > 2 else "noemail@example.com",
                score=(
                    float(parts[3].strip())
                    if len(parts) > 3 and parts[3].replace(".", "").isdigit()
                    else 0.0
                ),
            )
        else:
            return FlexibleUserModel(name=input_data)
    else:
        return FlexibleUserModel(name=str(input_data))


async def display_user_info(user: StrictUserModel) -> dict:
    """Display user information."""
    return {
        "display_name": f"{user.name} (Age: {user.age})",
        "contact": user.email,
        "rating": f"{user.score:.2%}",
        "category": "adult" if user.age >= 18 else "minor",
    }


async def demonstrate_type_mismatch_errors():
    """Show type mismatch validation errors."""

    print("\n🔴 Error Demo 1: Type Mismatch")
    print("-" * 40)

    graph = DirectedGraph()
    graph.add(NodeSpec("process", strict_processor))
    graph.add(NodeSpec("display", display_user_info).after("process"))

    # Test with different validation strategies
    test_cases = [
        {
            "name": "John Doe",
            "age": "25",  # String instead of int
            "email": "john@example.com",
            "score": "0.85",  # String instead of float
        },
        {"name": "Jane", "age": 30, "email": "invalid-email", "score": 0.9},  # Invalid email format
        {
            "name": "",  # Too short name
            "age": 25,
            "email": "test@example.com",
            "score": 1.5,  # Score too high
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test 1.{i}: {test_case}")

        # Try with strict validator
        orchestrator_strict = Orchestrator(validator=strict_validator())
        try:
            result = await orchestrator_strict.run(graph, test_case)
            print("   ✅ Strict validation passed (unexpected)")
        except Exception as e:
            print(f"   ❌ Strict validation failed: {type(e).__name__}")
            print(f"      Details: {str(e)[:80]}...")

        # Try with coerce validator
        orchestrator_coerce = Orchestrator(validator=coerce_validator())
        try:
            result = await orchestrator_coerce.run(graph, test_case)
            print("   🔄 Coerce validation passed")
            print(f"      Result: {result['display']['display_name']}")
        except Exception as e:
            print(f"   ❌ Coerce validation also failed: {type(e).__name__}")
            print(f"      Details: {str(e)[:80]}...")


async def demonstrate_schema_compatibility_errors():
    """Show schema compatibility issues between nodes."""

    print("\n🔴 Error Demo 2: Schema Compatibility")
    print("-" * 40)

    # This will create a compatibility issue
    async def return_wrong_type(input_data: Any) -> dict:
        """Return incompatible type."""
        return {"wrong_field": "not what display_user_info expects"}

    graph = DirectedGraph()
    graph.add(NodeSpec("wrong_processor", return_wrong_type))
    graph.add(NodeSpec("display", display_user_info).after("wrong_processor"))

    orchestrator = Orchestrator(validator=strict_validator())

    print("\n   Testing incompatible schema connection...")
    try:
        result = await orchestrator.run(
            graph, {"name": "Test", "age": 25, "email": "test@example.com", "score": 0.5}
        )
        print("   ✅ Unexpectedly succeeded")
    except Exception as e:
        print(f"   ❌ Schema compatibility error: {type(e).__name__}")
        print(f"      Details: {str(e)[:100]}...")
        print("   💡 Solution: Ensure output schema matches next node's input schema")


async def demonstrate_missing_dependencies():
    """Show missing dependency errors."""

    print("\n🔴 Error Demo 3: Missing Dependencies")
    print("-" * 40)

    # Create a node that depends on non-existent node
    graph = DirectedGraph()
    graph.add(NodeSpec("processor", strict_processor))

    try:
        # This should fail because "nonexistent" doesn't exist
        graph.add(NodeSpec("display", display_user_info).after("processor", "nonexistent_node"))
        print("   ⚠️  Graph creation succeeded (validation may be lazy)")

        # Error should appear during validation
        graph.validate()
        print("   ✅ Validation passed (unexpected)")
    except Exception as e:
        print(f"   ❌ Missing dependency error: {type(e).__name__}")
        print(f"      Details: {str(e)}")
        print("   💡 Solution: Ensure all dependencies exist before adding nodes")


async def demonstrate_circular_dependencies():
    """Show circular dependency detection."""

    print("\n🔴 Error Demo 4: Circular Dependencies")
    print("-" * 40)

    async def node_a(input_data: Any) -> dict:
        return {"from": "node_a", "data": input_data}

    async def node_b(input_data: dict) -> dict:
        return {"from": "node_b", "data": input_data}

    async def node_c(input_data: dict) -> dict:
        return {"from": "node_c", "data": input_data}

    try:
        graph = DirectedGraph()
        graph.add(NodeSpec("a", node_a))
        graph.add(NodeSpec("b", node_b).after("a"))
        graph.add(NodeSpec("c", node_c).after("b"))

        # Create circular dependency: a -> b -> c -> a
        graph.add(NodeSpec("a_circular", node_a).after("c"))  # This creates a cycle

        print("   Graph created, validating...")
        graph.validate()
        print("   ✅ Validation passed (unexpected)")

    except Exception as e:
        print(f"   ❌ Circular dependency detected: {type(e).__name__}")
        print(f"      Details: {str(e)}")
        print("   💡 Solution: Review dependency chain to eliminate cycles")


async def demonstrate_error_recovery_patterns():
    """Show patterns for recovering from validation errors."""

    print("\n🟢 Recovery Demo: Error Recovery Patterns")
    print("-" * 40)

    # Recovery pattern 1: Use flexible models
    print("\n   Pattern 1: Flexible Input Models")

    graph_flexible = DirectedGraph()
    graph_flexible.add(NodeSpec("flexible_process", flexible_processor))
    graph_flexible.add(NodeSpec("display", display_user_info).after("flexible_process"))

    # This problematic input should work with flexible processing
    problematic_input = {
        "name": "",  # Empty name
        "age": "unknown",  # Invalid age
        "email": "not-an-email",
        "score": "high",  # Invalid score
    }

    orchestrator = Orchestrator(validator=coerce_validator())
    try:
        result = await orchestrator.run(graph_flexible, problematic_input)
        print("   ✅ Flexible processing succeeded")
        print(f"      Result: {result['display']['display_name']}")
    except Exception as e:
        print(f"   ❌ Even flexible processing failed: {e}")

    # Recovery pattern 2: String input parsing
    print("\n   Pattern 2: String Input Parsing")
    string_input = "Alice Johnson, 28, alice@company.com, 0.92"
    try:
        result = await orchestrator.run(graph_flexible, string_input)
        print("   ✅ String parsing succeeded")
        print(f"      Parsed: {result['display']['display_name']}")
    except Exception as e:
        print(f"   ❌ String parsing failed: {e}")

    # Recovery pattern 3: Passthrough validation
    print("\n   Pattern 3: Passthrough Validation")
    orchestrator_passthrough = Orchestrator(validator=passthrough_validator())
    try:
        result = await orchestrator_passthrough.run(graph_flexible, problematic_input)
        print("   ✅ Passthrough validation succeeded")
        print(f"      Result: {result['display']['display_name']}")
    except Exception as e:
        print(f"   ❌ Passthrough validation failed: {e}")


async def main():
    """Demonstrate common validation errors and solutions."""

    print("⚠️ Example 16: Common Validation Errors and Solutions")
    print("=" * 60)

    print("\n🎯 This example demonstrates:")
    print("   • Type mismatch errors and fixes")
    print("   • Schema compatibility issues")
    print("   • Missing dependency detection")
    print("   • Circular dependency problems")
    print("   • Error recovery patterns")

    await demonstrate_type_mismatch_errors()
    await demonstrate_schema_compatibility_errors()
    await demonstrate_missing_dependencies()
    await demonstrate_circular_dependencies()
    await demonstrate_error_recovery_patterns()

    print("\n🎯 Key Lessons Learned:")
    print("   ✅ Validation Strategy Selection - Choose strict/coerce/passthrough wisely")
    print("   ✅ Schema Design - Use flexible models for problematic inputs")
    print("   ✅ Error Messages - Read validation errors carefully for clues")
    print("   ✅ Recovery Patterns - Plan fallback strategies")
    print("   ✅ Graph Validation - Validate early to catch structural issues")

    print("\n💡 Best Practices:")
    print("   • Start with coerce validation for flexibility")
    print("   • Use strict validation for critical data paths")
    print("   • Design flexible input models for user-facing nodes")
    print("   • Validate graphs immediately after construction")
    print("   • Implement graceful fallbacks for external data")

    print("\n🔗 Next: Run example 17 to learn about circular dependency detection!")


if __name__ == "__main__":
    asyncio.run(main())
