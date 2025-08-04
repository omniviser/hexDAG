#!/usr/bin/env python3
"""
✅ Example 03: Validation Basics.

This example teaches:
- Input/output type validation
- Pydantic model validation
- Type conversion and coercion
- Validation error handling

Run: python examples/03_validation_basics.py
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.validation import coerce_validator, strict_validator


class UserInput(BaseModel):
    """Input model for user data."""

    name: str = Field(..., description="User's name")
    age: int = Field(..., ge=0, le=150, description="User's age")
    email: str = Field(..., description="User's email address")


class ProcessedUser(BaseModel):
    """Output model for processed user data."""

    name: str
    age: int
    email: str
    category: str
    is_adult: bool
    processing_notes: list[str]


async def validate_user_input(input_data: Any) -> UserInput:
    """Validate and convert raw input to UserInput model."""
    if isinstance(input_data, dict):
        return UserInput(**input_data)
    else:
        # Handle string input by parsing
        parts = str(input_data).split(",")
        if len(parts) >= 3:
            return UserInput(
                name=parts[0].strip(), age=int(parts[1].strip()), email=parts[2].strip()
            )
        else:
            raise ValueError(f"Invalid input format: {input_data}")


async def process_user_data(user_input: UserInput) -> ProcessedUser:
    """Process validated user data into enriched output."""

    # Determine category based on age
    if user_input.age < 18:
        category = "minor"
    elif user_input.age < 65:
        category = "adult"
    else:
        category = "senior"

    # Add processing notes
    notes = []
    if "@" not in user_input.email:
        notes.append("Email format may be invalid")
    if user_input.age > 100:
        notes.append("Age seems unusually high")
    if len(user_input.name) < 2:
        notes.append("Name seems unusually short")

    return ProcessedUser(
        name=user_input.name,
        age=user_input.age,
        email=user_input.email,
        category=category,
        is_adult=user_input.age >= 18,
        processing_notes=notes,
    )


async def format_output(processed_user: ProcessedUser) -> dict:
    """Format the processed user into final output."""
    return {
        "user_summary": f"{processed_user.name} ({processed_user.age})",
        "classification": processed_user.category,
        "adult_status": "adult" if processed_user.is_adult else "minor",
        "email": processed_user.email,
        "warnings": processed_user.processing_notes,
        "validation_passed": True,
    }


async def demonstrate_validation_success():
    """Show successful validation with proper data."""

    print("\n🟢 Test 1: Valid Data (Dictionary Input)")
    print("-" * 40)

    graph = DirectedGraph()
    graph.add(NodeSpec("validate", validate_user_input))
    graph.add(NodeSpec("process", process_user_data).after("validate"))
    graph.add(NodeSpec("format", format_output).after("process"))

    orchestrator = Orchestrator(validator=coerce_validator())

    # Valid dictionary input
    valid_input = {"name": "Alice Johnson", "age": 28, "email": "alice@example.com"}

    try:
        results = await orchestrator.run(graph, valid_input)
        print("   ✅ Validation successful!")
        print(f"   📊 Result: {results['format']['user_summary']}")
        print(f"   🏷️  Category: {results['format']['classification']}")
        if results["format"]["warnings"]:
            print(f"   ⚠️  Warnings: {results['format']['warnings']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def demonstrate_type_coercion():
    """Show type coercion in action."""

    print("\n🔄 Test 2: Type Coercion (String Numbers)")
    print("-" * 40)

    graph = DirectedGraph()
    graph.add(NodeSpec("validate", validate_user_input))
    graph.add(NodeSpec("process", process_user_data).after("validate"))
    graph.add(NodeSpec("format", format_output).after("process"))

    orchestrator = Orchestrator(validator=coerce_validator())

    # Input with string age (will be coerced to int)
    coercion_input = {
        "name": "Bob Smith",
        "age": "45",  # String that should become int
        "email": "bob@company.com",
    }

    try:
        results = await orchestrator.run(graph, coercion_input)
        print("   ✅ Type coercion successful!")
        print(f"   📊 Age converted: '45' → {results['process'].age}")
        print(f"   🏷️  Category: {results['format']['classification']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def demonstrate_validation_errors():
    """Show what happens with invalid data."""

    print("\n� Test 3: Validation Errors")
    print("-" * 40)

    graph = DirectedGraph()
    graph.add(NodeSpec("validate", validate_user_input))
    graph.add(NodeSpec("process", process_user_data).after("validate"))

    orchestrator = Orchestrator(validator=strict_validator())

    # Invalid data that should fail validation
    invalid_inputs = [
        {"name": "", "age": -5, "email": "invalid"},  # Invalid age
        {"name": "John", "age": 200, "email": "john@test.com"},  # Age too high
        {"name": "Jane", "age": "not-a-number", "email": "jane@test.com"},  # Invalid age type
    ]

    for i, invalid_input in enumerate(invalid_inputs, 1):
        print(f"\n   Test 3.{i}: {invalid_input}")
        try:
            results = await orchestrator.run(graph, invalid_input)
            print("   ⚠️  Unexpectedly succeeded")
        except Exception as e:
            print(f"   ✅ Correctly caught error: {type(e).__name__}")
            print(f"   📝 Message: {str(e)[:100]}...")


async def demonstrate_string_parsing():
    """Show custom string input parsing."""

    print("\n📝 Test 4: String Input Parsing")
    print("-" * 40)

    graph = DirectedGraph()
    graph.add(NodeSpec("validate", validate_user_input))
    graph.add(NodeSpec("process", process_user_data).after("validate"))
    graph.add(NodeSpec("format", format_output).after("process"))

    orchestrator = Orchestrator(validator=coerce_validator())

    # String input that gets parsed
    string_input = "Carol Davis, 35, carol@email.com"

    try:
        results = await orchestrator.run(graph, string_input)
        print("   ✅ String parsing successful!")
        print(f"   📊 Parsed: '{string_input}'")
        print(f"   👤 Name: {results['process'].name}")
        print(f"   🎂 Age: {results['process'].age}")
        print(f"   📧 Email: {results['process'].email}")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def main():
    """Demonstrate validation concepts."""

    print("✅ Example 03: Validation Basics")
    print("=" * 40)

    print("\n🎯 This example demonstrates:")
    print("   • Pydantic model validation")
    print("   • Type coercion and conversion")
    print("   • Validation error handling")
    print("   • Custom input parsing")

    await demonstrate_validation_success()
    await demonstrate_type_coercion()
    await demonstrate_validation_errors()
    await demonstrate_string_parsing()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ Pydantic Models - Define data structure and validation")
    print("   ✅ Type Coercion - Automatic type conversion when possible")
    print("   ✅ Validation Strategies - strict vs coerce behavior")
    print("   ✅ Error Handling - Graceful failure with meaningful messages")
    print("   ✅ Custom Parsing - Handle various input formats")

    print("\n🔗 Next: Run example 04 to learn about validation strategies!")


if __name__ == "__main__":
    asyncio.run(main())
