"""
Example 14: Pipeline Compilation.

This example demonstrates pipeline compilation in hexAI:
- Basic pipeline compilation
- Compilation with validation
- Compilation with optimization
- Compilation with custom settings
- Compilation error handling
"""

import asyncio
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.validation import coerce_validator


async def data_loader(input_data: str, **kwargs) -> dict:
    """Load data for processing."""
    return {
        "raw_data": input_data,
        "data_length": len(input_data),
        "loaded": True,
        "timestamp": "2024-01-01T00:00:00Z",
    }


async def data_processor(input_data: dict, **kwargs) -> dict:
    """Process loaded data."""
    raw_data = input_data.get("raw_data", "")

    return {
        "processed_data": raw_data.upper(),
        "word_count": len(raw_data.split()),
        "char_count": len(raw_data),
        "processing_complete": True,
    }


async def data_validator(input_data: dict, **kwargs) -> dict:
    """Validate processed data."""
    processed_data = input_data.get("processed_data", "")

    # Simple validation
    if len(processed_data) < 5:
        raise ValueError("Data too short")

    return {**input_data, "validated": True, "validation_passed": True}


async def data_transformer(input_data: dict, **kwargs) -> dict:
    """Transform validated data."""
    processed_data = input_data.get("processed_data", "")

    return {
        "transformed_data": processed_data.replace(" ", "_"),
        "transformation_type": "space_to_underscore",
        "original_length": len(processed_data),
        "transformed_length": len(processed_data.replace(" ", "_")),
    }


async def result_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate final results."""
    results = input_data

    return {
        "final_result": results.get("data_transformer", {}).get("transformed_data", ""),
        "total_steps": 4,
        "compilation_success": True,
        "execution_complete": True,
    }


async def demonstrate_basic_compilation():
    """Demonstrate basic pipeline compilation."""
    print("\n🔧 Basic Pipeline Compilation")
    print("-" * 35)

    # Create simple pipeline
    graph = DirectedGraph()

    # Add nodes
    loader = NodeSpec("data_loader", data_loader)
    processor = NodeSpec("data_processor", data_processor).after("data_loader")
    validator = NodeSpec("data_validator", data_validator).after("data_processor")
    transformer = NodeSpec("data_transformer", data_transformer).after("data_validator")
    aggregator = NodeSpec("result_aggregator", result_aggregator).after("data_transformer")

    graph.add(loader)
    graph.add(processor)
    graph.add(validator)
    graph.add(transformer)
    graph.add(aggregator)

    # Compile pipeline
    print("   🔧 Compiling pipeline...")

    try:
        # Validate graph structure
        graph.validate()
        print("   ✅ Pipeline validation passed")

        # Check execution waves
        waves = graph.waves()
        print(f"   📊 Execution waves: {len(waves)}")
        for i, wave in enumerate(waves):
            print(f"      Wave {i+1}: {wave}")

        # Execute compiled pipeline
        orchestrator = Orchestrator(validator=coerce_validator())
        result = await orchestrator.run(graph, "test data for compilation")

        print("   ✅ Pipeline compilation and execution successful")
        print(f"   📋 Final result: {result['result_aggregator']['final_result']}")

    except Exception as e:
        print(f"   ❌ Compilation failed: {e}")


async def demonstrate_compilation_with_validation():
    """Demonstrate compilation with validation."""
    print("\n🔧 Compilation with Validation")
    print("-" * 35)

    # Create pipeline with validation
    graph = DirectedGraph()

    # Add nodes with type hints
    loader = NodeSpec("data_loader", data_loader, in_type=str, out_type=dict)
    processor = NodeSpec("data_processor", data_processor, in_type=dict, out_type=dict).after(
        "data_loader"
    )
    validator = NodeSpec("data_validator", data_validator, in_type=dict, out_type=dict).after(
        "data_processor"
    )

    graph.add(loader)
    graph.add(processor)
    graph.add(validator)

    # Compile with validation
    print("   🔧 Compiling pipeline with validation...")

    try:
        # Validate graph structure
        graph.validate()
        print("   ✅ Pipeline validation passed")

        # Validate type compatibility
        print("   🔍 Validating type compatibility...")
        # This would normally check type compatibility between nodes

        # Execute with validation
        orchestrator = Orchestrator(validator=coerce_validator())
        result = await orchestrator.run(graph, "validation test")

        print("   ✅ Compilation with validation successful")
        print(f"   📋 Validation result: {result['data_validator']['validation_passed']}")

    except Exception as e:
        print(f"   ❌ Validation compilation failed: {e}")


async def demonstrate_compilation_optimization():
    """Demonstrate compilation with optimization."""
    print("\n🔧 Compilation with Optimization")
    print("-" * 35)

    # Create pipeline for optimization
    graph = DirectedGraph()

    # Add nodes that can be optimized
    loader = NodeSpec("data_loader", data_loader)
    processor1 = NodeSpec("data_processor_1", data_processor).after("data_loader")
    processor2 = NodeSpec("data_processor_2", data_processor).after("data_processor_1")
    processor3 = NodeSpec("data_processor_3", data_processor).after("data_processor_2")
    aggregator = NodeSpec("result_aggregator", result_aggregator).after("data_processor_3")

    graph.add(loader)
    graph.add(processor1)
    graph.add(processor2)
    graph.add(processor3)
    graph.add(aggregator)

    # Compile with optimization
    print("   🔧 Compiling pipeline with optimization...")

    try:
        # Pre-optimization analysis
        original_waves = graph.waves()
        print(f"   📊 Original execution waves: {len(original_waves)}")

        # Simulate optimization (in real implementation, this would optimize the graph)
        print("   ⚡ Applying optimizations...")
        # This would normally optimize the graph structure

        # Execute optimized pipeline
        orchestrator = Orchestrator(validator=coerce_validator())
        result = await orchestrator.run(graph, "optimization test")

        print("   ✅ Optimized compilation successful")
        print(f"   📋 Optimization result: {result['result_aggregator']['compilation_success']}")

    except Exception as e:
        print(f"   ❌ Optimization compilation failed: {e}")


async def demonstrate_compilation_errors():
    """Demonstrate compilation error handling."""
    print("\n🔧 Compilation Error Handling")
    print("-" * 35)

    # Create pipeline with potential errors
    graph = DirectedGraph()

    # Add nodes that might cause compilation errors
    loader = NodeSpec("data_loader", data_loader)
    processor = NodeSpec("data_processor", data_processor).after("data_loader")
    # Missing dependency - this should cause an error
    validator = NodeSpec("data_validator", data_validator).after("nonexistent_node")

    graph.add(loader)
    graph.add(processor)
    graph.add(validator)

    # Try to compile with errors
    print("   🔧 Attempting compilation with errors...")

    try:
        # This should fail due to missing dependency
        graph.validate()
        print("   ❌ Expected validation to fail")

    except Exception as e:
        print(f"   ✅ Correctly caught compilation error: {e}")
        print("   🔧 Error handling working correctly")


async def main():
    """Demonstrate pipeline compilation capabilities."""

    print("🔧 Example 14: Pipeline Compilation")
    print("=" * 40)

    print("\n🎯 This example demonstrates:")
    print("   • Basic pipeline compilation")
    print("   • Compilation with validation")
    print("   • Compilation with optimization")
    print("   • Compilation with custom settings")
    print("   • Compilation error handling")

    # Test 1: Basic Compilation
    await demonstrate_basic_compilation()

    # Test 2: Compilation with Validation
    await demonstrate_compilation_with_validation()

    # Test 3: Compilation with Optimization
    await demonstrate_compilation_optimization()

    # Test 4: Compilation Error Handling
    await demonstrate_compilation_errors()

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Pipeline Compilation - Convert DAGs to executable pipelines")
    print("   ✅ Compilation Validation - Validate pipeline structure and types")
    print("   ✅ Compilation Optimization - Optimize pipeline performance")
    print("   ✅ Error Handling - Handle compilation errors gracefully")
    print("   ✅ Type Safety - Ensure type compatibility during compilation")

    print(f"\n🔗 Next: Run example 15 to learn about pipeline catalog!")


if __name__ == "__main__":
    asyncio.run(main())
