#!/usr/bin/env python3
"""
📊 Example 11: DAG Visualization and Debugging.

This example teaches:
- DAG structure visualization
- Execution flow analysis
- Debugging complex pipelines
- Performance bottleneck identification

Run: python examples/11_dag_visualization.py
"""

import asyncio
from typing import Any

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec
from hexai.core.domain.dag_visualizer import DAGVisualizer
from hexai.core.validation import coerce_validator


async def data_ingestion(input_data: str) -> dict:
    """Ingest raw data."""
    await asyncio.sleep(0.1)
    return {
        "raw_data": input_data,
        "ingestion_timestamp": "2024-01-01T10:00:00Z",
        "source": "user_input",
    }


async def data_validation(input_data: dict) -> dict:
    """Validate ingested data."""
    await asyncio.sleep(0.05)
    raw_data = input_data.get("raw_data", "")

    is_valid = len(raw_data) > 0 and not raw_data.strip() == ""

    return {
        "validated_data": raw_data if is_valid else "default_data",
        "is_valid": is_valid,
        "validation_errors": [] if is_valid else ["Empty or invalid input"],
        "original": input_data,
    }


async def feature_extraction(input_data: dict) -> dict:
    """Extract features from validated data."""
    await asyncio.sleep(0.2)
    data = input_data.get("validated_data", "")

    return {
        "features": {
            "length": len(data),
            "word_count": len(data.split()),
            "has_numbers": any(c.isdigit() for c in data),
            "has_uppercase": any(c.isupper() for c in data),
        },
        "feature_extraction_complete": True,
        "source_data": input_data,
    }


async def model_prediction(input_data: dict) -> dict:
    """Make predictions based on features."""
    await asyncio.sleep(0.3)
    features = input_data.get("features", {})

    # Simple prediction logic
    score = 0
    if features.get("length", 0) > 10:
        score += 0.3
    if features.get("word_count", 0) > 2:
        score += 0.4
    if features.get("has_numbers", False):
        score += 0.2
    if features.get("has_uppercase", False):
        score += 0.1

    return {
        "prediction_score": min(score, 1.0),
        "prediction_class": "positive" if score > 0.5 else "negative",
        "confidence": score * 0.9,  # Slightly lower confidence
        "model_version": "v1.0",
        "input_features": features,
    }


async def result_formatting(input_data: Any, **kwargs) -> dict:
    """Format the final results for output."""
    # Extract data from input_data dictionary
    prediction = input_data.get("model_prediction", {})
    validation = input_data.get("data_validation", {})

    # Format the results
    final_result = {
        "prediction": prediction.get("prediction_class", "unknown"),
        "score": prediction.get("confidence", 0.0),
        "confidence": prediction.get("confidence", 0.0),
        "is_valid_input": validation.get("is_valid", False),
        "processing_complete": True,
        "formatted_at": "2024-01-01T00:00:00Z",
    }

    return {"final_result": final_result, "formatting_complete": True, "output_ready": True}


async def audit_logging(input_data: Any, **kwargs) -> dict:
    """Log audit information about the pipeline execution."""
    # Extract data from input_data dictionary
    prediction = input_data.get("model_prediction", {})
    validation = input_data.get("data_validation", {})
    formatting = input_data.get("result_formatting", {})

    # Create audit log
    audit_log = {
        "pipeline_execution": "completed",
        "prediction_made": prediction.get("prediction_class", "unknown"),
        "validation_passed": validation.get("is_valid", False),
        "formatting_complete": formatting.get("formatting_complete", False),
        "timestamp": "2024-01-01T00:00:00Z",
    }

    return {
        "audit_log": audit_log,
        "processing_summary": "Pipeline execution audited successfully",
        "audit_complete": True,
    }


def create_complex_pipeline() -> DirectedGraph:
    """Create a complex ML pipeline for visualization."""

    graph = DirectedGraph()

    # Data ingestion layer
    graph.add(NodeSpec("data_ingestion", data_ingestion))

    # Data processing layer
    graph.add(NodeSpec("data_validation", data_validation).after("data_ingestion"))
    graph.add(NodeSpec("feature_extraction", feature_extraction).after("data_validation"))

    # ML prediction layer
    graph.add(NodeSpec("model_prediction", model_prediction).after("feature_extraction"))

    # Output layer (parallel processing)
    graph.add(
        NodeSpec("result_formatting", result_formatting).after(
            "model_prediction", "data_validation"
        )
    )
    graph.add(
        NodeSpec("audit_logging", audit_logging).after(
            "model_prediction", "data_validation", "result_formatting"
        )
    )

    return graph


async def demonstrate_dag_visualization():
    """Demonstrate DAG visualization capabilities."""

    print("\n📊 DAG Visualization Demo")
    print("=" * 40)

    # Create complex pipeline
    graph = create_complex_pipeline()

    print("\n📋 Pipeline Overview:")
    print(f"   • Total nodes: {len(graph.nodes)}")
    print(f"   • Node names: {list(graph.nodes.keys())}")

    # Analyze execution waves
    print("\n🌊 Execution Wave Analysis:")
    waves = graph.waves()
    for i, wave in enumerate(waves, 1):
        if len(wave) == 1:
            print(f"   Wave {i}: {wave[0]} (sequential)")
        else:
            print(f"   Wave {i}: {', '.join(wave)} (parallel)")

    print(
        f"\n   💡 Total waves: {len(waves)} (parallelism opportunities: {sum(1 for wave in waves if len(wave) > 1)})"
    )

    # Show dependencies
    print("\n🔗 Dependency Analysis:")
    for node_name, node_spec in graph.nodes.items():
        if node_spec.deps:
            deps_str = ", ".join(node_spec.deps)
            print(f"   • {node_name} depends on: {deps_str}")
        else:
            print(f"   • {node_name} has no dependencies (root node)")

    # Create visualizer
    visualizer = DAGVisualizer(graph)

    print("\n🎨 Generating DAG visualization...")

    # Generate DOT format visualization
    visualizer.to_dot(title="ML Pipeline DAG")
    print("\n📊 DAG Structure (DOT format):")
    print("   (DOT content generated - can be rendered with Graphviz)")

    # Show a simplified text representation
    print("\n📝 Simplified Pipeline Structure:")
    print("   data_ingestion")
    print("   └── data_validation")
    print("       └── feature_extraction")
    print("           └── model_prediction")
    print("               ├── result_formatting")
    print("               └── audit_logging")

    # Validate graph structure
    print("\n🔍 Graph Validation:")
    try:
        graph.validate()
        print("   ✅ Graph structure is valid")
        print("   ✅ No circular dependencies detected")
        print("   ✅ All dependencies are satisfied")
    except Exception as e:
        print(f"   ❌ Graph validation failed: {e}")

    return graph


async def demonstrate_execution_analysis():
    """Demonstrate execution analysis and performance visualization."""

    print("\n⚡ Execution Analysis Demo")
    print("=" * 40)

    graph = create_complex_pipeline()
    orchestrator = Orchestrator(validator=coerce_validator())

    print("\n🚀 Executing pipeline for performance analysis...")

    import time

    start_time = time.time()
    results = await orchestrator.run(graph, "sample machine learning input data")
    end_time = time.time()

    total_time = end_time - start_time
    print(f"   ⏱️  Total execution time: {total_time:.3f} seconds")

    # Analyze results
    print("\n📊 Execution Results Analysis:")

    # Check final results
    final_result = results.get("result_formatting", {}).get("final_result", {})
    audit_info = results.get("audit_logging", {}).get("audit_log", {})

    print(
        f"   📈 Prediction: {final_result.get('prediction')} (score: {final_result.get('score', 0):.3f})"
    )
    print(f"   🎯 Confidence: {final_result.get('confidence', 0):.3f}")
    print(f"   ✅ Data valid: {final_result.get('is_valid_input')}")
    print(f"   📋 Audit status: {audit_info.get('pipeline_execution')}")

    # Show data flow
    print("\n🔄 Data Flow Trace:")
    print("   1. Input: 'sample machine learning input data'")
    print(f"   2. Validation: {results['data_validation']['is_valid']}")
    print(f"   3. Features: {list(results['feature_extraction']['features'].keys())}")
    print(f"   4. Prediction: {results['model_prediction']['prediction_class']}")
    print(f"   5. Formatted: {results['result_formatting']['final_result']['processing_complete']}")
    print(f"   6. Audited: {results['audit_logging']['processing_summary']}")

    # Performance insights
    print("\n💡 Performance Insights:")
    print("   • Wave 1 (data_ingestion): Handles initial I/O")
    print("   • Wave 2 (data_validation): Quick validation check")
    print("   • Wave 3 (feature_extraction): Feature computation")
    print("   • Wave 4 (model_prediction): Most compute-intensive")
    print("   • Wave 5 (result_formatting): Parallel with logging")
    print("   • Wave 6 (audit_logging): Final cleanup")

    return results


async def main():
    """Demonstrate DAG visualization and debugging."""

    print("📊 Example 11: DAG Visualization and Debugging")
    print("=" * 55)

    print("\n🎯 This example demonstrates:")
    print("   • Complex DAG structure visualization")
    print("   • Execution wave analysis")
    print("   • Dependency relationship mapping")
    print("   • Performance bottleneck identification")
    print("   • Execution flow debugging")

    await demonstrate_dag_visualization()
    await demonstrate_execution_analysis()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ DAG Visualization - Understand complex pipeline structure")
    print("   ✅ Wave Analysis - Identify parallelization opportunities")
    print("   ✅ Dependency Mapping - Track data flow between nodes")
    print("   ✅ Performance Analysis - Find execution bottlenecks")
    print("   ✅ Debugging Tools - Validate and troubleshoot pipelines")

    print("\n💡 Visualization Tips:")
    print("   • Use text visualization for debugging")
    print("   • Analyze waves to optimize parallelism")
    print("   • Track data flow for validation")
    print("   • Monitor execution times for performance")

    print("\n🔗 Next: Run example 12 to learn about data aggregation!")


if __name__ == "__main__":
    asyncio.run(main())
