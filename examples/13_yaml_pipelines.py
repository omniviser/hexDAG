#!/usr/bin/env python3
"""
🏢 Example 13: YAML Pipeline Definitions.

This example teaches:
- YAML-based pipeline definitions
- Pipeline compilation
- Enterprise pipeline features
- Pipeline catalog management

Run: python examples/13_yaml_pipelines.py
"""

import asyncio

from hexai.core.application.orchestrator import Orchestrator
from hexai.core.domain.dag import DirectedGraph, NodeSpec


async def data_loader(input_data: str) -> dict:
    """Load and parse input data."""
    return {"raw_input": input_data, "processed": True, "timestamp": "2024-01-01T10:00:00Z"}


async def text_processor(input_data: dict) -> dict:
    """Process text data."""
    text = input_data.get("raw_input", "")
    words = text.split()

    return {
        "word_count": len(words),
        "char_count": len(text),
        "processed_text": text.upper(),
        "original": input_data,
    }


async def sentiment_analyzer(input_data: dict) -> dict:
    """Analyze sentiment of text."""
    text = input_data.get("processed_text", "")

    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "sad"]

    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)

    if positive_score > negative_score:
        sentiment = "positive"
        confidence = min(0.9, (positive_score - negative_score) / 5)
    elif negative_score > positive_score:
        sentiment = "negative"
        confidence = min(0.9, (negative_score - positive_score) / 5)
    else:
        sentiment = "neutral"
        confidence = 0.5

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "analysis_data": input_data,
    }


async def report_generator(text_data: dict, sentiment_data: dict) -> dict:
    """Generate comprehensive report."""
    return {
        "report": {
            "text_summary": {
                "word_count": text_data.get("word_count", 0),
                "char_count": text_data.get("char_count", 0),
                "processed_text": text_data.get("processed_text", ""),
            },
            "sentiment_analysis": {
                "sentiment": sentiment_data.get("sentiment"),
                "confidence": sentiment_data.get("confidence"),
                "positive_score": sentiment_data.get("positive_score"),
                "negative_score": sentiment_data.get("negative_score"),
            },
            "timestamp": text_data.get("original", {}).get("timestamp"),
        },
        "analysis_complete": True,
    }


def create_yaml_pipeline_definition() -> str:
    """Create a YAML pipeline definition."""

    yaml_content = """
name: text_analysis_pipeline
version: "1.0.0"
description: "Analyze text sentiment and generate reports"

input_schema:
  type: string
  description: "Text to analyze"

output_schema:
  type: object
  properties:
    report:
      type: object
      properties:
        text_summary:
          type: object
        sentiment_analysis:
          type: object
    analysis_complete:
      type: boolean

nodes:
  data_loader:
    type: function
    function: data_loader
    description: "Load and parse input data"

  text_processor:
    type: function
    function: text_processor
    depends_on: ["data_loader"]
    description: "Process and analyze text content"

  sentiment_analyzer:
    type: function
    function: sentiment_analyzer
    depends_on: ["text_processor"]
    description: "Analyze sentiment of processed text"

  report_generator:
    type: function
    function: report_generator
    depends_on: ["text_processor", "sentiment_analyzer"]
    description: "Generate comprehensive analysis report"

config:
  validation_strategy: "coerce"
  max_concurrent_nodes: 4
  timeout_seconds: 300
"""

    return yaml_content


def create_simple_yaml_pipeline() -> str:
    """Create a simpler YAML pipeline for demonstration."""

    yaml_content = """
name: simple_text_pipeline
version: "1.0.0"
description: "Simple text processing pipeline"

nodes:
  process:
    type: function
    function: text_processor
    description: "Process input text"

  analyze:
    type: function
    function: sentiment_analyzer
    depends_on: ["process"]
    description: "Analyze sentiment"

config:
  validation_strategy: "coerce"
"""

    return yaml_content


async def demonstrate_yaml_parsing():
    """Demonstrate parsing YAML pipeline definitions."""

    print("\n📄 YAML Pipeline Definition Demo")
    print("=" * 40)

    # Create YAML content
    yaml_content = create_yaml_pipeline_definition()

    print("\n📋 YAML Pipeline Definition:")
    print("   (Showing key parts of the YAML)")
    print("   • Pipeline name: text_analysis_pipeline")
    print("   • Version: 1.0.0")
    print("   • Nodes: 4 (data_loader, text_processor, sentiment_analyzer, report_generator)")
    print("   • Dependencies: Complex multi-level dependencies")
    print("   • Config: coerce validation, 4 concurrent nodes")

    # Show the structure
    print("\n🏗️  Pipeline Structure:")
    print("   data_loader")
    print("   └── text_processor")
    print("       ├── sentiment_analyzer")
    print("       └── report_generator")
    print("           └── (depends on both text_processor and sentiment_analyzer)")

    return yaml_content


async def demonstrate_pipeline_execution():
    """Demonstrate executing a pipeline from YAML."""

    print("\n🚀 Pipeline Execution Demo")
    print("=" * 40)

    # Create the DAG manually (simulating YAML parsing)
    graph = DirectedGraph()

    # Add nodes
    graph.add(NodeSpec("data_loader", data_loader))
    graph.add(NodeSpec("text_processor", text_processor).after("data_loader"))
    graph.add(NodeSpec("sentiment_analyzer", sentiment_analyzer).after("text_processor"))
    graph.add(
        NodeSpec("report_generator", report_generator).after("text_processor", "sentiment_analyzer")
    )

    # Validate
    graph.validate()

    print("\n📊 Pipeline Analysis:")
    waves = graph.waves()
    print(f"   • Total waves: {len(waves)}")
    for i, wave in enumerate(waves, 1):
        print(f"   • Wave {i}: {wave}")

    # Execute
    orchestrator = Orchestrator()

    test_inputs = [
        "I love this product! It's amazing and wonderful.",
        "This is terrible. I hate it so much.",
        "The product is okay. Not great, not bad.",
    ]

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🧪 Test {i}: '{test_input[:30]}...'")

        try:
            results = await orchestrator.run(graph, test_input)

            report = results.get("report_generator", {}).get("report", {})
            sentiment = report.get("sentiment_analysis", {})

            confidence = sentiment.get("confidence", 0)
            print(f"   📈 Sentiment: {sentiment.get('sentiment')} (confidence: {confidence:.2f})")
            print(f"   📊 Word count: {report.get('text_summary', {}).get('word_count', 0)}")
            analysis_complete = results.get("report_generator", {}).get("analysis_complete", False)
            print(f"   ✅ Analysis complete: {analysis_complete}")

        except Exception as e:
            print(f"   ❌ Execution failed: {e}")

    return graph


async def demonstrate_pipeline_compilation():
    """Demonstrate pipeline compilation concepts."""

    print("\n⚙️ Pipeline Compilation Demo")
    print("=" * 40)

    # Simulate compilation steps
    print("\n🔧 Compilation Steps:")
    print("   1. Parse YAML definition")
    print("   2. Validate node dependencies")
    print("   3. Generate execution plan")
    print("   4. Optimize for performance")
    print("   5. Generate type stubs")
    print("   6. Create compiled pipeline")

    # Show compilation benefits
    print("\n💡 Compilation Benefits:")
    print("   • Type safety validation")
    print("   • Performance optimization")
    print("   • Early error detection")
    print("   • Code generation")
    print("   • Schema validation")

    # Simulate compiled output
    print("\n📦 Compiled Pipeline Features:")
    print("   • Optimized execution order")
    print("   • Type-validated inputs/outputs")
    print("   • Error handling built-in")
    print("   • Performance monitoring hooks")
    print("   • Schema compatibility checks")


async def demonstrate_pipeline_catalog():
    """Demonstrate pipeline catalog management."""

    print("\n📚 Pipeline Catalog Demo")
    print("=" * 40)

    # Simulate catalog operations
    pipelines = {
        "text_analysis": {
            "version": "1.0.0",
            "description": "Analyze text sentiment",
            "nodes": 4,
            "status": "active",
        },
        "simple_text": {
            "version": "1.0.0",
            "description": "Simple text processing",
            "nodes": 2,
            "status": "active",
        },
        "data_processing": {
            "version": "0.9.0",
            "description": "Data processing pipeline",
            "nodes": 6,
            "status": "beta",
        },
    }

    print("\n📋 Available Pipelines:")
    for name, info in pipelines.items():
        status_icon = "✅" if info["status"] == "active" else "🟡"
        print(f"   {status_icon} {name} v{info['version']} ({info['nodes']} nodes)")
        print(f"      {info['description']}")

    print("\n🔍 Catalog Operations:")
    print("   • List available pipelines")
    print("   • Get pipeline metadata")
    print("   • Version management")
    print("   • Dependency tracking")
    print("   • Usage statistics")


async def main():
    """Demonstrate YAML pipeline features."""

    print("🏢 Example 13: YAML Pipeline Definitions")
    print("=" * 50)

    print("\n🎯 This example demonstrates:")
    print("   • YAML-based pipeline definitions")
    print("   • Pipeline compilation concepts")
    print("   • Enterprise pipeline features")
    print("   • Pipeline catalog management")
    print("   • Complex dependency handling")

    await demonstrate_yaml_parsing()
    await demonstrate_pipeline_execution()
    await demonstrate_pipeline_compilation()
    await demonstrate_pipeline_catalog()

    print("\n🎯 Key Concepts Learned:")
    print("   ✅ YAML Definitions - Declarative pipeline specification")
    print("   ✅ Pipeline Compilation - Type safety and optimization")
    print("   ✅ Enterprise Features - Catalog management and versioning")
    print("   ✅ Complex Dependencies - Multi-level node relationships")
    print("   ✅ Validation Strategies - Configurable validation behavior")

    print("\n💡 Enterprise Benefits:")
    print("   • Declarative pipeline definitions")
    print("   • Version control and management")
    print("   • Type safety and validation")
    print("   • Performance optimization")
    print("   • Team collaboration")

    print("\n🔗 Next: Run example 14 to learn about pipeline compilation!")


if __name__ == "__main__":
    asyncio.run(main())
