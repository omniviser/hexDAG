"""
Example 15: Pipeline Catalog.

This example demonstrates pipeline catalog management in hexAI:
- Basic catalog operations
- Pipeline registration
- Pipeline discovery
- Catalog management
- Pipeline metadata
"""

import asyncio
from typing import Any, Dict

from hexai.app.application.orchestrator import Orchestrator
from hexai.app.domain.dag import DirectedGraph, NodeSpec


async def simple_processor(input_data: str, **kwargs) -> dict:
    """Simple data processor."""
    return {
        "processed_data": input_data.upper(),
        "length": len(input_data),
        "processor_type": "simple",
    }


async def advanced_processor(input_data: dict, **kwargs) -> dict:
    """Advanced data processor."""
    processed_data = input_data.get("processed_data", "")

    return {
        "advanced_data": processed_data.replace(" ", "_"),
        "word_count": len(processed_data.split()),
        "processor_type": "advanced",
    }


async def result_aggregator(input_data: Any, **kwargs) -> dict:
    """Aggregate results."""
    results = input_data

    return {
        "final_result": results.get("advanced_processor", {}).get("advanced_data", ""),
        "total_processors": 2,
        "catalog_pipeline": True,
    }


class MockPipelineCatalog:
    """Mock pipeline catalog for demonstration."""

    def __init__(self):
        self.pipelines = {}
        self.metadata = {}

    def register_pipeline(self, name: str, graph: DirectedGraph, metadata: Dict[str, Any] = None):
        """Register a pipeline in the catalog."""
        self.pipelines[name] = graph
        self.metadata[name] = metadata or {}
        print(f"   📝 Registered pipeline: {name}")

    def get_pipeline(self, name: str) -> DirectedGraph | None:
        """Get a pipeline from the catalog."""
        return self.pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        """List all pipelines in the catalog."""
        return list(self.pipelines.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a pipeline."""
        return self.metadata.get(name, {})

    def remove_pipeline(self, name: str) -> bool:
        """Remove a pipeline from the catalog."""
        if name in self.pipelines:
            del self.pipelines[name]
            del self.metadata[name]
            print(f"   🗑️  Removed pipeline: {name}")
            return True
        return False


async def demonstrate_basic_catalog():
    """Demonstrate basic catalog operations."""
    print("\n📚 Basic Pipeline Catalog")
    print("-" * 35)

    # Create catalog
    catalog = MockPipelineCatalog()

    # Create and register pipeline
    graph = DirectedGraph()

    # Add nodes
    simple = NodeSpec("simple_processor", simple_processor)
    advanced = NodeSpec("advanced_processor", advanced_processor).after("simple_processor")
    aggregator = NodeSpec("result_aggregator", result_aggregator).after("advanced_processor")

    graph.add(simple)
    graph.add(advanced)
    graph.add(aggregator)

    # Register pipeline
    metadata = {
        "description": "Basic data processing pipeline",
        "version": "1.0.0",
        "author": "hexAI Team",
        "tags": ["data-processing", "basic"],
    }

    catalog.register_pipeline("basic_processing", graph, metadata)

    # List pipelines
    pipelines = catalog.list_pipelines()
    print(f"   📋 Available pipelines: {pipelines}")

    # Get pipeline metadata
    pipeline_metadata = catalog.get_metadata("basic_processing")
    print(f"   📊 Pipeline metadata: {pipeline_metadata}")


async def demonstrate_pipeline_discovery():
    """Demonstrate pipeline discovery."""
    print("\n🔍 Pipeline Discovery")
    print("-" * 35)

    # Create catalog with multiple pipelines
    catalog = MockPipelineCatalog()

    # Create multiple pipelines
    pipelines_data = [
        ("text_processing", "Text processing pipeline"),
        ("data_analysis", "Data analysis pipeline"),
        ("ml_inference", "Machine learning inference pipeline"),
        ("etl_pipeline", "ETL data pipeline"),
    ]

    for name, description in pipelines_data:
        # Create simple graph for each pipeline
        graph = DirectedGraph()
        processor = NodeSpec(f"{name}_processor", simple_processor)
        graph.add(processor)

        metadata = {"description": description, "version": "1.0.0", "category": "processing"}

        catalog.register_pipeline(name, graph, metadata)

    # Discover pipelines
    available_pipelines = catalog.list_pipelines()
    print(f"   🔍 Discovered {len(available_pipelines)} pipelines:")

    for pipeline in available_pipelines:
        metadata = catalog.get_metadata(pipeline)
        print(f"      • {pipeline}: {metadata['description']}")


async def demonstrate_catalog_management():
    """Demonstrate catalog management operations."""
    print("\n📚 Catalog Management")
    print("-" * 35)

    # Create catalog
    catalog = MockPipelineCatalog()

    # Register pipelines
    for i in range(3):
        name = f"pipeline_{i+1}"
        graph = DirectedGraph()
        processor = NodeSpec(f"processor_{i+1}", simple_processor)
        graph.add(processor)

        metadata = {"description": f"Pipeline {i+1}", "version": "1.0.0", "created": "2024-01-01"}

        catalog.register_pipeline(name, graph, metadata)

    # List all pipelines
    print(f"   📋 Total pipelines: {len(catalog.list_pipelines())}")

    # Remove a pipeline
    catalog.remove_pipeline("pipeline_2")

    # List remaining pipelines
    remaining = catalog.list_pipelines()
    print(f"   📋 Remaining pipelines: {remaining}")


async def demonstrate_pipeline_execution():
    """Demonstrate executing pipelines from catalog."""
    print("\n🚀 Pipeline Execution from Catalog")
    print("-" * 35)

    # Create catalog
    catalog = MockPipelineCatalog()

    # Create and register a pipeline
    graph = DirectedGraph()

    # Add nodes
    simple = NodeSpec("simple_processor", simple_processor)
    advanced = NodeSpec("advanced_processor", advanced_processor).after("simple_processor")
    aggregator = NodeSpec("result_aggregator", result_aggregator).after("advanced_processor")

    graph.add(simple)
    graph.add(advanced)
    graph.add(aggregator)

    # Register pipeline
    catalog.register_pipeline("executable_pipeline", graph)

    # Execute pipeline from catalog
    print("   🚀 Executing pipeline from catalog...")

    try:
        # Get pipeline from catalog
        pipeline = catalog.get_pipeline("executable_pipeline")

        if pipeline:
            # Execute pipeline
            orchestrator = Orchestrator()
            result = await orchestrator.run(pipeline, "test data from catalog")

            print("   ✅ Pipeline execution successful")
            print(f"   📋 Result: {result['result_aggregator']['final_result']}")
        else:
            print("   ❌ Pipeline not found in catalog")

    except Exception as e:
        print(f"   ❌ Pipeline execution failed: {e}")


async def main():
    """Demonstrate pipeline catalog capabilities."""

    print("📚 Example 15: Pipeline Catalog")
    print("=" * 35)

    print("\n🎯 This example demonstrates:")
    print("   • Basic catalog operations")
    print("   • Pipeline registration")
    print("   • Pipeline discovery")
    print("   • Catalog management")
    print("   • Pipeline metadata")

    # Test 1: Basic Catalog
    await demonstrate_basic_catalog()

    # Test 2: Pipeline Discovery
    await demonstrate_pipeline_discovery()

    # Test 3: Catalog Management
    await demonstrate_catalog_management()

    # Test 4: Pipeline Execution
    await demonstrate_pipeline_execution()

    print(f"\n🎯 Key Concepts Learned:")
    print("   ✅ Pipeline Catalog - Manage and organize pipelines")
    print("   ✅ Pipeline Registration - Register pipelines with metadata")
    print("   ✅ Pipeline Discovery - Find and list available pipelines")
    print("   ✅ Catalog Management - Add, remove, and manage pipelines")
    print("   ✅ Pipeline Execution - Execute pipelines from catalog")

    print(f"\n🔗 Next: Run example 16 to learn about validation errors!")


if __name__ == "__main__":
    asyncio.run(main())
