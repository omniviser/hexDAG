#!/usr/bin/env python3
"""Example 24: Unified Configuration System with Type-Specific Ports.

This example demonstrates:
- Declarative YAML pipeline configuration
- Global ports (adapters) configuration
- Type-specific port defaults (type_ports)
- Global policies configuration
- PipelineConfig and OrchestratorFactory usage

Run: PYTHONPATH=. uv run python examples/24_unified_configuration.py
"""

import asyncio

from hexdag.core.orchestration.orchestrator_factory import OrchestratorFactory
from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder


async def process_text(input_data: str) -> dict:
    """Simple text processing function.

    Args:
        input_data: Input text string

    Returns:
        Dictionary with text statistics
    """
    return {
        "original": input_data,
        "length": len(input_data),
        "words": len(input_data.split()),
        "uppercase": input_data.upper(),
    }


async def analyze_data(input_data: dict) -> dict:
    """Analyze processed data.

    Args:
        input_data: Processed text data from previous node

    Returns:
        Dictionary with analysis results
    """
    return {
        "analysis": f"Processed {input_data['words']} words in text of length {input_data['length']}",
        "stats": input_data,
    }


async def main():
    """Demonstrate unified configuration system."""

    # YAML pipeline with global ports and type-specific defaults
    yaml_content = """
apiVersion: hexdag.omniviser.io/v1alpha1
kind: Pipeline
metadata:
  name: unified-config-demo
  description: Demonstration of unified configuration system
  version: "1.0"

spec:
  # Global ports - available to all nodes
  ports:
    observer_manager: plugin:local_observer_manager
    policy_manager: plugin:local_policy_manager

  # Type-specific ports - Not used in this example but demonstrates capability
  # When you have LLM/agent nodes, you can configure type-specific ports:
  # type_ports:
  #   agent:
  #     llm: plugin:some_llm_adapter

  # Global policies - apply to all nodes unless overridden
  policies:
    timeout: core:timeout(timeout_seconds=60)
    retry: core:retry(max_retries=2, delay_seconds=1.0)

  # Pipeline nodes
  nodes:
    - kind: function_node
      metadata:
        name: process_input
        annotations:
          description: Process input text
      spec:
        fn: process_text
        dependencies: []

    - kind: function_node
      metadata:
        name: analyze_result
        annotations:
          description: Analyze processed data
      spec:
        fn: analyze_data
        dependencies: [process_input]
"""

    print("=" * 70)
    print("🚀 Example 24: Unified Configuration System")
    print("=" * 70)

    # Step 1: Create YAML builder and register functions
    print("\n📦 Step 1: Creating YAML builder and registering functions...")
    builder = YamlPipelineBuilder()
    builder.register_function("process_text", process_text)
    builder.register_function("analyze_data", analyze_data)

    # Step 2: Build DirectedGraph and PipelineConfig from YAML
    print("\n📝 Step 2: Building pipeline from YAML...")
    graph, pipeline_config = builder.build_from_yaml_string(yaml_content)

    print(f"   ✓ Built graph with {len(graph.nodes)} nodes")
    print(f"   ✓ Pipeline: {pipeline_config.metadata['name']}")
    print(f"   ✓ Global ports configured: {len(pipeline_config.ports)}")
    if pipeline_config.type_ports:
        print(f"   ✓ Type-specific ports: {len(pipeline_config.type_ports)}")
    print(f"   ✓ Global policies: {len(pipeline_config.policies)}")

    # Step 3: Create orchestrator from PipelineConfig
    print("\n🏗️  Step 3: Creating orchestrator from PipelineConfig...")
    factory = OrchestratorFactory()
    orchestrator = factory.create_orchestrator(
        pipeline_config=pipeline_config,
        additional_ports=None,  # Could add runtime-specific ports here
    )

    print(f"   ✓ Orchestrator created with {len(orchestrator.ports)} global ports")
    if orchestrator.ports_config:
        print("   ✓ PortsConfiguration active with type-specific routing")

    # Step 4: Execute the pipeline
    print("\n⚡ Step 4: Executing pipeline...")
    print("\nInput: 'Hello world! This is a test of the unified configuration system.'")

    result = await orchestrator.run(
        graph=graph,
        initial_input="Hello world! This is a test of the unified configuration system.",
    )

    print("\n📊 Execution Results:")
    print("=" * 70)

    # Display results from each node
    for node_name in ["process_input", "analyze_result"]:
        if node_name in result:
            print(f"\n{node_name}:")
            node_result = result[node_name]
            if isinstance(node_result, dict):
                for key, value in node_result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {node_result}")

    print("\n" + "=" * 70)
    print("✅ Demonstration Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Declarative YAML pipeline configuration")
    print("   • Global ports (adapters) configured in YAML")
    print("   • Type-specific port defaults (type_ports)")
    print("   • Global policies (timeout, retry)")
    print("   • OrchestratorFactory for seamless instantiation")
    print("   • Component instantiation from shorthand specs")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
