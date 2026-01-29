"""Test script to demonstrate hexDAG MCP tools locally.

This script shows what the MCP server does when Claude calls its tools.
Run this to understand how the MCP server works before configuring Claude Desktop.

Usage:
    uv run python examples/mcp/test_mcp_tools.py
"""

import json

# Import MCP tools directly
from hexdag.mcp_server import (
    explain_yaml_structure,
    generate_pipeline_template,
    list_adapters,
    list_macros,
    list_nodes,
    list_policies,
    list_tools,
    validate_yaml_pipeline,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_list_nodes() -> None:
    """Demonstrate the list_nodes tool."""
    print_section("1. List Available Nodes")
    result = list_nodes()
    data = json.loads(result)

    print("Available node types by namespace:\n")
    for namespace, nodes in data.items():
        print(f"Namespace: {namespace}")
        for node in nodes[:3]:  # Show first 3 nodes per namespace
            print(f"  • {node['name']}")
            print(f"    {node['description'][:80]}...")
        if len(nodes) > 3:
            print(f"  ... and {len(nodes) - 3} more nodes")
        print()


def demo_list_adapters() -> None:
    """Demonstrate the list_adapters tool."""
    print_section("2. List Available Adapters")

    # List all adapters
    result = list_adapters()
    data = json.loads(result)

    print("Available adapters by port type:\n")
    for port_type, adapters in data.items():
        print(f"Port: {port_type}")
        for adapter in adapters:
            print(f"  • {adapter['name']} ({adapter['namespace']})")
        print()


def demo_list_tools() -> None:
    """Demonstrate the list_tools tool."""
    print_section("3. List Available Tools")
    result = list_tools()
    data = json.loads(result)

    print("Available tools by namespace:\n")
    for namespace, tools in data.items():
        print(f"Namespace: {namespace}")
        for tool in tools:
            print(f"  • {tool['name']}")
        print()


def demo_list_macros() -> None:
    """Demonstrate the list_macros tool."""
    print_section("4. List Available Macros")
    result = list_macros()
    data = json.loads(result)

    print("Available macro templates:\n")
    for macro in data:
        print(f"  • {macro['name']} ({macro['namespace']})")
        print(f"    {macro['description']}")
    print()


def demo_list_policies() -> None:
    """Demonstrate the list_policies tool."""
    print_section("5. List Available Policies")
    result = list_policies()
    data = json.loads(result)

    print("Available execution policies:\n")
    for policy in data:
        print(f"  • {policy['name']}")
        print(f"    {policy['description']}")
    print()


def demo_generate_template() -> None:
    """Demonstrate the generate_pipeline_template tool."""
    print_section("6. Generate Pipeline Template")

    print("Generating a simple 2-node pipeline template...\n")

    result = generate_pipeline_template(
        pipeline_name="example-pipeline",
        description="Example LLM and function pipeline",
        node_types=["llm_node", "function_node"],
    )

    print(result)


def demo_validate_pipeline() -> None:
    """Demonstrate the validate_yaml_pipeline tool."""
    print_section("7. Validate Pipeline")

    # Create a valid pipeline
    valid_pipeline = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: test-pipeline
  description: Test pipeline
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
        output_key: result
      dependencies: []
"""

    print("Validating a pipeline configuration...\n")
    result = validate_yaml_pipeline(valid_pipeline)
    data = json.loads(result)

    if data["valid"]:
        print("✓ Pipeline is VALID")
        print(f"  Nodes: {data['node_count']}")
        print(f"  Node names: {', '.join(data['nodes'])}")
    else:
        print("✗ Pipeline is INVALID")
        print(f"  Error: {data['error']}")

    print()

    # Try an invalid pipeline
    invalid_pipeline = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: bad-pipeline
spec:
  nodes:
    - kind: nonexistent_node
      metadata:
        name: bad_node
      spec: {}
"""

    print("Validating an invalid pipeline...\n")
    result = validate_yaml_pipeline(invalid_pipeline)
    data = json.loads(result)

    if data["valid"]:
        print("✓ Pipeline is VALID")
    else:
        print("✗ Pipeline is INVALID (as expected)")
        print(f"  Error type: {data['error_type']}")
        print(f"  Error: {data['error'][:100]}...")

    print()


def demo_explain_structure() -> None:
    """Demonstrate the explain_yaml_structure tool."""
    print_section("8. YAML Structure Documentation")

    result = explain_yaml_structure()
    # Print first 500 characters
    print(result[:500] + "...\n")
    print("(Full documentation available via the tool)")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  hexDAG MCP Server - Tool Demonstrations")
    print("  This shows what Claude sees when it uses the MCP tools")
    print("=" * 70)

    try:
        demo_list_nodes()
        demo_list_adapters()
        demo_list_tools()
        demo_list_macros()
        demo_list_policies()
        demo_generate_template()
        demo_validate_pipeline()
        demo_explain_structure()

        print_section("Summary")
        print("✓ All MCP tools demonstrated successfully!")
        print()
        print("Next steps:")
        print("  1. Configure Claude Desktop (see QUICKSTART.md)")
        print("  2. Ask Claude to build pipelines for you")
        print("  3. Claude will use these tools automatically")
        print()

    except Exception as e:
        print(f"\n✗ Error running demonstrations: {e}")
        print("Make sure hexDAG is properly installed with: uv sync --extra mcp")


if __name__ == "__main__":
    main()
