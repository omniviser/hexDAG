"""Test the create_environment_pipelines_with_includes MCP tool."""

import json

import yaml

from hexdag.mcp_server import create_environment_pipelines_with_includes


def test_include_pattern_dev_staging_prod():
    """Test creating base + environment-specific files with includes."""
    print("\n" + "=" * 80)
    print("Test: Create Base + Environment Files (Include Pattern)")
    print("=" * 80)

    # Define shared nodes
    nodes = [
        {
            "kind": "macro_invocation",
            "name": "research_agent",
            "spec": {
                "macro": "core:reasoning_agent",
                "config": {
                    "main_prompt": "Research: {{question}}",
                    "max_steps": 5,
                    "allowed_tools": ["search", "calculate"],
                    "tool_format": "mixed",
                },
            },
            "dependencies": [],
        }
    ]

    # Define staging ports
    staging_ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {
                "api_key": "${STAGING_OPENAI_API_KEY}",
                "model": "gpt-4o-mini",
            },
        }
    }

    # Define production ports
    prod_ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4o"},
        }
    }

    # Create all environments with includes
    result_json = create_environment_pipelines_with_includes(
        pipeline_name="research-agent",
        description="Deep research agent",
        nodes=nodes,
        staging_ports=staging_ports,
        prod_ports=prod_ports,
    )

    result = json.loads(result_json)

    print("\n✅ Generated files:", list(result.keys()))

    # Validate base file
    print("\n" + "-" * 80)
    print("Base Configuration (research-agent_base.yaml):")
    print("-" * 80)
    base_yaml = yaml.safe_load(result["base"])
    print(f"  Name: {base_yaml['metadata']['name']}")
    print(f"  Nodes: {len(base_yaml['spec']['nodes'])}")
    print(f"  Has ports: {'ports' in base_yaml['spec']}")
    assert base_yaml["metadata"]["name"] == "research-agent-base"
    assert len(base_yaml["spec"]["nodes"]) == 1
    assert "ports" not in base_yaml["spec"]  # No ports in base
    print("  ✅ Base contains only nodes (no ports)")

    # Validate dev environment
    print("\n" + "-" * 80)
    print("Dev Environment (research-agent_dev.yaml):")
    print("-" * 80)
    dev_yaml = yaml.safe_load(result["dev"])
    print(f"  Include: {dev_yaml['include']}")
    print(f"  Name: {dev_yaml['metadata']['name']}")
    print(f"  LLM Adapter: {dev_yaml['ports']['llm']['adapter']}")
    assert dev_yaml["include"] == "./research-agent_base.yaml"
    assert dev_yaml["metadata"]["name"] == "research-agent-dev"
    assert dev_yaml["ports"]["llm"]["adapter"] == "plugin:mock_llm"
    print("  ✅ Dev includes base + uses mock adapters")

    # Validate staging environment
    print("\n" + "-" * 80)
    print("Staging Environment (research-agent_staging.yaml):")
    print("-" * 80)
    staging_yaml = yaml.safe_load(result["staging"])
    print(f"  Include: {staging_yaml['include']}")
    print(f"  Name: {staging_yaml['metadata']['name']}")
    print(f"  LLM Adapter: {staging_yaml['ports']['llm']['adapter']}")
    print(f"  LLM Model: {staging_yaml['ports']['llm']['config']['model']}")
    assert staging_yaml["include"] == "./research-agent_base.yaml"
    assert staging_yaml["metadata"]["name"] == "research-agent-staging"
    assert staging_yaml["ports"]["llm"]["config"]["model"] == "gpt-4o-mini"
    print("  ✅ Staging includes base + uses cheaper model")

    # Validate production environment
    print("\n" + "-" * 80)
    print("Production Environment (research-agent_prod.yaml):")
    print("-" * 80)
    prod_yaml = yaml.safe_load(result["prod"])
    print(f"  Include: {prod_yaml['include']}")
    print(f"  Name: {prod_yaml['metadata']['name']}")
    print(f"  LLM Adapter: {prod_yaml['ports']['llm']['adapter']}")
    print(f"  LLM Model: {prod_yaml['ports']['llm']['config']['model']}")
    assert prod_yaml["include"] == "./research-agent_base.yaml"
    assert prod_yaml["metadata"]["name"] == "research-agent-prod"
    assert prod_yaml["ports"]["llm"]["config"]["model"] == "gpt-4o"
    print("  ✅ Production includes base + uses full model")

    # Show file structure
    print("\n" + "-" * 80)
    print("File Structure:")
    print("-" * 80)
    print("""
research-agent/
├── research-agent_base.yaml      # Shared nodes (DRY principle)
├── research-agent_dev.yaml       # include: base + mock ports
├── research-agent_staging.yaml   # include: base + staging ports
└── research-agent_prod.yaml      # include: base + prod ports
    """)

    # Show benefits
    print("\n" + "-" * 80)
    print("Benefits of Include Pattern:")
    print("-" * 80)
    print("  ✅ DRY: Nodes defined once in base")
    print("  ✅ Maintainable: Update logic in one place")
    print("  ✅ Clear separation: Environment configs are minimal")
    print("  ✅ Easy review: Diff shows only port changes")

    print("\n" + "=" * 80)
    print("✅ Test Passed: Include pattern works correctly!")
    print("=" * 80)

    return result


def test_write_include_files_to_disk():
    """Demonstrate writing include-based files to disk."""
    print("\n" + "=" * 80)
    print("Demo: Writing Include-Based Files")
    print("=" * 80)

    nodes = [
        {
            "kind": "llm_node",
            "name": "analyzer",
            "spec": {"prompt_template": "Analyze: {{input}}"},
            "dependencies": [],
        }
    ]

    prod_ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {"api_key": "${OPENAI_API_KEY}", "model": "gpt-4o"},
        }
    }

    result_json = create_environment_pipelines_with_includes(
        pipeline_name="analyzer",
        description="Simple analyzer",
        nodes=nodes,
        prod_ports=prod_ports,
    )

    result = json.loads(result_json)

    print("\n📝 Example code to write files:")
    print("-" * 80)
    print("""
from pathlib import Path
import json

# Parse MCP result
result = json.loads(result_json)

# Write base file
Path("analyzer_base.yaml").write_text(result["base"])

# Write dev file (includes base)
Path("analyzer_dev.yaml").write_text(result["dev"])

# Write prod file (includes base)
Path("analyzer_prod.yaml").write_text(result["prod"])
    """)

    print("\n📄 Generated analyzer_base.yaml:")
    print("-" * 80)
    print(result["base"][:300] + "...")

    print("\n📄 Generated analyzer_dev.yaml:")
    print("-" * 80)
    print(result["dev"])

    print("\n📄 Generated analyzer_prod.yaml:")
    print("-" * 80)
    print(result["prod"])

    print("\n" + "=" * 80)
    print("✅ Demo Complete: Files ready to write!")
    print("=" * 80)


def test_compare_standalone_vs_include():
    """Compare standalone files vs include pattern."""
    print("\n" + "=" * 80)
    print("Comparison: Standalone vs Include Pattern")
    print("=" * 80)

    # Import both functions
    from hexdag.mcp_server import create_environment_pipelines

    nodes = [
        {
            "kind": "llm_node",
            "name": "node1",
            "spec": {"prompt_template": "Test"},
            "dependencies": [],
        }
    ]

    prod_ports = {"llm": {"adapter": "core:openai", "config": {"api_key": "${OPENAI_API_KEY}"}}}

    # Standalone approach
    standalone_json = create_environment_pipelines(
        pipeline_name="test", description="Test", nodes=nodes, prod_ports=prod_ports
    )
    standalone_result = json.loads(standalone_json)

    # Include approach
    include_json = create_environment_pipelines_with_includes(
        pipeline_name="test", description="Test", nodes=nodes, prod_ports=prod_ports
    )
    include_result = json.loads(include_json)

    print("\n" + "-" * 80)
    print("Standalone Approach:")
    print("-" * 80)
    print(f"  Files: {list(standalone_result.keys())}")
    print(f"  Dev size: {len(standalone_result['dev'])} bytes")
    print(f"  Prod size: {len(standalone_result['prod'])} bytes")
    print(f"  Total size: {sum(len(v) for v in standalone_result.values())} bytes")
    print("  Duplication: Nodes repeated in each file")

    print("\n" + "-" * 80)
    print("Include Approach:")
    print("-" * 80)
    print(f"  Files: {list(include_result.keys())}")
    print(f"  Base size: {len(include_result['base'])} bytes")
    print(f"  Dev size: {len(include_result['dev'])} bytes")
    print(f"  Prod size: {len(include_result['prod'])} bytes")
    print(f"  Total size: {sum(len(v) for v in include_result.values())} bytes")
    print("  Duplication: None - nodes in base only")

    print("\n" + "-" * 80)
    print("Recommendation:")
    print("-" * 80)
    print("  Use standalone: Simple pipelines, few environments")
    print("  Use includes: Complex pipelines, many environments, team collaboration")

    print("\n" + "=" * 80)
    print("✅ Comparison Complete!")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 MCP Include Pattern Test Suite")
    print("=" * 80)

    try:
        # Test 1: Full include pattern
        test_include_pattern_dev_staging_prod()

        # Test 2: Writing to disk demo
        test_write_include_files_to_disk()

        # Test 3: Compare approaches
        test_compare_standalone_vs_include()

        # Summary
        print("\n" + "=" * 80)
        print("✅ All Tests Passed!")
        print("=" * 80)
        print("\n💡 Usage Guide:")
        print("""
# Option 1: Standalone files (simple, self-contained)
from hexdag.mcp_server import create_environment_pipelines

result_json = create_environment_pipelines(
    pipeline_name="my-agent",
    description="My agent",
    nodes=[...],
    prod_ports={...}
)

# Each file is complete and independent
# Good for: Simple pipelines, single environment, quick prototypes


# Option 2: Include pattern (DRY, maintainable)
from hexdag.mcp_server import create_environment_pipelines_with_includes

result_json = create_environment_pipelines_with_includes(
    pipeline_name="my-agent",
    description="My agent",
    nodes=[...],
    staging_ports={...},
    prod_ports={...}
)

# Base file + environment configs that include base
# Good for: Complex pipelines, multiple environments, team projects
        """)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
