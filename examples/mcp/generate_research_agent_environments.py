"""Generate deep research agent environments using MCP tools.

This script uses the MCP environment management tools to create
dev/staging/prod versions of the deep research agent.
"""

import json
from pathlib import Path

from hexdag.mcp_server import create_environment_pipelines


def generate_research_agent_environments():
    """Generate deep research agent for all environments."""
    print("\n" + "=" * 80)
    print("Generating Deep Research Agent Environments")
    print("=" * 80)

    # Define shared pipeline logic (reasoning agent macro)
    # Note: We use generic tool names that work with both mock and real routers
    nodes = [
        {
            "kind": "macro_invocation",
            "name": "research_agent",
            "spec": {
                "macro": "core:reasoning_agent",
                "config": {
                    "main_prompt": (
                        "You are a deep research agent. "
                        "Research the following question thoroughly: {{research_question}}"
                    ),
                    "max_steps": 5,  # Reduced for dev/staging
                    "allowed_tools": [
                        "search",  # Generic tool name (works with mock)
                        "calculate",
                    ],
                    "tool_format": "mixed",
                },
            },
            "dependencies": [],
        }
    ]

    # Dev environment: Mock adapters (auto-generated)
    # No dev_ports needed - will use default mocks

    # Staging environment: Real APIs with cheaper models
    staging_ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {
                "api_key": "${STAGING_OPENAI_API_KEY}",
                "model": "gpt-4o-mini",  # Cheaper model for staging
                "temperature": 0.7,
            },
        },
        "tool_router": {
            "adapter": "plugin:mock_tool_router",  # Mock tools in staging
            "config": {"available_tools": ["search", "calculate"]},
        },
    }

    # Production environment: Real APIs with full models
    prod_ports = {
        "llm": {
            "adapter": "core:openai",
            "config": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-4o",  # Full model for production
                "temperature": 0.7,
            },
        },
        "tool_router": {
            "adapter": "core:function_tool_router",  # Real Tavily tools
            "config": {},
        },
    }

    print("\n📝 Generating environments...")
    print("  - Dev: Mock LLM + Mock Tools (no API keys)")
    print("  - Staging: GPT-4o-mini + Mock Tools (staging keys)")
    print("  - Production: GPT-4o + Real Tavily (production keys)")

    # Generate all environments
    result_json = create_environment_pipelines(
        pipeline_name="deep-research-agent",
        description="Deep research agent with web search capabilities",
        nodes=nodes,
        staging_ports=staging_ports,
        prod_ports=prod_ports,
    )

    result = json.loads(result_json)

    print(f"\n✅ Generated {len(result)} environment files")

    # Save files
    output_dir = Path(__file__).parent
    files_written = []

    for env, yaml_content in result.items():
        filename = f"deep_research_agent_{env}.yaml"
        filepath = output_dir / filename
        filepath.write_text(yaml_content)
        files_written.append(filename)
        print(f"   ✅ Wrote: {filename} ({len(yaml_content)} bytes)")

    print("\n" + "=" * 80)
    print("Files Created:")
    print("=" * 80)

    for filename in files_written:
        filepath = output_dir / filename
        print(f"\n📄 {filename}")
        print("-" * 80)

        # Show first few lines
        content = filepath.read_text()
        lines = content.split("\n")[:15]
        for line in lines:
            print(line)
        if len(content.split("\n")) > 15:
            print("...")

    print("\n" + "=" * 80)
    print("Environment Details:")
    print("=" * 80)

    print("\n🔧 Development Environment:")
    print("  File: deep_research_agent_dev.yaml")
    print("  LLM: plugin:mock_llm (no API key)")
    print("  Tools: plugin:mock_tool_router (no API key)")
    print("  Cost: $0.00 per run")
    print("  Speed: <1 second")
    print("  Use: Local testing, CI/CD, demos")

    print("\n🧪 Staging Environment:")
    print("  File: deep_research_agent_staging.yaml")
    print("  LLM: core:openai (gpt-4o-mini)")
    print("  Tools: plugin:mock_tool_router")
    print("  Cost: ~$0.01-0.05 per run")
    print("  Speed: 10-20 seconds")
    print("  Use: Pre-production testing, validation")

    print("\n🚀 Production Environment:")
    print("  File: deep_research_agent_prod.yaml")
    print("  LLM: core:openai (gpt-4o)")
    print("  Tools: core:function_tool_router (real Tavily)")
    print("  Cost: ~$0.10-0.30 per run")
    print("  Speed: 20-40 seconds")
    print("  Use: Production deployments")

    print("\n" + "=" * 80)
    print("Running the Agents:")
    print("=" * 80)

    print("\n# Development (no API keys needed)")
    print("python examples/mcp/run_deep_research_agent.py \\")
    print("  --env dev \\")
    print("  --pipeline deep_research_agent_dev.yaml \\")
    print('  --question "What are the latest AI trends?"')

    print("\n# Staging (requires staging OpenAI key)")
    print("export STAGING_OPENAI_API_KEY=sk-...")
    print("python examples/mcp/run_deep_research_agent.py \\")
    print("  --env staging \\")
    print("  --pipeline deep_research_agent_staging.yaml \\")
    print('  --question "What are the latest AI trends?"')

    print("\n# Production (requires OpenAI + Tavily keys)")
    print("export OPENAI_API_KEY=sk-...")
    print("export TAVILY_API_KEY=tvly-...")
    print("python examples/mcp/run_deep_research_agent.py \\")
    print("  --env prod \\")
    print("  --pipeline deep_research_agent_prod.yaml \\")
    print('  --question "What are the latest AI trends?"')

    print("\n" + "=" * 80)
    print("✅ Environment Generation Complete!")
    print("=" * 80)

    return result


def compare_with_original():
    """Compare new generated files with original manual files."""
    print("\n" + "=" * 80)
    print("Comparison: Generated vs Original")
    print("=" * 80)

    output_dir = Path(__file__).parent

    # Check if original files exist
    original_dev = output_dir / "deep_research_agent_dev.yaml"
    original_prod = output_dir / "deep_research_agent.yaml"

    if original_dev.exists():
        original_dev_size = len(original_dev.read_text())
        print(f"\n📄 Original dev.yaml: {original_dev_size} bytes")

    generated_dev = output_dir / "deep_research_agent_dev.yaml"
    if generated_dev.exists():
        generated_dev_size = len(generated_dev.read_text())
        print(f"📄 Generated dev.yaml: {generated_dev_size} bytes")

    if original_prod.exists():
        original_prod_size = len(original_prod.read_text())
        print(f"\n📄 Original prod.yaml: {original_prod_size} bytes")

    generated_prod = output_dir / "deep_research_agent_prod.yaml"
    if generated_prod.exists():
        generated_prod_size = len(generated_prod.read_text())
        print(f"📄 Generated prod.yaml: {generated_prod_size} bytes")

    print("\n✅ Generated files are ready!")
    print("   You can now use these instead of manually maintaining multiple versions")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("🎬 Deep Research Agent Environment Generator")
    print("=" * 80)
    print("\nThis script uses MCP tools to generate:")
    print("  1. Development environment (mock adapters, no API keys)")
    print("  2. Staging environment (cheaper models, partial mocks)")
    print("  3. Production environment (full models, real APIs)")

    try:
        # Generate all environments
        generate_research_agent_environments()

        # Compare with original if exists
        compare_with_original()

        print("\n" + "=" * 80)
        print("✅ Success!")
        print("=" * 80)
        print("\n💡 Next Steps:")
        print("  1. Review generated YAML files")
        print("  2. Test dev environment: python examples/mcp/run_dev_agent.py")
        print("  3. Configure staging/prod credentials")
        print("  4. Deploy to your environments")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
