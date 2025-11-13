# hexDAG MCP Server Configuration

This directory contains example configurations for integrating the hexDAG MCP server with various LLM-powered editors.

## What is MCP?

Model Context Protocol (MCP) is an open protocol that enables LLM applications to connect with external tools and data sources. The hexDAG MCP server exposes hexDAG's pipeline building capabilities as MCP tools, making it easier for LLMs to construct valid YAML pipelines.

## Features

The hexDAG MCP server provides:

- **Dynamic Component Discovery**: Automatically lists all registered nodes, adapters, tools, macros, and policies
- **YAML Pipeline Building**: Interactive pipeline construction with validation
- **Schema Inspection**: Get detailed schemas for any component
- **Registry-Aware**: Reflects custom plugins loaded from your `pyproject.toml` or `hexdag.toml`

## Installation

```bash
# Development (inside hexDAG repository)
uv sync --extra mcp

# Production (install from PyPI)
pip install "hexdag[mcp]"

# Or with uv
uv pip install "hexdag[mcp]"
```

## Configuration

### Claude Desktop

1. Locate your Claude Desktop config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add the hexDAG MCP server configuration:

```json
{
  "mcpServers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"],
      "env": {
        "HEXDAG_CONFIG_PATH": ""
      }
    }
  }
}
```

3. Restart Claude Desktop

### Cursor

1. Create or edit `.cursor/mcp.json` in your project root:

```json
{
  "mcp": {
    "servers": {
      "hexdag": {
        "command": "uv",
        "args": ["run", "python", "-m", "hexdag", "--mcp"],
        "env": {
          "HEXDAG_CONFIG_PATH": ""
        }
      }
    }
  }
}
```

2. Restart Cursor

### VSCode (with MCP extension)

Install the MCP extension and add to your workspace settings:

```json
{
  "mcp.servers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"],
      "env": {
        "HEXDAG_CONFIG_PATH": ""
      }
    }
  }
}
```

## Configuration Options

### Using a Specific hexDAG Config

Set `HEXDAG_CONFIG_PATH` to point to your hexDAG configuration:

```json
{
  "env": {
    "HEXDAG_CONFIG_PATH": "/path/to/your/hexdag.toml"
  }
}
```

### Auto-Discovery (Default)

Leave `HEXDAG_CONFIG_PATH` empty to auto-discover configuration from:
1. Current working directory's `hexdag.toml`
2. Current working directory's `pyproject.toml` (with `[tool.hexdag]`)
3. Parent directories (walks up tree)
4. Defaults (core + builtin components only)

### Using Python Interpreter Directly

If you don't use `uv`, replace the command:

```json
{
  "command": "python",
  "args": ["-m", "hexdag", "--mcp"]
}
```

## Available MCP Tools

Once configured, the following tools are available to the LLM:

### Component Discovery
- `list_nodes()` - List all node types
- `list_adapters(port_type?)` - List adapters by port
- `list_tools(namespace?)` - List registered tools
- `list_macros()` - List available macros
- `list_policies()` - List execution policies
- `get_component_schema(type, name, namespace)` - Get component schema

### YAML Pipeline Building
- `validate_yaml_pipeline(yaml)` - Validate pipeline configuration
- `build_yaml_pipeline_interactive(name, description, nodes, ports)` - Build complete pipeline
- `generate_pipeline_template(name, description, node_types)` - Generate quick template
- `explain_yaml_structure()` - Get YAML structure documentation

### Environment Management (New!)
- `create_environment_pipelines(name, description, nodes, dev_ports?, staging_ports?, prod_ports?)` - Generate standalone dev/staging/prod YAMLs
- `create_environment_pipelines_with_includes(name, description, nodes, dev_ports?, staging_ports?, prod_ports?)` - Generate base + environment includes (DRY pattern)

See [ENVIRONMENT_MANAGEMENT.md](ENVIRONMENT_MANAGEMENT.md) for complete guide.

## Usage Examples

### Example 1: Deep Research Agent with Tavily

See [DEEP_RESEARCH_AGENT.md](DEEP_RESEARCH_AGENT.md) for a complete production-ready example:

```
Build me a deep research agent with Tavily web search
```

This creates a multi-step reasoning agent that:
- Uses Tavily's AI-powered web search
- Conducts comprehensive research with source verification
- Synthesizes findings from multiple sources
- Provides well-sourced, comprehensive answers

**Production Files:**
- [deep_research_agent.yaml](deep_research_agent.yaml) - Production YAML pipeline
- [tavily_adapter.py](tavily_adapter.py) - Tavily web search tools
- [run_deep_research_agent.py](run_deep_research_agent.py) - Production runner
- [DEEP_RESEARCH_AGENT.md](DEEP_RESEARCH_AGENT.md) - Complete documentation

**Dev Environment (No API Keys Required):**
- [deep_research_agent_dev.yaml](deep_research_agent_dev.yaml) - Dev pipeline with mock adapters
- [run_dev_agent.py](run_dev_agent.py) - Dev runner (instant, $0 cost)
- [DEV_ENVIRONMENT.md](DEV_ENVIRONMENT.md) - Dev environment guide
- [MOCK_TESTING.md](MOCK_TESTING.md) - Mock adapter patterns

**Quick Start:**

```bash
# Dev mode (no API keys needed)
python examples/mcp/run_dev_agent.py

# Production mode (requires OpenAI + Tavily keys)
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
python examples/mcp/run_deep_research_agent.py
```

### Example 2: Ask Claude to Build a Pipeline

```
Create a hexDAG pipeline that:
1. Takes an input document
2. Analyzes it with GPT-4
3. Stores the results in a database
```

Claude will use the MCP tools to:
1. List available nodes (`list_nodes()`)
2. Check adapter options (`list_adapters()`)
3. Build the YAML (`build_yaml_pipeline_interactive()`)
4. Validate it (`validate_yaml_pipeline()`)

### Example 3: Explore Available Components

```
What node types are available in hexDAG?
```

Claude will call `list_nodes()` and present the results.

### Example 4: Get Component Schemas

```
What parameters does the llm_node take?
```

Claude will call `get_component_schema("node", "llm_node", "core")`.

## Development Mode

For testing the MCP server locally:

```bash
# Start in development mode (shows verbose logging)
uv run mcp dev hexdag/mcp_server.py

# Or run directly with uv
uv run python -m hexdag --mcp
```

## Troubleshooting

### MCP Server Not Starting

1. Check that `hexdag[mcp]` is installed:
   ```bash
   uv pip list | grep mcp
   ```

2. Test the server directly:
   ```bash
   uv run python -m hexdag --mcp
   ```

3. Check logs in Claude Desktop:
   - **macOS**: `~/Library/Logs/Claude/mcp*.log`
   - **Windows**: `%APPDATA%\Claude\logs\mcp*.log`

### Components Not Showing Up

1. Verify your hexDAG config is loaded:
   ```bash
   export HEXDAG_CONFIG_PATH=/path/to/hexdag.toml
   python -c "from hexdag.core.bootstrap import ensure_bootstrapped; from hexdag.core.registry import registry; ensure_bootstrapped(); print(len(registry._components))"
   ```

2. Check that plugins are listed in your config:
   ```toml
   [tool.hexdag]
   plugins = ["my_custom_plugin"]
   ```

### Permission Issues

Ensure the Python environment is accessible:
```bash
which python
which uv
```

## Learn More

- [hexDAG Documentation](https://hexdag.ai/docs)
- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [Claude Desktop Configuration](https://docs.anthropic.com/claude/docs/claude-desktop)
