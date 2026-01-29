# hexDAG MCP Server - Quick Start Guide

This guide will walk you through setting up and using the hexDAG MCP server with Claude Desktop.

## Prerequisites

- hexDAG installed with MCP support
- Claude Desktop app installed
- Basic understanding of YAML

## Step 1: Install Dependencies

```bash
# Navigate to your hexDAG directory
cd /path/to/hexdag

# Install MCP dependencies
uv sync --extra mcp

# Verify installation
uv run python -c "from hexdag.mcp_server import mcp; print('✓ MCP server ready')"
```

You should see: `✓ MCP server ready`

## Step 2: Test the MCP Server Locally

Before configuring Claude Desktop, test that the server works:

```bash
# Start the MCP server (will run until you press Ctrl+C)
uv run python -m hexdag --mcp
```

You should see logs indicating the registry is bootstrapped with components. Press `Ctrl+C` to stop.

## Step 3: Configure Claude Desktop

1. **Locate your Claude Desktop config file:**
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Edit the config file** (create it if it doesn't exist):

```json
{
  "mcpServers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"],
      "cwd": "/Users/jankwapisz/Documents/Praca/Omniviser/hexdag",
      "env": {
        "HEXDAG_CONFIG_PATH": ""
      }
    }
  }
}
```

**Important**: Update the `"cwd"` field to point to your hexDAG installation directory!

3. **Save the file** and **restart Claude Desktop**

## Step 4: Verify MCP Server is Connected

1. Open Claude Desktop
2. Look for an indication that MCP tools are available (usually shown as a tools icon or in settings)
3. You can ask Claude: "What MCP tools do you have access to?"

Claude should respond with information about the hexDAG tools available.

## Step 5: Use hexDAG MCP Tools

Now you can ask Claude to help you build hexDAG pipelines! Here are some examples:

### Example 1: Discover Available Components

**You ask:**
```
What node types are available in hexDAG?
```

**Claude will:**
1. Call the `list_nodes()` MCP tool
2. Present you with a formatted list of all available node types (llm_node, agent_node, function_node, etc.)
3. Include descriptions for each node type

### Example 2: Build a Simple Pipeline

**You ask:**
```
Create a hexDAG pipeline that:
1. Takes a text input
2. Analyzes it with an LLM
3. Outputs the result

Use the mock LLM adapter for testing.
```

**Claude will:**
1. Use `list_nodes()` to check available node types
2. Use `list_adapters("llm")` to find LLM adapters
3. Use `build_yaml_pipeline_interactive()` to create the YAML
4. Use `validate_yaml_pipeline()` to ensure it's valid
5. Present you with the complete, validated YAML pipeline

**Example output:**
```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: text-analysis
  description: Analyze text with LLM
spec:
  ports:
    llm:
      adapter: mock_llm
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input}}"
        output_key: result
      dependencies: []
```

### Example 3: Get Component Details

**You ask:**
```
What parameters does the agent_node accept?
```

**Claude will:**
1. Call `get_component_schema("node", "agent_node", "core")`
2. Present you with detailed schema information including parameter types, defaults, and descriptions

### Example 4: Build a Complex Multi-Step Pipeline

**You ask:**
```
Create a research pipeline with:
- An agent that searches for information
- An LLM that summarizes findings
- A function that saves results to a file

Include retry policies and proper error handling.
```

**Claude will:**
1. Check available nodes, adapters, policies, and tools
2. Design the pipeline structure with dependencies
3. Add retry policies using `list_policies()`
4. Build and validate the complete YAML
5. Explain each step of the pipeline

## Step 6: Save and Run Your Pipeline

Once Claude generates a pipeline for you:

1. **Save it to a file:**
   ```bash
   # Save the YAML to a file
   cat > my_pipeline.yaml << 'EOF'
   <paste YAML here>
   EOF
   ```

2. **Run it with hexDAG:**
   ```python
   # In Python or a notebook
   from hexdag.core.pipeline_builder import YamlPipelineBuilder
   from hexdag.core.orchestration import Orchestrator

   # Load pipeline
   builder = YamlPipelineBuilder()
   graph, config = builder.build_from_yaml_file("my_pipeline.yaml")

   # Run it
   orchestrator = Orchestrator()
   result = await orchestrator.run(
       graph,
       initial_data={"input": "Your input data here"}
   )

   print(result)
   ```

## Advanced Usage

### Using Custom Plugins

If you have custom plugins in your `pyproject.toml`:

```toml
[tool.hexdag]
plugins = ["my_custom_plugin"]
```

The MCP server will automatically discover and expose them! Just ask Claude:
```
What custom components do I have available?
```

### Using a Different Config File

Set the `HEXDAG_CONFIG_PATH` in your Claude Desktop config:

```json
{
  "mcpServers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"],
      "cwd": "/path/to/hexdag",
      "env": {
        "HEXDAG_CONFIG_PATH": "/path/to/my/hexdag.toml"
      }
    }
  }
}
```

### Getting YAML Structure Documentation

**You ask:**
```
Explain the hexDAG YAML pipeline structure
```

**Claude will:**
- Call `explain_yaml_structure()`
- Present you with comprehensive documentation about YAML pipeline format

## Troubleshooting

### MCP Server Not Showing Up

1. **Check Claude Desktop logs:**
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

2. **Verify the server works locally:**
   ```bash
   uv run python -m hexdag --mcp
   ```

3. **Check the config file path is correct** - Make sure `"cwd"` points to your hexDAG directory

### Tools Not Working

1. **Ask Claude what tools it sees:**
   ```
   What MCP tools do you have access to?
   ```

2. **Check that hexDAG registry is loaded:**
   ```bash
   uv run python -c "
   from hexdag.core.bootstrap import ensure_bootstrapped
   from hexdag.core.registry import registry
   ensure_bootstrapped()
   print(f'Registry has {len(registry._components)} components')
   "
   ```

   Should output: `Registry has 41 components` (or more if you have plugins)

### Permission Errors

Make sure the `uv` command is accessible from Claude Desktop:

```bash
which uv
# Should output the path to uv, e.g., /Users/you/.local/bin/uv
```

If `uv` is not in your PATH, use the full path in your config:
```json
"command": "/Users/you/.local/bin/uv"
```

## Tips for Best Results

1. **Be Specific**: The more details you provide about your pipeline requirements, the better the result
2. **Iterate**: Ask Claude to modify or improve the generated pipeline
3. **Validate**: Claude will automatically validate pipelines, but you can ask it to double-check
4. **Ask Questions**: Claude can explain any part of the generated YAML or hexDAG concepts
5. **Use Examples**: Reference the examples in `examples/` directory for inspiration

## Next Steps

- Explore the [full MCP README](README.md) for advanced configuration
- Check out [hexDAG examples](../README.md) for pipeline patterns
- Read the [hexDAG documentation](../../docs/) for in-depth guides

---

**Happy pipeline building! 🚀**
