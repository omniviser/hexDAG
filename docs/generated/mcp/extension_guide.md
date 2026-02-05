# Extending hexDAG - Overview

## Extension Points

| Component | Purpose | Available |
|-----------|---------|-----------|
| **Adapter** | Connect to external services | 18 |
| **Node** | Custom processing logic | 9 |
| **Tool** | Agent-callable functions | 7 |

## Quick Reference

### Adapters
Use `get_custom_adapter_guide()` for full documentation.

### Nodes
Use `get_custom_node_guide()` for full documentation.

### Tools
Use `get_custom_tool_guide()` for full documentation.

## MCP Tools for Development

| Tool | Purpose |
|------|---------|
| `list_nodes()` | See available nodes |
| `list_adapters()` | See available adapters |
| `list_tools()` | See available tools |
| `get_component_schema()` | Get config schema |
| `validate_yaml_pipeline()` | Validate your YAML |
| `get_pipeline_schema()` | Get full JSON schema |
