# JSON Schema Setup Guide

This guide explains how to automatically configure JSON schemas in your IDE for hexDAG development.

## 🎯 What This Does

Enables **IntelliSense**, **autocomplete**, and **validation** for:
- ✅ **Pipeline YAML files** - Full autocomplete for all node types, parameters, and structure
- ✅ **Policy YAML files** - Validation for policy configurations
- ✅ **pyproject.toml** - Schema validation for `[tool.hexdag]` configuration

## 📦 Available Schemas

| Schema | Description | File Pattern |
|--------|-------------|--------------|
| `pipeline-schema.json` | Complete pipeline structure with all node types | `*pipeline*.yaml`, `pipelines/**/*.yaml` |
| `policy-schema.json` | Policy configuration schema | `*policy*.yaml`, `policies/**/*.yaml` |
| `hexdag-config-schema.json` | Configuration for pyproject.toml | `pyproject.toml` |

## 🚀 Automatic Setup (Recommended)

### VS Code (Already Configured!)

The schemas are **automatically configured** in `.vscode/settings.json`. Just install the recommended extensions:

1. Open VS Code
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Show Recommended Extensions"
4. Install all recommended extensions (especially `redhat.vscode-yaml`)

**Or install manually:**
```bash
code --install-extension redhat.vscode-yaml
code --install-extension tamasfe.even-better-toml
```

### JetBrains IDEs (PyCharm, IntelliJ)

1. Open Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings
2. Add mappings:

**Pipeline Schema:**
- Name: `hexDAG Pipeline`
- Schema file: `schemas/pipeline-schema.json`
- File patterns: `*pipeline*.yaml`, `pipelines/**/*.yaml`, `examples/**/*.yaml`

**Policy Schema:**
- Name: `hexDAG Policy`
- Schema file: `schemas/policy-schema.json`
- File patterns: `*policy*.yaml`, `policies/**/*.yaml`

## 🔧 Manual Configuration

### VS Code Settings

Add to your `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schemas/pipeline-schema.json": [
      "*pipeline*.yaml",
      "pipelines/**/*.yaml",
      "examples/**/*.yaml"
    ],
    "./schemas/policy-schema.json": [
      "*policy*.yaml",
      "policies/**/*.yaml"
    ]
  }
}
```

### Global User Settings

To use these schemas across **all your projects**, add to your global VS Code settings:

```json
{
  "yaml.schemas": {
    "https://raw.githubusercontent.com/your-org/hexdag/main/schemas/pipeline-schema.json": [
      "*hexdag-pipeline*.yaml",
      "*hexdag-workflow*.yaml"
    ]
  }
}
```

## 🎨 What You Get

### 1. IntelliSense & Autocomplete

When editing YAML files:
- Type `kind:` → See available node types (llm_node, agent_node, etc.)
- Type `spec:` → See parameters for that node type
- Hover over properties → See documentation

### 2. Real-time Validation

- ❌ Red squiggles for invalid properties
- ⚠️ Yellow warnings for deprecated fields
- ✅ Green checkmarks for valid structure

### 3. Documentation on Hover

Hover over any property to see:
- Description
- Type information
- Default values
- Examples

## 🧪 Testing Schema Configuration

Create a test file `test-pipeline.yaml`:

```yaml
apiVersion: v1
kind: Pipeline
metadata:
  name: test
spec:
  nodes:
    - kind: llm_node  # ← Should show autocomplete here
      metadata:
        name: my_llm
      spec:
        # ← Should show available parameters here
```

If you see autocomplete when typing, it's working! 🎉

## 🔄 Updating Schemas

Schemas are **automatically regenerated** by pre-commit hooks when you modify node types or configurations.

To manually regenerate:
```bash
uv run python scripts/generate_schemas.py
```

## 🌐 Schema URLs for Remote Access

If you want to reference schemas from external projects:

```
https://raw.githubusercontent.com/your-org/hexdag/main/schemas/pipeline-schema.json
https://raw.githubusercontent.com/your-org/hexdag/main/schemas/policy-schema.json
https://raw.githubusercontent.com/your-org/hexdag/main/schemas/hexdag-config-schema.json
```

## 📚 Additional Resources

- [JSON Schema Specification](http://json-schema.org/)
- [VS Code YAML Extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
- [Schema Store](https://www.schemastore.org/) - Public schema registry

## ❓ Troubleshooting

### Schema not loading?

1. **Reload VS Code**: Press `Cmd+Shift+P` → "Reload Window"
2. **Check extension**: Ensure `redhat.vscode-yaml` is installed and enabled
3. **Check file path**: Schema paths are relative to workspace root
4. **Check YAML extension settings**: Open Output panel → Select "YAML Support"

### Autocomplete not working?

1. File must match pattern (e.g., `*pipeline*.yaml`)
2. File must be valid YAML syntax
3. Try typing after a newline with proper indentation

### Wrong schema applied?

File pattern matching is order-sensitive. More specific patterns should come first in settings.

## 🎯 Pro Tips

1. **Use schema $id in YAML files** for explicit schema selection:
   ```yaml
   # yaml-language-server: $schema=./schemas/pipeline-schema.json
   ```

2. **Disable schema for specific files**:
   ```yaml
   # yaml-language-server: $schema=null
   ```

3. **Custom schema per file**:
   ```yaml
   # yaml-language-server: $schema=https://example.com/custom-schema.json
   ```

---

**Need help?** Open an issue or check the [VS Code YAML documentation](https://github.com/redhat-developer/vscode-yaml).
