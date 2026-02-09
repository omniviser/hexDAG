# HexDAG Configuration Schemas

This directory contains **dynamically generated** JSON Schema definitions for HexDAG configuration files, providing autocomplete, validation, and documentation for YAML pipelines and configuration.

> **⚠️ Important**: These schemas are auto-generated from the codebase. **Do not edit them manually!** Pre-commit hooks will reject any manual modifications. Instead, modify the source code (node factories, Pydantic models) and regenerate schemas using `scripts/generate_schemas.py`.

## 📚 For Library Users

If you're using hexDAG as a library in your project, you can enable IDE autocomplete and validation for your YAML pipeline files:

### Quick Setup

**1. Install VS Code YAML Extension:**
```bash
code --install-extension redhat.vscode-yaml
```

**2. Add to your project's `.vscode/settings.json`:**

Reference schemas from your hexDAG installation:

```json
{
  "yaml.schemas": {
    "node_modules/hexdag/schemas/pipeline-schema.json": [
      "*pipeline*.yaml",
      "*workflow*.yaml",
      "pipelines/**/*.yaml"
    ]
  }
}
```

Or use the package site-packages path (Python environment):

```bash
# Find your hexDAG installation path
python -c "import hexdag; import os; print(os.path.dirname(hexdag.__file__))"
```

```json
{
  "yaml.schemas": {
    "/path/to/site-packages/hexdag/schemas/pipeline-schema.json": ["*pipeline*.yaml"]
  }
}
```

**Alternative: Use remote URL (no installation required):**

```json
{
  "yaml.schemas": {
    "https://raw.githubusercontent.com/omniviser/hexdag/main/schemas/pipeline-schema.json": [
      "*pipeline*.yaml",
      "pipelines/**/*.yaml"
    ]
  }
}
```

**3. Start writing pipelines with full IntelliSense!** 🎉

Your YAML files will now have:
- ✅ Autocomplete for node types and parameters
- ✅ Real-time validation
- ✅ Documentation on hover
- ✅ Error detection

### Alternative: Inline Schema Reference

Add this comment at the top of your YAML files:
```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/omniviser/hexdag/main/schemas/pipeline-schema.json

apiVersion: v1
kind: Pipeline
metadata:
  name: my-pipeline
# ...
```

### For PyCharm/IntelliJ Users

1. Go to **Settings** → **Languages & Frameworks** → **Schemas and DTDs** → **JSON Schema Mappings**
2. Click **+** to add a new mapping:
   - **Name**: hexDAG Pipeline
   - **Schema URL**: `https://raw.githubusercontent.com/omniviser/hexdag/main/schemas/pipeline-schema.json`
   - **File pattern**: `*pipeline*.yaml`, `*workflow*.yaml`

---

## Available Schemas

### 1. Pipeline Schema (`pipeline-schema.json`)

Defines the structure for declarative YAML pipeline manifests. **This schema includes only the core builtin node types** (function, llm, agent, loop, conditional).

> **Note on Plugin Nodes**: Plugin-provided node types (e.g., `mypackage.nodes.CustomNode`) are validated at runtime by the resolver. The base schema accepts any node kind. Plugins can generate their own schemas for enhanced IDE support.

**Usage in YAML files:**

```yaml
# yaml-language-server: $schema=../schemas/pipeline-schema.json

apiVersion: v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Example pipeline
  version: "1.0"
spec:
  nodes:
    - kind: llm_node
      metadata:
        name: analyzer
      spec:
        prompt_template: "Analyze: {{input.text}}"
        dependencies: []
```

**Supported Node Types:**

- `function_node` - Execute Python functions
- `llm_node` - Language model interactions
- `agent_node` - ReAct agents with tool access
- `conditional_node` - Conditional branching
- `loop_node` - Iterative processing

**Key Properties:**

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `apiVersion` | string | No | API version (default: "v1") |
| `kind` | string | Yes | Must be "Pipeline" |
| `metadata` | object | Yes | Pipeline metadata |
| `metadata.name` | string | Yes | Unique pipeline identifier |
| `metadata.description` | string | No | Pipeline description |
| `metadata.author` | string | No | Pipeline author |
| `metadata.version` | string | No | Semantic version |
| `metadata.tags` | array | No | Categorization tags |
| `spec` | object | Yes | Pipeline specification |
| `spec.nodes` | array | Yes | List of pipeline nodes |
| `spec.input_schema` | object | No | JSON Schema for inputs |
| `spec.common_field_mappings` | object | No | Reusable field mappings |

### 2. Policy Schema (`policy-schema.json`)

Defines policy configuration for execution control and error handling.

**Policy Signals:**

- `proceed` - Continue normal execution
- `retry` - Retry the operation
- `skip` - Skip this operation
- `fallback` - Use fallback value/behavior
- `fail` - Fail the operation

**Policy Context Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `event` | object | Event that triggered evaluation |
| `dag_id` | string | DAG identifier |
| `node_id` | string\|null | Current node (if applicable) |
| `wave_index` | integer | Current wave index |
| `attempt` | integer | Attempt number (1-based) |
| `error` | object\|null | Exception details (if any) |
| `metadata` | object\|null | Additional context |

**Subscriber Types:**

- `core` - Core framework policies (strong reference)
- `plugin` - Plugin policies (strong reference)
- `user` - User-defined policies (weak reference)
- `temporary` - Temporary policies (weak reference)

### 3. HexDAG Config Schema (`hexdag-config-schema.json`)

Defines configuration for the `[tool.hexdag]` section in `pyproject.toml`.

**Usage in pyproject.toml:**

```toml
[tool.hexdag]
modules = [
    "hexdag.core.ports",
    "hexdag.builtin.nodes",
    "myapp.adapters",
]
plugins = [
    "hexdag.builtin.adapters.local",
    "hexdag-openai",
]
dev_mode = true

[tool.hexdag.logging]
level = "INFO"
format = "structured"
use_color = true
include_timestamp = true
output_file = "logs/hexdag.log"
```

**Configuration Properties:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `modules` | array | See schema | Module paths to load |
| `plugins` | array | See schema | Plugin packages to load |
| `dev_mode` | boolean | false | Development mode |
| `logging` | object | See below | Logging configuration |
| `settings` | object | {} | Custom settings |

**Logging Configuration:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `level` | string | "INFO" | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `format` | string | "structured" | console, json, structured, dual, rich |
| `output_file` | string\|null | null | Log file path |
| `use_color` | boolean | true | ANSI color codes |
| `include_timestamp` | boolean | true | Timestamp in output |
| `use_rich` | boolean | false | Rich library formatting |
| `dual_sink` | boolean | false | Console + JSON output |
| `enable_stdlib_bridge` | boolean | false | Intercept stdlib logging |
| `backtrace` | boolean | true | Debug backtraces (disable in production) |
| `diagnose` | boolean | true | Variable diagnostics (disable in production) |

## IDE Integration

### VS Code

1. Install the [YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)

2. Add schema associations to `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schemas/pipeline-schema.json": ["*.yaml", "pipelines/*.yaml"],
    "./schemas/hexdag-config-schema.json": ["pyproject.toml"]
  }
}
```

3. Or use inline schema reference in YAML files:

```yaml
# yaml-language-server: $schema=./schemas/pipeline-schema.json
```

### PyCharm / IntelliJ

1. Go to **Settings** → **Languages & Frameworks** → **Schemas and DTDs** → **JSON Schema Mappings**

2. Add mappings:
   - Schema: `schemas/pipeline-schema.json` → File pattern: `*.yaml`
   - Schema: `schemas/hexdag-config-schema.json` → File pattern: `pyproject.toml`

## Environment Variable Overrides

HexDAG logging configuration can be overridden with environment variables:

```bash
# Override log level
export HEXDAG_LOG_LEVEL=DEBUG

# Override format
export HEXDAG_LOG_FORMAT=rich

# Set log file
export HEXDAG_LOG_FILE=/var/log/hexdag/app.log

# Enable dual-sink mode
export HEXDAG_LOG_DUAL_SINK=true

# Enable Rich output
export HEXDAG_LOG_RICH=true

# Disable color
export HEXDAG_LOG_COLOR=false
```

Environment variables take precedence over `pyproject.toml` configuration.

## Validation

### CLI Validation

```bash
# Validate pipeline YAML
hexdag pipeline validate my-pipeline.yaml

# Validate with verbose output
hexdag pipeline validate my-pipeline.yaml --verbose
```

### Python API Validation

```python
from hexdag.core.pipeline_builder.yaml_validator import YamlValidator
import yaml

# Load and validate pipeline
with open("my-pipeline.yaml") as f:
    config = yaml.safe_load(f)

validator = YamlValidator()
report = validator.validate(config)

if report.is_valid:
    print("✓ Pipeline is valid")
else:
    for error in report.errors:
        print(f"✗ {error}")
```

## Schema Generation

### How It Works

Schemas are dynamically generated from the codebase:

1. **Pipeline Schema** - Generated from node factory schemas in the registry
2. **Policy Schema** - Generated from `PolicyContext`, `PolicyResponse`, and enum types
3. **HexDAG Config Schema** - Generated from `HexDAGConfig` and `LoggingConfig` Pydantic models

This ensures schemas stay in sync with the actual code and never become outdated.

### Regenerating Schemas

Run the schema generator manually:

```bash
uv run python scripts/generate_schemas.py
```

Or let pre-commit hooks handle it automatically when relevant files change:

```bash
# Schemas will be regenerated if you modify:
# - hexdag/builtin/nodes/ (node factories)
# - hexdag/core/config/models.py (config models)
# - hexdag/core/orchestration/policies/models.py (policy models)

git add hexdag/builtin/nodes/my_new_node.py
git commit -m "feat: add new node type"
# -> Schemas automatically regenerated
```

### Protection Against Manual Edits

Pre-commit hooks **block any manual modifications** to schema files:

```bash
# This will be rejected:
vim schemas/pipeline-schema.json  # Make manual edits
git add schemas/pipeline-schema.json
git commit -m "update schema"
# ✗ Check JSON schemas are auto-generated........Failed
# ERROR: Schema files are out of sync!

# Correct approach:
vim hexdag/builtin/nodes/my_node.py  # Modify source code
uv run python scripts/generate_schemas.py  # Regenerate
git add hexdag/builtin/nodes/my_node.py schemas/
git commit -m "feat: add new node with schema"
# ✓ All checks pass
```

The `check-schemas` pre-commit hook compares committed schemas against freshly generated ones to ensure they match exactly.

### When to Regenerate

Regenerate schemas when:

- ✅ Adding new node types
- ✅ Changing node parameters or validation
- ✅ Updating configuration models
- ✅ Modifying policy interfaces
- ✅ Before releases

### Plugin Schema Generation

Plugins can generate their own schemas for enhanced IDE support:

```python
from scripts.generate_schemas import generate_pipeline_schema
import json

# Generate schema including your plugin's node types
plugin_schema = generate_pipeline_schema()

# Save to your plugin directory
with open("my-plugin/schema.json", "w") as f:
    json.dump(plugin_schema, f, indent=2)
```

This creates a schema that includes your plugin's custom node types alongside the base pipeline structure.

### Schema Versioning

Schemas follow the HexDAG version.

**Core schema** (`pipeline-schema.json`): Contains the builtin node types
**Plugin schemas**: Each plugin can generate its own schema

Current schema version: **v0.3.0-a2**

## Examples

See the following directories for examples:

- `examples/` - Full pipeline examples
- `examples/manifests/` - Declarative YAML pipelines
- `tests/hexdag/pipeline_builder/` - Test fixtures with various configurations

## References

- [JSON Schema Specification](https://json-schema.org/)
- [YAML Language Server](https://github.com/redhat-developer/yaml-language-server)
- [HexDAG Documentation](https://dev.azure.com/omniviser/hexDAG)
