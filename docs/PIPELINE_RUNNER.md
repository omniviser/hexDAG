# PipelineRunner

**The recommended entry point for running YAML pipelines.**

`PipelineRunner` eliminates the 15-30 lines of boilerplate previously required to run a pipeline: parsing YAML, loading secrets, instantiating ports, creating an orchestrator, and executing the graph. It replaces all of that with a single method call.

```python
from hexdag import PipelineRunner

runner = PipelineRunner()
result = await runner.run("pipeline.yaml", input_data={"query": "hello"})
```

## Table of Contents

- [Quick Start](#quick-start)
- [Constructor Parameters](#constructor-parameters)
- [Methods](#methods)
  - [run()](#run)
  - [run_from_string()](#run_from_string)
  - [validate()](#validate)
- [Secret Management](#secret-management)
  - [Constructor-Based Secrets](#constructor-based-secrets)
  - [YAML-Declared Secrets](#yaml-declared-secrets)
  - [Secret Caching](#secret-caching)
- [Port Overrides](#port-overrides)
- [Environment Variables](#environment-variables)
- [Multi-Environment YAML](#multi-environment-yaml)
- [CLI Usage](#cli-usage)
- [Lifecycle Hooks](#lifecycle-hooks)
- [Error Handling](#error-handling)
- [Migration from Orchestrator](#migration-from-orchestrator)

---

## Quick Start

### Minimal — run a YAML file

```python
import asyncio
from hexdag import PipelineRunner

async def main():
    runner = PipelineRunner()
    result = await runner.run("my-pipeline.yaml")
    print(result)

asyncio.run(main())
```

### With secrets from Azure KeyVault

```python
from hexdag import PipelineRunner
from hexdag_plugins.azure import AzureKeyVaultAdapter

runner = PipelineRunner(
    secrets_provider=AzureKeyVaultAdapter(
        vault_url="https://my-vault.vault.azure.net"
    ),
    secret_keys=["OPENAI-API-KEY", "DB-PASSWORD"],
)
result = await runner.run("pipeline.yaml", input_data={"query": "hello"})
```

### With mock ports for testing

```python
from hexdag import PipelineRunner, MockLLM

runner = PipelineRunner(
    port_overrides={"llm": MockLLM(responses="test response")}
)
result = await runner.run_from_string(yaml_content)
```

### Dry-run validation

```python
runner = PipelineRunner()
issues = await runner.validate(pipeline_path="pipeline.yaml")
if issues:
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Pipeline is valid")
```

---

## Constructor Parameters

```python
PipelineRunner(
    *,
    port_overrides: dict[str, Any] | None = None,
    secrets_provider: SecretPort | None = None,
    secret_keys: list[str] | None = None,
    max_concurrent_nodes: int = 10,
    strict_validation: bool = False,
    default_node_timeout: float | None = None,
    pre_hook_config: HookConfig | None = None,
    post_hook_config: PostDagHookConfig | None = None,
    base_path: Path | None = None,
    environment: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port_overrides` | `dict[str, Any] \| None` | `None` | Runtime port overrides. Merges with YAML-declared ports; overrides win on name collision. |
| `secrets_provider` | `SecretPort \| None` | `None` | Secret adapter for pre-instantiation secret loading. Overrides any `secret` port in YAML. |
| `secret_keys` | `list[str] \| None` | `None` | Specific secret keys to load. If `None`, loads all available secrets. |
| `max_concurrent_nodes` | `int` | `10` | Maximum nodes to execute concurrently. |
| `strict_validation` | `bool` | `False` | Raise on schema validation failure. |
| `default_node_timeout` | `float \| None` | `None` | Default per-node timeout in seconds. |
| `pre_hook_config` | `HookConfig \| None` | `None` | Pre-DAG hook configuration (health checks, custom hooks). |
| `post_hook_config` | `PostDagHookConfig \| None` | `None` | Post-DAG hook configuration (cleanup, checkpoints). |
| `base_path` | `Path \| None` | `None` | Base path for resolving `!include` directives in YAML. |
| `environment` | `str \| None` | `None` | Default environment for multi-document YAML selection. |

All parameters are keyword-only.

---

## Methods

### `run()`

Run a YAML pipeline from a file.

```python
async def run(
    self,
    pipeline_path: str | Path,
    input_data: Any = None,
    *,
    environment: str | None = None,
) -> dict[str, Any]
```

**Parameters:**
- `pipeline_path` — Path to the YAML pipeline file.
- `input_data` — Initial input data passed to the pipeline.
- `environment` — Environment override for multi-document YAML (overrides constructor `environment`).

**Returns:** `dict[str, Any]` — Node results keyed by node name.

**Raises:** `PipelineRunnerError` if the file is not found, env vars are missing, or execution fails.

```python
result = await runner.run("pipelines/analysis.yaml", input_data={"text": "..."})
print(result["summarizer"])  # Output of the 'summarizer' node
```

### `run_from_string()`

Run a YAML pipeline from an inline string.

```python
async def run_from_string(
    self,
    yaml_content: str,
    input_data: Any = None,
    *,
    environment: str | None = None,
) -> dict[str, Any]
```

Identical to `run()` but accepts YAML content directly. Useful for testing, notebooks, and dynamically generated pipelines.

```python
yaml = """
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: quick-test
spec:
  nodes:
    - kind: data_node
      metadata:
        name: greet
      spec:
        output:
          value: "hello world"
      dependencies: []
"""
result = await runner.run_from_string(yaml)
```

### `validate()`

Validate a pipeline without executing it (dry-run).

```python
async def validate(
    self,
    pipeline_path: str | Path | None = None,
    yaml_content: str | None = None,
    *,
    environment: str | None = None,
) -> list[str]
```

Checks YAML parsing, DAG validity, environment variable availability, and port instantiation. Returns a list of issue strings — an empty list means the pipeline is valid.

**Provide exactly one of** `pipeline_path` or `yaml_content`.

```python
issues = await runner.validate(yaml_content=my_yaml)
# ['Missing env var: OPENAI_API_KEY', 'Port instantiation error: ...']
```

---

## Secret Management

PipelineRunner loads secrets into `os.environ` **before** port adapters are instantiated. This means `${OPENAI_API_KEY}` in your YAML port configs resolves correctly at adapter creation time.

### Constructor-Based Secrets

Pass a `SecretPort` adapter directly:

```python
from hexdag_plugins.azure import AzureKeyVaultAdapter

runner = PipelineRunner(
    secrets_provider=AzureKeyVaultAdapter(
        vault_url="https://my-vault.vault.azure.net"
    ),
    secret_keys=["OPENAI-API-KEY", "DB-PASSWORD"],
)
```

Key names are normalised: hyphens become underscores and the result is upper-cased (`OPENAI-API-KEY` becomes `OPENAI_API_KEY` in `os.environ`).

### YAML-Declared Secrets

If your pipeline YAML declares a `secret` port, PipelineRunner auto-detects it:

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: production-pipeline
spec:
  ports:
    secret:
      adapter: hexdag_plugins.azure.AzureKeyVaultAdapter
      config:
        vault_url: "https://my-vault.vault.azure.net"
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: "${OPENAI_API_KEY}"
```

```python
# No secrets_provider needed — auto-detected from YAML
runner = PipelineRunner()
result = await runner.run("production-pipeline.yaml")
```

The constructor `secrets_provider` overrides any YAML-declared `secret` port.

### Secret Caching

PipelineRunner caches whether secrets have been loaded via an internal `_secrets_loaded` flag. On subsequent `run()` calls, `load_to_environ()` is skipped entirely. This avoids redundant vault calls when reusing a runner instance.

To force a reload, create a new `PipelineRunner` instance.

---

## Port Overrides

Override YAML-declared ports at runtime. Common for testing or swapping adapters without changing YAML:

```python
from hexdag import PipelineRunner, MockLLM
from hexdag.builtin.adapters.memory import InMemoryMemory

runner = PipelineRunner(
    port_overrides={
        "llm": MockLLM(responses="mocked response"),
        "memory": InMemoryMemory(),
    }
)
```

Overrides are merged with YAML-declared ports. If both YAML and `port_overrides` define the same port name, the override wins.

---

## Environment Variables

YAML port configs can reference environment variables with `${VAR}` or `${VAR:default}` syntax:

```yaml
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: "${OPENAI_API_KEY}"
        model: "${LLM_MODEL:gpt-4}"
```

PipelineRunner performs a **pre-flight check** before port instantiation: it scans all port configs for `${VAR}` patterns without a `:default` fallback and verifies they exist in `os.environ`. If any are missing, it raises a single `PipelineRunnerError` listing **all** missing variables at once.

Variables matching secret patterns (`*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL`) are deferred through the YAML build phase and resolved at adapter instantiation time (Phase 3b).

---

## Multi-Environment YAML

Multi-document YAML files can define different configurations per environment. Select which environment to use via the `environment` parameter:

```yaml
# pipeline.yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  namespace: development
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.mock.MockLLM
      config:
        responses: "dev response"
  nodes: [...]
---
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  namespace: production
spec:
  ports:
    llm:
      adapter: hexdag.builtin.adapters.openai.OpenAIAdapter
      config:
        api_key: "${OPENAI_API_KEY}"
  nodes: [...]
```

```python
# Select environment at construction time (applies to all runs)
runner = PipelineRunner(environment="production")

# Or per-run
result = await runner.run("pipeline.yaml", environment="development")
```

---

## CLI Usage

The `hexdag pipeline run` command uses PipelineRunner internally:

```bash
# Basic execution
hexdag pipeline run pipeline.yaml

# With input data
hexdag pipeline run pipeline.yaml -i '{"query": "hello"}'

# With input from file
hexdag pipeline run pipeline.yaml -f inputs.json

# Select environment
hexdag pipeline run pipeline.yaml -e production

# Concurrency and timeout
hexdag pipeline run pipeline.yaml --max-concurrent 5 --timeout 60

# Dry-run validation (no execution)
hexdag pipeline run pipeline.yaml --dry-run

# Save output to file
hexdag pipeline run pipeline.yaml -o results.json

# Verbose output
hexdag pipeline run pipeline.yaml -v
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input data as JSON string |
| `--input-file` | `-f` | Input data from JSON file |
| `--output` | `-o` | Save output to JSON file |
| `--verbose` | `-v` | Show detailed execution information |
| `--env` | `-e` | Pipeline environment (multi-document YAML) |
| `--max-concurrent` | | Maximum concurrent nodes (default: 10) |
| `--timeout` | | Default node timeout in seconds |
| `--dry-run` | | Validate pipeline without executing |

---

## Lifecycle Hooks

Configure pre-DAG and post-DAG hooks for health checks, cleanup, and custom logic:

```python
from hexdag.core.orchestration.components.lifecycle_manager import (
    HookConfig,
    PostDagHookConfig,
)

runner = PipelineRunner(
    pre_hook_config=HookConfig(
        health_check_ports=["llm", "database"],
        custom_hooks=[my_pre_hook],
    ),
    post_hook_config=PostDagHookConfig(
        cleanup_hooks=[my_cleanup_hook],
    ),
)
```

---

## Error Handling

PipelineRunner raises `PipelineRunnerError` (inherits from `HexDAGError`) for runner-specific failures:

```python
from hexdag.core.pipeline_runner import PipelineRunnerError

try:
    result = await runner.run("pipeline.yaml")
except PipelineRunnerError as e:
    print(f"Runner error: {e}")
```

Common error scenarios:

| Scenario | Error |
|----------|-------|
| YAML file not found | `PipelineRunnerError("Pipeline file not found: ...")` |
| Missing env vars | `PipelineRunnerError("Missing environment variables required by pipeline ports: VAR1, VAR2. ...")` |
| Invalid YAML | Propagated from `YamlPipelineBuilderError` |
| Port instantiation failure | Propagated from `ComponentInstantiationError` |

---

## Migration from Orchestrator

### Before (15-30 lines)

```python
from hexdag.core.pipeline_builder import YamlPipelineBuilder
from hexdag.core.orchestration.orchestrator import Orchestrator
from hexdag.core.orchestration.orchestrator_factory import OrchestratorFactory
from hexdag_plugins.azure import AzureKeyVaultAdapter

# 1. Load secrets manually
vault = AzureKeyVaultAdapter(vault_url="https://my-vault.vault.azure.net")
await vault.load_to_environ(keys=["OPENAI-API-KEY"])

# 2. Build pipeline
builder = YamlPipelineBuilder()
graph, config = builder.build_from_yaml_file("pipeline.yaml")

# 3. Create orchestrator with ports
factory = OrchestratorFactory()
orchestrator = factory.create_orchestrator(
    pipeline_config=config,
    max_concurrent_nodes=10,
)

# 4. Execute
result = await orchestrator.run(graph, {"query": "hello"})
```

### After (3 lines)

```python
from hexdag import PipelineRunner
from hexdag_plugins.azure import AzureKeyVaultAdapter

runner = PipelineRunner(
    secrets_provider=AzureKeyVaultAdapter(
        vault_url="https://my-vault.vault.azure.net"
    ),
    secret_keys=["OPENAI-API-KEY"],
)
result = await runner.run("pipeline.yaml", input_data={"query": "hello"})
```

### Deprecation Notes

The following top-level imports from `hexdag` now emit `DeprecationWarning`:

| Deprecated Import | Replacement |
|-------------------|-------------|
| `from hexdag import Orchestrator` | `PipelineRunner` or `from hexdag.core.orchestration.orchestrator import Orchestrator` |
| `from hexdag import YamlPipelineBuilder` | `PipelineRunner` or `from hexdag.core.pipeline_builder.yaml_builder import YamlPipelineBuilder` |
| `from hexdag import DirectedGraph` | `from hexdag.core.domain import DirectedGraph` |
| `from hexdag import NodeSpec` | `from hexdag.core.domain import NodeSpec` |
| `from hexdag import resolve` | `from hexdag.core.resolver import resolve` |

These continue to work but will be removed in a future release. Direct submodule imports remain stable.

---

## Execution Lifecycle

PipelineRunner orchestrates five internal steps on every `run()` call:

```
1. Build        YAML string/file  -->  (DirectedGraph, PipelineConfig)
                via YamlPipelineBuilder

2. Load secrets  SecretPort.load_to_environ()  -->  os.environ
                 (skipped if already cached or no provider)

3. Validate      Scan port configs for ${VAR} patterns
                 Fail-fast with all missing vars listed

4. Instantiate   OrchestratorFactory.create_orchestrator()
                 Auto-creates adapters, merges port_overrides

5. Execute       Orchestrator.run(graph, input_data)
                 Returns dict[str, Any] of node results
```

---

## Related Documentation

- [Core Concepts](concepts.md) — DAGs, NodeSpec, Orchestrator fundamentals
- [YAML Pipelines Architecture](YAML_PIPELINES_ARCHITECTURE.md) — Multi-phase rendering pipeline
- [Plugin System](PLUGIN_SYSTEM.md) — Custom adapters and node types
- [CLI Reference](CLI_REFERENCE.md) — Full CLI documentation
- [Implementation Guide](IMPLEMENTATION_GUIDE.md) — Production deployment patterns
