# YAML Macro System Overhaul — Phased Plan

## Context

The YAML macro building system has 9 infrastructure bugs plus 4 major capability gaps that prevent macros from being fully YAML-first. This plan addresses everything in 4 incremental phases, each independently shippable and testable.

---

## Phase 1: Fix Building Infrastructure Bugs ✅ DONE

*Goal: Make the existing macro system correct and robust.*

### 1a. Defensive entry node collection in `_merge_subgraph()`
**File:** `hexdag/compiler/plugins/macro_entity.py` (lines 134-154)

Collect entry nodes into a set BEFORE iterating/mutating:
```python
entry_nodes = {node.name for node in subgraph if not subgraph.get_dependencies(node.name)}
for node in subgraph:
    if node.name in entry_nodes:
        graph += node.after(*external_deps)
    else:
        graph += node
```

### 1b. Fix default `None` silently omitted
**File:** `hexdag/kernel/yaml_macro.py` (lines 97-121, 356-358)

Add `_has_explicit_default` tracking to `YamlMacroParameterSpec` using a `model_validator`:
```python
@model_validator(mode="after")
def _check_default(self) -> Self:
    object.__setattr__(self, "_has_explicit_default", "default" in self.model_fields_set)
    return self
```
Change condition from `elif param.default is not None` to `elif param._has_explicit_default`.

### 1c. Add type validation for YAML macro parameters
**File:** `hexdag/kernel/yaml_macro.py` (lines 331-360)

Add `_TYPE_MAP` dict and isinstance check in `_validate_and_normalize_inputs()` after enum validation. Support union types via `param.type.split("|")`. Skip for `Any`.

### 1d. Protect structural fields in `DynamicYamlMacro`
**File:** `hexdag/compiler/plugins/macro_definition.py` (lines 111-121)

Filter kwargs to exclude `{"macro_name", "macro_description", "parameters", "nodes", "outputs"}`.

### 1e. Add deprecation warning on fallback builder path
**File:** `hexdag/kernel/yaml_macro.py` (lines 508-528)

Add `warnings.warn()` when `node_builder is None`.

### 1f. Fix config/inputs double-passing
**File:** `hexdag/compiler/plugins/macro_entity.py` (lines 45-73)

- Raise `YamlPipelineBuilderError` if `config` and `inputs` keys overlap
- Pass only `inputs` to `expand()`, not merged `{**config_params, **inputs}`

### 1g. Fix `previous_node_id` tracking across macro boundaries (CRITICAL)
**Files:** `hexdag/compiler/plugins/macro_entity.py`, `hexdag/compiler/yaml_builder.py`

- Add `_last_exit_nodes: list[str]` to `MacroEntityPlugin`
- After `_merge_subgraph()`, set `self._last_exit_nodes = subgraph.get_exit_nodes()`
  - `DirectedGraph.get_exit_nodes()` already exists at `hexdag/kernel/domain/dag.py:1158`
- In `yaml_builder.py` `_build_graph()`:
  - Change `previous_node_id: str | None` to `previous_node_ids: list[str]` (supports multiple exit nodes)
  - After macro plugin handles a node, read `plugin._last_exit_nodes` to set `previous_node_ids`
  - For regular nodes, set `previous_node_ids = [node_id]`
  - On line 530-532, use the list: `node_config = {**node_config, "dependencies": previous_node_ids}`

### 1h. Add circular macro expansion detection
**File:** `hexdag/kernel/configurable.py`

- Add `contextvars.ContextVar[list[str]]` expansion stack
- Add `_expansion_guard(instance_name)` context manager to `ConfigurableMacro`
- Check for cycles and enforce depth limit (20)
- Wrap `expand()` call in `MacroEntityPlugin.build()` with this guard

### 1i. Integration tests for macro_invocation through pipeline builder
**File:** `tests/hexdag/kernel/test_yaml_macro.py` (new test class `TestYamlMacroBuilderIntegration`)

Tests:
1. Full roundtrip: `kind: Macro` + `kind: Pipeline` with `macro_invocation` via multi-doc YAML
2. Sequential: macro followed by function_node — verify correct dependency on exit node
3. Config/inputs separation — verify overlap raises error
4. Macro with typed parameters — verify type validation
5. Macro with `default: null` — verify None is passed to context
6. Macro with external dependencies — verify entry nodes get them

---

## Phase 2: Enable Nested Macro Invocation ✅ DONE

*Goal: Allow YAML macros to compose other macros, removing the biggest YAML-only limitation.*

### 2a. Remove the nested `macro_invocation` restriction
**File:** `hexdag/kernel/yaml_macro.py` (lines 500-506)

Remove the explicit check that raises `YamlPipelineBuilderError("YAML macros cannot contain nested macro_invocations")`.

### 2b. Make `_build_graph_from_nodes` support macro_invocation
**File:** `hexdag/kernel/yaml_macro.py` (method `_build_graph_from_nodes`)

The `node_builder` callback (injected from `MacroEntityPlugin._make_node_builder()`) currently only uses `NodeEntityPlugin`. Extend it to also include `MacroEntityPlugin`:

**File:** `hexdag/compiler/plugins/macro_entity.py` (method `_make_node_builder`)
```python
def _build(rendered_nodes):
    graph = DirectedGraph()
    for node_config in rendered_nodes:
        for plugin in [macro_entity_plugin, node_plugin]:
            if plugin.can_handle(node_config):
                result = plugin.build(node_config, builder, graph)
                if result is not None:
                    graph += result
                break
    return graph
```

This leverages the circular detection from Phase 1h to prevent infinite recursion.

### 2c. Update the fallback path
**File:** `hexdag/kernel/yaml_macro.py` (fallback in `_build_graph_from_nodes`)

The standalone fallback (when `node_builder` is None) should also support `MacroEntityPlugin`. Import and register both plugins.

### 2d. Update YAML validator
**File:** `hexdag/compiler/yaml_validator.py`

Allow `macro_invocation` as a valid node kind inside Macro definitions.

### 2e. Tests for nested macro composition
- Define macro A with nodes, define macro B that invokes macro A, invoke macro B from a Pipeline
- Verify depth limit (20) is enforced
- Verify circular A→B→A raises error with clear message
- Verify parameters pass through correctly across nesting levels

---

## Phase 3: Enable Dynamic Node Generation ✅ DONE

*Goal: Support `{% for %}` at the node list level for generating N nodes from parameters.*

### 3a. Add pre-parse Jinja2 rendering for macro node templates
**File:** `hexdag/kernel/yaml_macro.py`

The key insight: currently YAML is parsed first, then Jinja2 renders individual string fields. For dynamic node generation, we need to render the **raw YAML string** of the `nodes:` section BEFORE parsing it.

**New method in `YamlMacro`:**
```python
def _render_nodes_template(self, nodes_yaml: str, context: dict) -> list[dict]:
    """Pre-parse Jinja2 rendering for dynamic node generation.

    Renders the raw YAML nodes template string with Jinja2,
    then parses the result as YAML to get the node list.
    """
    template = self.jinja_env.from_string(nodes_yaml)
    rendered_yaml = template.render(context)
    return yaml.safe_load(rendered_yaml)
```

### 3b. Store raw YAML string in macro definition
**File:** `hexdag/compiler/plugins/macro_definition.py`

When processing `kind: Macro`, store the raw YAML string of the `nodes:` section alongside the parsed list. Add `nodes_raw: str | None` to `YamlMacroConfig`.

**File:** `hexdag/kernel/yaml_macro.py`

Add to `YamlMacroConfig`:
```python
nodes_raw: str | None = None  # Raw YAML string for pre-parse rendering
```

### 3c. Two-path expansion in `YamlMacro.expand()`
**File:** `hexdag/kernel/yaml_macro.py`

In `expand()`, check if `nodes_raw` is available:
- If `nodes_raw` exists and contains Jinja2 block tags (`{%`): use pre-parse path (render YAML string → parse → build)
- Otherwise: use existing post-parse path (render individual string fields)

This preserves backward compatibility — existing macros without `{% for %}` work exactly as before.

### 3d. Update macro definition to capture raw YAML
**File:** `hexdag/compiler/plugins/macro_definition.py`

The challenge: by the time `MacroDefinitionPlugin.build()` runs, YAML is already parsed into Python dicts. We need to capture the raw string earlier.

**Approach:** In `YamlPipelineBuilder.build_from_yaml_string()`, when processing `kind: Macro` documents, extract the raw `nodes:` block as a string before full parsing. Store it alongside the parsed version.

Alternative simpler approach: serialize the parsed `nodes` list back to YAML string using `yaml.dump()`. This loses original formatting but preserves all content including Jinja2 tags (since YAML parser treats `{% %}` as regular strings in most contexts).

### 3e. Tests for dynamic node generation
- Macro with `{% for i in range(count) %}` generating N function_nodes
- Macro with `{% if mode == 'fast' %}` conditionally including/excluding nodes
- Verify node names are unique after dynamic generation
- Verify dependencies between dynamically generated nodes work

---

## Phase 4: Port Requirements & YAML Macro Examples ✅ DONE

*Goal: Make YAML macros self-documenting and provide reference examples.*

### 4a. Add `requires_ports` to YAML macro definition
**File:** `hexdag/kernel/yaml_macro.py`

Add to `YamlMacroConfig`:
```python
requires_ports: list[PortRequirement] | None = None

class PortRequirement(BaseModel):
    name: str           # Port name (e.g., "llm")
    protocol: str       # Expected protocol (e.g., "LLM", "SupportsKeyValue")
    optional: bool = False
```

### 4b. Validate port requirements at invocation time
**File:** `hexdag/compiler/plugins/macro_entity.py`

After resolving the macro, before expanding, check that all required ports are declared in the parent Pipeline's `spec.ports`. Raise `YamlPipelineBuilderError` if missing:
```python
if hasattr(macro_instance.config, "requires_ports") and macro_instance.config.requires_ports:
    pipeline_ports = builder.current_pipeline_ports  # Need to expose this
    for req in macro_instance.config.requires_ports:
        if not req.optional and req.name not in pipeline_ports:
            raise YamlPipelineBuilderError(
                f"Macro '{macro_ref}' requires port '{req.name}' ({req.protocol}) "
                f"but it is not declared in pipeline ports"
            )
```

### 4c. Expose pipeline ports to builder context
**File:** `hexdag/compiler/yaml_builder.py`

Store the current pipeline's `spec.ports` on the builder instance so plugins can access it during `_build_graph()`.

### 4d. Update YAML validator for macro port requirements
**File:** `hexdag/compiler/yaml_validator.py`

Validate `requires_ports` structure in Macro definitions.

### 4e. Create example YAML macro files

Create `examples/macros/` directory with:
1. `retry_workflow.yaml` — Simple retry pattern (function_node + composite_node)
2. `llm_chain.yaml` — Multi-step LLM chain (llm_node → llm_node → function_node)
3. `validate_and_process.yaml` — Validation + processing with conditional routing
4. `dynamic_pipeline.yaml` — Dynamic node generation with `{% for %}`

Each example should be a multi-doc YAML with `kind: Macro` definition + `kind: Pipeline` that invokes it.

### 4f. Tests for port requirements
- Macro with required port + Pipeline that provides it → success
- Macro with required port + Pipeline missing it → build-time error
- Macro with optional port + Pipeline missing it → success (no error)

---

## Implementation Order

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **Phase 1** (9 fixes) | Building infrastructure bugs | None — independent |
| **Phase 2** (nested macros) | Remove restriction, extend builder | Phase 1h (circular detection) |
| **Phase 3** (dynamic nodes) | Pre-parse Jinja2, raw YAML storage | Phase 2 (nested support in builder) |
| **Phase 4** (ports + examples) | Port requirements, validation, docs | Phase 1 (correct builder) |

Phases 3 and 4 are independent of each other and can be done in either order after their prerequisites.

---

## Verification (per phase)

```bash
# Phase 1
uv run pytest tests/hexdag/kernel/test_yaml_macro.py -x --tb=short
uv run pytest tests/hexdag/kernel/pipeline_builder/ -x --tb=short
uv run pytest tests/hexdag/stdlib/macros/ -x --tb=short

# Phase 2 (adds to Phase 1)
uv run pytest tests/hexdag/kernel/test_yaml_macro.py -x --tb=short -k "nested"

# Phase 3 (adds to above)
uv run pytest tests/hexdag/kernel/test_yaml_macro.py -x --tb=short -k "dynamic"

# Phase 4 (adds to above)
uv run pytest tests/hexdag/kernel/test_yaml_macro.py -x --tb=short -k "port"

# Always: type check + full suite
uv run pyright hexdag/kernel/yaml_macro.py hexdag/compiler/plugins/ hexdag/kernel/configurable.py
uv run pytest --tb=short
```

---

## Files Modified (Summary)

| File | Phases |
|------|--------|
| `hexdag/kernel/yaml_macro.py` | 1b, 1c, 1e, 3a, 3b, 3c, 4a |
| `hexdag/compiler/plugins/macro_entity.py` | 1a, 1f, 1g, 1h, 2b, 4b |
| `hexdag/compiler/yaml_builder.py` | 1g, 4c |
| `hexdag/kernel/configurable.py` | 1h |
| `hexdag/compiler/plugins/macro_definition.py` | 1d, 3d |
| `hexdag/compiler/yaml_validator.py` | 2d, 4d |
| `tests/hexdag/kernel/test_yaml_macro.py` | 1i, 2e, 3e, 4f |
| `examples/macros/` (new) | 4e |
