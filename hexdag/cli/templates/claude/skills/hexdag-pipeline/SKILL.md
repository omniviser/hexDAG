---
name: hexdag-pipeline
description: Author, validate, and run a hexDAG YAML pipeline. Use when building or editing a pipeline/system manifest, wiring nodes and ports, or debugging a pipeline that won't build. Triggers "write a hexdag pipeline", "add a node", "wire up a port", "why won't my pipeline build", "run this pipeline".
argument-hint: "[path/to/pipeline.yaml] — the pipeline you're working on"
---

Build and iterate on a hexDAG pipeline the YAML-first way. hexDAG is a workflow engine for AI agents: you describe a typed DAG in YAML, and the engine runs it with validation, observability, and replay.

## Why this skill exists
A hexDAG pipeline that validates cleanly is the declarative equivalent of a passing type-check — dependencies are inferred, schemas are checked, and refs are resolved at build time. This skill keeps you on the fast author → validate → run loop and steers you away from the common footguns (redundant `$input` pass-throughs, unknown `kind`s, dangling refs).

## Core mental model (n8n-like data flow)
- **Ambient input**: every top-level field of the pipeline input is available to every node by name — in expressions, templates, and `@step`/function params. **Never** write `field: $input.field` pass-throughs.
- **Upstream outputs**: reference any upstream node output by name — `node_name.field.subfield`.
- **Auto-inferred dependencies**: any `{{node.field}}` or `{{node}}` ref in any spec string creates an edge. You rarely need to write `dependencies` by hand.
- **Closest-wins precedence**: explicit `input_mapping` > upstream data > ambient input. `strict_mapping: true` opts a node out.
- Use `wait_for: [node]` for ordering-only edges (run after a side-effect node without reading its data).

## Step 1 — Scaffold or open the pipeline
Start a new one from the CLI, or edit the file passed as the argument:
```bash
hexdag create <name>            # scaffold a starter pipeline
# or open the existing $1 and read it end-to-end first
```
A pipeline manifest looks like:
```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
spec:
  ports:
    llm:
      adapter: hexdag.stdlib.adapters.openai.OpenAIAdapter
      config: {model: gpt-4}
  nodes:
    - kind: llm_node
      metadata: {name: analyze}
      spec:
        human_message: "Analyze {{input.topic}}"
```

## Step 2 — Add / wire nodes
- Reference built-in nodes by alias: `llm_node`, `function_node`, `agent_node`, `expression_node`, `data_node`, `composite_node`, `transition`, `api_call_node`, `service_call_node` (or the full module path).
- Pull upstream data with `{{node.field}}` in any string; the edge is inferred automatically.
- For derived values, prefer an `expression_node` or inline `input_mapping` expressions (`total: "order.price * order.quantity"`) over a separate function.
- Discover what a node/adapter accepts with `hexdag explain <topic>` or the MCP `get_component_schema` tool.

## Step 3 — Validate (the type-check)
```bash
hexdag validate "$1" --verbose
hexdag lint "$1"
```
`validate` runs the real build path (include expansion, alias/custom-type registration, the validator) — a green result means it will build. `lint` adds best-practice warnings (unused nodes, `{{name}}` typos with "did you mean?", incomplete explicit deps). Fix every `validate` error and any `error`-severity lint finding before running.

## Step 4 — Run it
```bash
hexdag pipeline run "$1" --input '{"topic": "..."}'   # exact subcommand: hexdag pipeline --help
```
Use mock adapters (`hexdag.stdlib.adapters.mock.MockLLM`) for a dry run with no API keys.

## Report back
- The pipeline file, node graph (names + inferred edges), and `validate`/`lint` status.
- On a build failure: quote the node + field from the error and the fix applied.

## What NOT to do
- Do NOT write `field: $input.field` pass-through mappings — ambient input already covers them.
- Do NOT hand-write `dependencies` that a `{{ref}}` already implies (the builder infers + merges them).
- Do NOT invent a `kind` — use an alias or a resolvable module path (validate will reject unknowns with a suggestion).
- Do NOT hand-edit anything under a generated `schemas/` directory.
