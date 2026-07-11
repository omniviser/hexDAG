# hexDAG — Claude Code assets

These files were scaffolded by `hexdag init --claude`. They give Claude Code (and compatible
agents) first-class support for **using hexDAG** in this project: authoring, validating, and
debugging YAML pipelines.

| Asset | Type | What it does |
|---|---|---|
| `skills/hexdag-pipeline/` | skill | `/hexdag-pipeline` — author/edit/run a pipeline the YAML-first way (ambient input, `{{node.field}}` refs, closest-wins). |
| `skills/hexdag-validate/` | skill | `/hexdag-validate` — `hexdag validate` + `hexdag lint` on changed pipelines (the build-safety check). |
| `agents/hexdag-pipeline-debugger.md` | subagent | Auto-dispatches when a pipeline won't build/validate/run; reasons from the data-flow model. |
| `workflows/hexdag-pipeline-review.js` | workflow | Adversarial multi-agent review of pipeline YAML (data-flow · ports · correctness · simplification). |

## Using them
- **Skills**: type `/hexdag-pipeline` or `/hexdag-validate`, or just describe the task.
- **Agent**: describe a broken pipeline ("my pipeline says cannot resolve kind X") and the debugger fires.
- **Workflow**: ask to "run the hexdag-pipeline-review workflow on pipelines/foo.yaml".

## The data-flow model these assume
- **Ambient input** — every top-level input field is available to every node by name. Never write
  `field: $input.field` pass-throughs.
- **Auto-inferred deps** — any `{{node.field}}` ref creates an edge; you rarely hand-write `dependencies`.
- **Closest-wins** — explicit `input_mapping` > upstream output > ambient input.

## Keeping them current
Re-run `hexdag init --claude --force` after upgrading hexdag to refresh these assets. Feel free to
edit them for your project — but a `--force` refresh overwrites local edits, so keep project-specific
tweaks in separate files.

See the hexdag MCP server (`python -m hexdag --mcp`) for tool-based pipeline inspection/validation
that complements these filesystem assets.
