export const meta = {
  name: 'hexdag-pipeline-review',
  description: 'Adversarial review of hexDAG pipeline YAML across data-flow, ports/adapters, correctness, and simplification lenses, with a verify stage. Args: a pipeline file path (or omit to review changed *.yaml).',
  phases: [
    { title: 'Scope' },
    { title: 'Find' },
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

// Which pipeline files to review: the args path, else changed YAML.
const target = (typeof args === 'string' ? args : (args && args.join ? args.join(' ') : '')).trim()

const CONTEXT = `
You are reviewing hexDAG pipeline YAML (a workflow engine for AI agents). Pipelines are typed DAGs:
kind: Pipeline / System with spec.ports (adapters) and spec.nodes. Data flow is n8n-like:
- Ambient input: every top-level input field is available to every node by name. A "field: $input.field"
  pass-through is REDUNDANT, not required.
- Upstream outputs referenced as {{node.field}} (or bare {{node}}); each ref auto-creates a dependency edge.
- Closest-wins precedence: explicit input_mapping > upstream output > ambient input. strict_mapping opts out.
- wait_for: [node] is an ordering-only edge (run after a side-effect node without reading its data).
- Node kinds are aliases (llm_node, function_node, expression_node, agent_node, data_node, composite_node,
  transition, api_call_node, service_call_node) or resolvable module paths.

To ground findings, READ the actual file(s) and, where possible, run:
  hexdag validate <file> --verbose
  hexdag lint <file>
Cite file + node name (and line if visible) for every finding.
`

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          node: { type: 'string', description: 'the node metadata.name (or "ports"/"spec")' },
          severity: { type: 'string', enum: ['blocker', 'important', 'minor', 'nit'] },
          category: { type: 'string', enum: ['data-flow', 'ports-adapters', 'correctness', 'simplification'] },
          description: { type: 'string', description: 'the problem and the exact offending YAML' },
          impact: { type: 'string', description: 'concrete build error or runtime failure it causes' },
          suggested_fix: { type: 'string' },
        },
        required: ['title', 'file', 'node', 'severity', 'category', 'description', 'impact'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    is_real: { type: 'boolean', description: 'true only if the actual YAML clearly proves the issue' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    corrected_severity: { type: 'string', enum: ['blocker', 'important', 'minor', 'nit'] },
    verdict_reasoning: { type: 'string', description: 'quote the YAML, justify keep/reject' },
    suggested_fix: { type: 'string' },
  },
  required: ['is_real', 'confidence', 'corrected_severity', 'verdict_reasoning', 'suggested_fix'],
}

phase('Scope')
log(target ? `Reviewing pipeline(s): ${target}` : 'Reviewing changed *.yaml pipelines')

const LENSES = [
  {
    key: 'data-flow',
    focus:
      'DATA FLOW. Hunt: dangling {{node.field}} refs to a node that does not exist in THIS pipeline ' +
      '(common with !include fragments), expression-var names colliding with node names or builtins ' +
      '(len/coalesce) — a build error, {{name}} typos that near-miss a real node, missing edges (a node ' +
      'that should run after another but has no ref/wait_for), and redundant "field: $input.field" ' +
      'pass-throughs. Also flag input_mapping that unintentionally overrides a closer upstream (closest-wins).',
  },
  {
    key: 'ports-adapters',
    focus:
      'PORTS & ADAPTERS. Hunt: an unknown/typo\'d kind or adapter module path (validate rejects it), a ' +
      'node using a port not declared in spec.ports, hard-coded secrets/keys/tokens in config instead of ' +
      '${ENV_VAR}, an adapter pool (adapters:+strategy) with heterogeneous member capabilities, and ' +
      'model/config values that look wrong for the declared adapter.',
  },
  {
    key: 'correctness',
    focus:
      'CORRECTNESS. Hunt: a node whose template/expression references a field the upstream node does not ' +
      'output (renders None at runtime), composite/loop nodes with wrong condition/items refs, transition ' +
      'nodes targeting a state not in the state machine, and input/output-schema mismatches between wired nodes.',
  },
  {
    key: 'simplification',
    focus:
      'SIMPLIFICATION. Hunt: hand-written dependencies that a {{ref}} already implies, a separate ' +
      'function/expression node that an inline input_mapping expression could replace, duplicated ' +
      'subgraphs a macro could capture, and dead/unreachable nodes lint would flag. Only propose changes ' +
      'that reduce complexity without changing behavior.',
  },
]

phase('Find')
const finderResults = await parallel(LENSES.map((l) => () =>
  agent(
    `${CONTEXT}\n\nTASK: Review lens — ${l.key}.\n${l.focus}\n\n` +
    `Read the pipeline file(s)${target ? ` (${target})` : ' — find changed *.yaml with git'} and any included ` +
    `fragments. Run hexdag validate/lint if available. Report only defensible findings with the file + node ` +
    `and a concrete build/runtime scenario. An empty list is a valid, honest answer — do not pad.`,
    { label: `find:${l.key}`, phase: 'Find', schema: FINDING_SCHEMA, effort: 'high' }
  ).then((r) => (r && r.findings ? r.findings.map((f) => ({ ...f, lens: l.key })) : []))
))

const allFindings = finderResults.filter(Boolean).flat()
log(`Collected ${allFindings.length} candidate findings across ${LENSES.length} lenses -> verifying`)

phase('Verify')
const verified = await parallel(allFindings.map((f, i) => () =>
  agent(
    `${CONTEXT}\n\nTASK: Adversarially VERIFY this candidate. Default is_real=FALSE unless the actual YAML ` +
    `clearly proves it. Read the cited node and quote it.\n\n` +
    `CANDIDATE (lens=${f.lens}):\nTitle: ${f.title}\nFile: ${f.file} | node: ${f.node}\n` +
    `Severity: ${f.severity} | Category: ${f.category}\nDescription: ${f.description}\nImpact: ${f.impact}\n` +
    `Proposed fix: ${f.suggested_fix || '(none)'}\n\n` +
    `Reject if: ambient input already covers a "missing" value; the ref is a single-dep flattened output ` +
    `(not a typo); a closer input_mapping is intentional; or it's pure style. For a "dangling ref", confirm ` +
    `the node truly is absent from this pipeline AND its includes before keeping.`,
    { label: `verify:${f.lens}#${i}`, phase: 'Verify', schema: VERDICT_SCHEMA, effort: 'high' }
  ).then((v) => ({ finding: f, verdict: v }))
))

const confirmed = verified
  .filter((x) => x && x.verdict && x.verdict.is_real)
  .map((x) => ({ ...x.finding, severity: x.verdict.corrected_severity, confidence: x.verdict.confidence,
                 reasoning: x.verdict.verdict_reasoning, suggested_fix: x.verdict.suggested_fix || x.finding.suggested_fix }))

phase('Synthesize')
const order = { blocker: 0, important: 1, minor: 2, nit: 3 }
const sorted = confirmed.sort((a, b) => (order[a.severity] - order[b.severity]))
const digest = sorted.map((f) =>
  `- [${f.severity}] (${f.category}) ${f.title}\n  ${f.file} · node=${f.node} | confidence=${f.confidence}\n` +
  `  ${f.reasoning}\n  FIX: ${f.suggested_fix}`
).join('\n\n')

const report = await agent(
  `${CONTEXT}\n\nWrite the final pipeline-review report from the adversarially-CONFIRMED findings below. ` +
  `Tight, skimmable markdown grouped by severity, tagged by category, each with file + node and a fix. ` +
  `Open with a one-line verdict: "ship" if no blocker/important, else "fix-first". Invent nothing not ` +
  `below.\n\n=== CONFIRMED FINDINGS ===\n${digest || '(none confirmed)'}`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' }
)

return {
  target: target || 'changed *.yaml',
  candidates: allFindings.length,
  confirmed: sorted.map((f) => ({
    title: f.title, file: f.file, node: f.node, severity: f.severity,
    category: f.category, confidence: f.confidence, fix: f.suggested_fix,
  })),
  verdict: sorted.some((f) => f.severity === 'blocker' || f.severity === 'important') ? 'fix-first' : 'ship',
  report_markdown: report,
}
