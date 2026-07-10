export const meta = {
  name: 'code-review',
  description: 'Adversarial review of the current git diff across correctness, architecture-boundary, type-safety, test-coverage, and simplification lenses, with a verify stage before synthesis',
  phases: [
    { title: 'Scope' },
    { title: 'Find' },
    { title: 'Verify' },
    { title: 'Synthesize' },
  ],
}

// Review the working diff vs the merge-base with main by default.
const BASE = 'main'

const CONTEXT = `
Repo: hexDAG — a developer-first workflow engine for AI agents (single uv package, Python 3.12+).
OS-inspired layering under hexdag/:
  kernel/    — execution engine: DAG/NodeSpec domain models, orchestrator, ports (LLM, DataStore,
               PipelineSpawner), validation, Service base (@tool/@step). MUST NOT import stdlib/drivers.
  stdlib/    — builtins: adapters (OpenAI, SQLite, Mock), node factories (LLMNode, AgentNode...),
               macros, prompts, system libs (ProcessRegistry, EntityState, PipelineMemory).
  drivers/   — low-level infra: LocalExecutor, LocalObserverManager, LocalPipelineSpawner.
  compiler/  — YAML build/validation: config loader, reference/dependency inference
               (extract_refs_from_spec), validator rules, schema gen/sync, system builder.
  api/, cli/ — user-facing surfaces (excluded from mypy/pyright + coverage).

Review the CURRENT change set. To see it:
  git diff ${BASE}...HEAD            # committed on this branch vs main
  git diff HEAD                      # uncommitted, staged+unstaged
  git status --short
Read the actual files (do not trust the diff header). Always cite file:line for current code.

House rules that findings must respect (from CLAUDE.md + .pre-commit-config.yaml):
- Modern Python 3.12+ type hints ONLY: list/dict/set (not List/Dict/Set), X | Y (not Union),
  X | None (not Optional), 'type' alias (not TypeAlias). ruff UP006/UP007/UP035/UP037/UP040 enforce this.
- ruff line-length 100, numpy docstrings. mypy strict + pyright standard on hexdag/ (excl. cli/studio).
- Async-first: no blocking I/O inside async functions (scripts/check_async_io.py). Use a* methods,
  aiofiles, etc. Use Timer from hexdag/kernel/utils/node_timer.py for durations — never time.time().
- Kernel purity: hexdag/kernel/** must not import stdlib/drivers (scripts/check_kernel_boundary.py,
  check_core_imports.py). Port protocols use '...' bodies, not 'pass'. Adapter/component __init__
  must use EXPLICIT typed params, not **kwargs-only (breaks SchemaGenerator → empty schemas).
- All framework exceptions inherit from HexDAGError.
- Coverage gate 70% (--cov-fail-under=70); docstring coverage 95% (interrogate). Tests live under
  tests/hexdag/<area>/ mirroring the source tree; test dirs need __init__.py.
- Versioning/CHANGELOG are CI-owned — never suggest manual bumps.
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
          line: { type: 'string', description: 'line number or range in current code' },
          severity: { type: 'string', enum: ['blocker', 'important', 'minor', 'nit'] },
          category: { type: 'string', enum: ['correctness', 'architecture-boundary', 'type-safety', 'coverage', 'simplification'] },
          description: { type: 'string', description: 'the problem and the exact buggy/weak code' },
          impact: { type: 'string', description: 'concrete failure or risk scenario at runtime' },
          suggested_fix: { type: 'string' },
        },
        required: ['title', 'file', 'line', 'severity', 'category', 'description', 'impact'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    is_real: { type: 'boolean', description: 'true only if the current code clearly proves the issue' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    corrected_severity: { type: 'string', enum: ['blocker', 'important', 'minor', 'nit'] },
    verdict_reasoning: { type: 'string', description: 'quote the actual code, trace the path, justify keep/reject' },
    suggested_fix: { type: 'string', description: 'concrete fix if real; empty if not' },
  },
  required: ['is_real', 'confidence', 'corrected_severity', 'verdict_reasoning', 'suggested_fix'],
}

phase('Scope')
log(`Reviewing current diff vs ${BASE}`)

const LENSES = [
  {
    key: 'correctness',
    focus:
      'CORRECTNESS BUGS. Trace the changed logic end to end. Hunt: None/KeyError hazards, ' +
      'blocking I/O inside async functions or missing await, off-by-one / boundary errors, ' +
      'mishandled exceptions (and exceptions not inheriting HexDAGError), type confusion, ' +
      'wrong DAG/dependency/wave logic, node input/output-schema mismatches, race conditions in ' +
      'asyncio.gather fan-out. Read helpers the diff calls into.',
  },
  {
    key: 'architecture-boundary',
    focus:
      'ARCHITECTURE & BOUNDARY PURITY. Hunt: hexdag/kernel/** importing stdlib or drivers (violates ' +
      'check_kernel_boundary / check_core_imports), port protocols using "pass" instead of "...", ' +
      'adapter/component __init__ that is **kwargs-only instead of explicit typed params (breaks ' +
      'SchemaGenerator), time.time() used for durations instead of Timer, Supports* protocols ' +
      'leaked/exported from hexdag top-level, new blocking I/O paths that should be async, circular ' +
      'imports. These map to the custom scripts/check_*.py hooks — a finding here is a hook failure.',
  },
  {
    key: 'type-safety',
    focus:
      'TYPE SAFETY (mypy strict + pyright + modern-hint ruff rules). Hunt: legacy typing ' +
      '(List/Dict/Set/Optional/Union/TypeAlias) that UP006/UP007/UP035/UP037/UP040 reject, missing ' +
      'or Any-typed function signatures under disallow_untyped_defs, `# type: ignore` without a code ' +
      'or without justification, unsound casts, return-type mismatches, Protocol/Generic misuse.',
  },
  {
    key: 'coverage',
    focus:
      'TEST-COVERAGE GAPS. For each changed src symbol, is there a matching unit test under the ' +
      'mirrored tests/hexdag/<area>/ tree? Identify untested branches/error paths that could drop ' +
      'coverage below the 70% gate. Flag missing doctests where a docstring shows an example (the ' +
      'doctest hook runs them). Flag test dirs missing __init__.py, or tests placed in the wrong tree.',
  },
  {
    key: 'simplification',
    focus:
      'REUSE / SIMPLIFICATION / EFFICIENCY. Hunt: duplicated logic an existing kernel/stdlib helper ' +
      'already provides (e.g. coalesce/default in expressions, Timer, resolve()), needless ' +
      'complexity (radon B+), dead code (vulture would flag), redundant validation/serialization, ' +
      'per-item awaits that could be gathered. Only propose changes that clearly reduce risk or lines ' +
      'without changing behavior.',
  },
]

phase('Find')
const finderResults = await parallel(LENSES.map((l) => () =>
  agent(
    `${CONTEXT}\n\nTASK: Review lens — ${l.key}.\n${l.focus}\n\n` +
    `Run the git diff commands above, READ the current files (and callers/callees), and report only ` +
    `defensible findings with exact file:line and a concrete scenario. An empty list is a valid, honest ` +
    `answer — do not pad. For architecture-boundary, default severity UP (a boundary violation fails a hook).`,
    { label: `find:${l.key}`, phase: 'Find', schema: FINDING_SCHEMA, effort: 'high' }
  ).then((r) => (r && r.findings ? r.findings.map((f) => ({ ...f, lens: l.key })) : []))
))

const allFindings = finderResults.filter(Boolean).flat()
log(`Collected ${allFindings.length} candidate findings across ${LENSES.length} lenses -> verifying`)

phase('Verify')
const verified = await parallel(allFindings.map((f, i) => () =>
  agent(
    `${CONTEXT}\n\nTASK: Adversarially VERIFY this candidate finding. Default is_real=FALSE unless the ` +
    `current code clearly proves it. Read the cited code and trace the path; quote it.\n\n` +
    `CANDIDATE (lens=${f.lens}):\nTitle: ${f.title}\nFile: ${f.file}:${f.line}\n` +
    `Severity: ${f.severity} | Category: ${f.category}\nDescription: ${f.description}\n` +
    `Impact: ${f.impact}\nProposed fix: ${f.suggested_fix || '(none)'}\n\n` +
    `Reject if: the code doesn't do what's claimed; a guard elsewhere prevents it; it's pre-existing on ` +
    `${BASE} (not introduced by this change); it's pure style with no rule behind it; or it's out of the ` +
    `change's scope. For architecture-boundary, only reject if you can point to why the boundary is NOT ` +
    `crossed (e.g. the import is under TYPE_CHECKING, or the path is an allowlisted exception).`,
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
  `- [${f.severity}] (${f.category}) ${f.title}\n  ${f.file}:${f.line} | confidence=${f.confidence}\n` +
  `  ${f.reasoning}\n  FIX: ${f.suggested_fix}`
).join('\n\n')

const report = await agent(
  `${CONTEXT}\n\nWrite the final code-review report. Below are the adversarially-CONFIRMED findings only. ` +
  `Produce tight, skimmable markdown grouped by severity (Blockers, Important, Minor, Nits) and, within ` +
  `each, tagged by category. Give file:line, the concrete failure, and the fix for each. Deduplicate. ` +
  `Open with a one-line verdict: "approve" if no blocker/important, else "needs-changes". Invent nothing ` +
  `not present below.\n\n=== CONFIRMED FINDINGS ===\n${digest || '(none confirmed)'}`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' }
)

return {
  base: BASE,
  candidates: allFindings.length,
  confirmed: sorted.map((f) => ({
    title: f.title, file: f.file, line: f.line, severity: f.severity,
    category: f.category, confidence: f.confidence, fix: f.suggested_fix,
  })),
  verdict: sorted.some((f) => f.severity === 'blocker' || f.severity === 'important') ? 'needs-changes' : 'approve',
  report_markdown: report,
}
