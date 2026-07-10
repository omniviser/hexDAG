export const meta = {
  name: 'feature-fanout',
  description: 'Dispatch one worktree-isolated specialist per independent subtask of a feature, collect diffs, then run an integration + coverage critic pass. Args: newline- or pipe-separated subtasks (optionally "agent-name:: task").',
  phases: [
    { title: 'Plan' },
    { title: 'Implement' },
    { title: 'Integrate' },
  ],
}

// Base branch every isolated worktree must branch from.
const BASE = 'main'

// Valid specialist agent names in this repo (used if a subtask specifies "agent:: task"):
// core-engine-specialist | compiler-yaml-specialist

// ---- Parse subtasks from args -------------------------------------------------
const rawArgs = (typeof args === 'string' ? args : (args && args.join ? args.join('\n') : '')).trim()
const subtaskSpecs = rawArgs
  .split(/\n|\s\|\s/)
  .map((s) => s.trim())
  .filter(Boolean)
  .map((s, i) => {
    const m = s.match(/^([a-z0-9-]+)::\s*(.+)$/i)
    return {
      index: i + 1,
      agentType: m ? m[1] : null,
      task: m ? m[2] : s,
      // Worktree branch name MUST satisfy the enforced regex:
      // ^(ci|dependabot/pip|docs|experiment|feat|fix|refactor|test)/[A-Za-z0-9._-]+$
      branch: `feat/fanout-${i + 1}-${rawArgs.length}`,
    }
  })

const CONTEXT = `
Repo: hexDAG (single uv package, Python 3.12+; layers hexdag/{kernel,stdlib,drivers,compiler,api,cli}).
You are ONE specialist implementing ONE independent subtask of a larger feature, in your OWN
git worktree isolated from sibling agents.

CRITICAL — VERIFY YOUR WORKTREE BASE before writing code:
  git rev-parse --abbrev-ref HEAD          # must be YOUR branch, not ${BASE}, not another agent's
  git merge-base --is-ancestor origin/${BASE} HEAD && echo "based on ${BASE}: OK" || echo "STALE BASE"
If the base is stale or you are on the wrong branch, STOP and report — do NOT build on the wrong tree.

House rules (must pass local checks before you claim done):
- Modern 3.12 type hints only (list/dict, X | Y, X | None); ruff line-length 100; mypy strict + pyright
  on hexdag/ (excl. cli/studio). Async-first (no blocking I/O in async; use Timer for durations).
- Kernel purity: hexdag/kernel/** never imports stdlib/drivers. Port protocols use '...'. Adapter
  __init__ uses explicit typed params, not **kwargs-only. Exceptions inherit HexDAGError.
- Coverage gate 70%; docstring coverage 95% (interrogate). Add tests under tests/hexdag/<area>/
  mirroring the source; test dirs need __init__.py.
- Do NOT touch pyproject version / CHANGELOG (CI-owned). Do NOT commit secrets.
Run the relevant qa commands inside your worktree before reporting:
  uv run ruff format --config=pyproject.toml hexdag/ && uv run ruff check --fix --config=pyproject.toml hexdag/
  uv run mypy hexdag/
  uv run python scripts/check_kernel_boundary.py && uv run python scripts/check_core_imports.py
  uv run pytest tests/hexdag/<area> -x -q
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    base_verified: { type: 'boolean', description: 'true only if you confirmed the worktree is based on ' + BASE },
    summary: { type: 'string', description: 'what you implemented' },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_added: { type: 'array', items: { type: 'string' } },
    qa: {
      type: 'object',
      properties: {
        ruff: { type: 'string', enum: ['pass', 'fail', 'skipped'] },
        mypy: { type: 'string', enum: ['pass', 'fail', 'skipped'] },
        boundary: { type: 'string', enum: ['pass', 'fail', 'skipped'] },
        pytest: { type: 'string', enum: ['pass', 'fail', 'skipped'] },
      },
      required: ['ruff', 'mypy', 'pytest'],
    },
    diff: { type: 'string', description: 'output of `git diff ' + BASE + '...HEAD` from your worktree (may be truncated)' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['branch', 'base_verified', 'summary', 'files_changed', 'qa'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    integration_risks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'important', 'minor'] },
          subtasks_involved: { type: 'array', items: { type: 'number' } },
          description: { type: 'string' },
        },
        required: ['title', 'severity', 'description'],
      },
    },
    coverage_gaps: { type: 'array', items: { type: 'string' } },
    merge_order: { type: 'array', items: { type: 'number' }, description: 'suggested subtask merge order' },
    verdict: { type: 'string', enum: ['integrates-cleanly', 'needs-reconciliation'] },
  },
  required: ['integration_risks', 'coverage_gaps', 'merge_order', 'verdict'],
}

phase('Plan')
if (subtaskSpecs.length === 0) {
  log('No subtasks provided. Pass subtasks separated by newlines or " | ", optionally "agent-name:: task".')
  return { error: 'no subtasks provided', hint: 'e.g. "core-engine-specialist:: add a CircuitBreaker middleware | compiler-yaml-specialist:: add a new validator rule for orphaned nodes"' }
}
log(`Fanning out ${subtaskSpecs.length} independent subtasks, each in its own worktree off ${BASE}`)

phase('Implement')
const impl = await parallel(subtaskSpecs.map((st) => () =>
  agent(
    `${CONTEXT}\n\nYOUR SUBTASK #${st.index}:\n${st.task}\n\n` +
    `Your worktree branch is "${st.branch}" created off ${BASE}. First run the base-verification ` +
    `commands above and set base_verified accordingly. Implement ONLY this subtask; do not stray into ` +
    `sibling subtasks. Add/adjust tests to keep the 70% coverage gate. Run the local qa commands and ` +
    `record results. Finally run: git add -A && git diff ${BASE}...HEAD and paste the diff.`,
    {
      label: `impl#${st.index}`,
      phase: 'Implement',
      schema: IMPL_SCHEMA,
      effort: 'high',
      isolation: 'worktree',
      ...(st.agentType ? { agentType: st.agentType } : {}),
    }
  ).then((r) => ({ spec: st, result: r }))
))

const done = impl.filter((x) => x && x.result)
const staleBase = done.filter((x) => x.result && x.result.base_verified === false)
if (staleBase.length) {
  log(`WARNING: ${staleBase.length} subtask(s) reported an unverified/stale worktree base — treat their diffs as suspect.`)
}

phase('Integrate')
const bundle = done.map((x) => {
  const r = x.result
  return `### Subtask #${x.spec.index} — branch ${r.branch}${r.base_verified ? '' : '  [BASE NOT VERIFIED]'}\n` +
    `Task: ${x.spec.task}\nSummary: ${r.summary}\n` +
    `Files: ${(r.files_changed || []).join(', ')}\n` +
    `Tests: ${(r.tests_added || []).join(', ') || 'none'}\n` +
    `QA: ruff=${r.qa.ruff} mypy=${r.qa.mypy} boundary=${r.qa.boundary || 'n/a'} pytest=${r.qa.pytest}\n` +
    `--- diff ---\n${r.diff || '(no diff captured)'}`
}).join('\n\n')

const critique = await agent(
  `${CONTEXT}\n\nYou are the INTEGRATION + COVERAGE CRITIC. Below are the isolated diffs from ${done.length} ` +
  `specialist worktrees, each branched off ${BASE}.\n\n` +
  `First re-state the base caution: each worktree was created off ${BASE} at fan-out time; if siblings ` +
  `touched shared files (kernel domain models, ports, the orchestrator, compiler reference_resolver, ` +
  `stdlib node factories, or schemas/) the branches may CONFLICT. Any subtask flagged ` +
  `"[BASE NOT VERIFIED]" must be re-based and re-checked before merge.\n\n` +
  `Analyze for: (1) integration risks — overlapping edits to the same file/symbol, incompatible ` +
  `interface assumptions between subtasks, a kernel-boundary violation introduced only when two diffs ` +
  `combine, schema drift if one changed a domain model without regenerating schemas; (2) coverage gaps — ` +
  `combined change that drops below the 70% gate or leaves cross-subtask interactions untested; (3) a ` +
  `safe merge order. Do not re-review intra-subtask correctness in depth — focus on how the pieces fit.\n\n` +
  `=== SUBTASK DIFFS ===\n${bundle}`,
  { label: 'integration-critic', phase: 'Integrate', schema: CRITIC_SCHEMA, effort: 'high' }
)

return {
  base: BASE,
  subtasks: done.map((x) => ({
    index: x.spec.index,
    branch: x.result.branch,
    agentType: x.spec.agentType,
    base_verified: x.result.base_verified,
    qa: x.result.qa,
    files_changed: x.result.files_changed,
  })),
  stale_base_subtasks: staleBase.map((x) => x.spec.index),
  integration: critique,
  next: 'Reconcile per merge_order; rebase any [BASE NOT VERIFIED] worktrees; then run pr-prep per branch or a combined branch.',
}
