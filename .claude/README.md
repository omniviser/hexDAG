# Claude Code setup — what it is, what it gives you, how to use it

This `.claude/` directory turns Claude Code from one big generalist chat into a **thin
orchestrator that routes work to narrow specialists, runs deterministic skills for repeatable
mechanics, and fans out workflows for structured multi-agent work.** Everything here is tuned to
*this* repo — hexDAG, a single `uv` package (Python 3.12+) with OS-inspired layers
(`kernel`/`stdlib`/`drivers`/`compiler`/`api`/`cli`) and a large `.pre-commit-config.yaml` that is
the source of truth for "clean".

Ported from the Omniviser backend-chat setup and adapted to hexDAG's toolchain: GitHub (not Azure
DevOps), no commitizen (branch-name regex + conventional-commit *style*), and hexDAG's own
boundary checks + YAML validation.

---

## Contents

| Piece | Files | What it is |
|---|---|---|
| **Skills** | `.claude/skills/*/SKILL.md` | Deterministic, copy-pasteable procedures. Run **in your current context**. |
| **Workflows** | `.claude/workflows/*.js` | Multi-agent orchestration — fan out, adversarially verify, synthesize. |
| **Specialist agents** | `.claude/agents/*.md` | Deep experts on one subsystem each, in their **own context window**. |
| **Settings + hooks** | `.claude/settings.json` | Shared permission allow/deny, a hook that auto-formats `.py` on edit, and a session-start reminder. `settings.local.json` = personal, git-ignored overrides. |

> **Skill vs agent vs workflow.** A **skill** is a fixed recipe run identically every time
> (shares your context). An **agent** is open-ended judgement in one domain (fresh isolated
> context, carries that domain's footguns). A **workflow** coordinates many agents deterministically
> (fan-out → verify → synthesize). Skill = "do these exact steps"; agent = "figure this out";
> workflow = "do this at scale / in parallel".

---

## The 6 skills (slash commands)

| Command | Does | Run it |
|---|---|---|
| `/qa-fast` | ruff `--fix`+format on changed files, mypy, the relevant boundary checks, `pytest --lf`. Sub-minute. | while iterating |
| `/qa-gate` | full gate mirroring `.pre-commit-config.yaml`: ruff, pyupgrade, mypy+pyright, bandit, ~11 boundary checks, doctests, pytest w/ 70% coverage, interrogate/radon/vulture. `all` adds pre-push integration tests. | before committing |
| `/commit` | conventional-commit-style commit; stages by name; appends Co-Authored-By; never `--no-verify`/`--amend`. | to commit |
| `/pr-prep [title]` | qa-gate → branch-regex check → commit → push → open PR to `main` via `gh`. | to ship |
| `/worktree` | worktree hygiene; spin up an isolated worktree under `.claude/worktrees/` for parallel work. | for parallel work |
| `/yaml-check [file]` | `hexdag validate` + `hexdag lint` on changed pipeline YAML + schema-sync. | after a YAML/compiler edit |

Example — `/qa-gate` output *(illustrative)*:
```
qa-gate ▸ changed: py[✓] schema[–]
| Step               | Status | Notes                        |
| ruff format        | PASS   | 1 file reformatted           |
| mypy + pyright     | PASS   | 0 errors                     |
| boundary checks    | PASS   | kernel-boundary OK           |
| pytest (cov gate)  | PASS   | 812 passed · cov 74% (gate 70)|
| interrogate        | PASS   | docstrings 96% (gate 95)     |
qa-gate clean — safe to commit / run /pr-prep.
```

---

## Workflows (structured multi-agent)

Workflows spawn **many** agents in parallel — use them for exhaustive or parallel work, not
one-liners. Trigger by asking in plain language; run `/workflows` to watch the live progress tree.

### `code-review` — review the current diff before you ship
```
Scope      → resolves the diff (git diff main...HEAD + git diff HEAD)
Find       → 5 finder agents in PARALLEL, each a lens:
             correctness · architecture-boundary · type-safety · coverage · simplification
Verify     → every candidate gets an independent skeptic that defaults to REJECT unless the
             current code proves it (kills plausible-but-wrong findings)
Synthesize → one agent merges survivors into a ranked report + approve/needs-changes verdict
```
The `architecture-boundary` lens maps directly to hexDAG's custom hooks — a confirmed finding there
is a hook failure waiting to happen (kernel importing stdlib, port using `pass`, kwargs-only
`__init__`, `time.time()` for durations).

### `feature-fanout` — build a multi-part feature in parallel
Pass subtasks as `agent-name:: task`, separated by `|` or newlines:
```
Run feature-fanout with:
core-engine-specialist:: add a CircuitBreaker middleware to the orchestrator |
compiler-yaml-specialist:: add a validator rule flagging orphaned nodes
```
```
Plan       → parses the subtasks
Implement  → each specialist runs in PARALLEL in its OWN git worktree off main
             (isolation:'worktree' — no file collisions). Verifies its base, implements only its
             subtask, runs local ruff/mypy/boundary/pytest, returns a diff.
Integrate  → one critic reads all diffs and reports conflicts, coverage gaps, and a safe merge order
```

---

## The 3 specialist agents

| Agent | Owns | Route when |
|---|---|---|
| `core-engine-specialist` | `hexdag/kernel/**` + `hexdag/drivers/**` — orchestrator, DAG/NodeSpec, ports/adapters, Service `@tool`/`@step`, executors, observers, lifecycle | execution/orchestration/port/event/state-machine work, or a boundary/async/timer hook failure |
| `compiler-yaml-specialist` | `hexdag/compiler/**` + `hexdag/stdlib/nodes/**` — YAML build/validation, dependency inference, node factories, schema sync, ambient-input data flow | a pipeline won't build/validate, dep-inference/ref bugs, node-factory behavior, schema drift |
| `qa-triage` | **read-only** — reproduces a vague bug, localizes it, routes to the owning specialist. No write tools. | any vague/cross-cutting report or pasted stack trace — fire it first |

### How to invoke
- **Auto-dispatch** — describe the task; the matching agent fires from its `description` triggers.
  *"The orchestrator runs waves out of order"* → `core-engine-specialist`.
- **Explicit** — *"Use the compiler-yaml-specialist to add a validator rule for orphaned nodes."*
- **Triage first** — paste a stack trace you can't place; `qa-triage` reproduces and hands a clean
  packet to the owning specialist.

---

## How to use it day to day

1. **Focused work in a fragile area** — describe it; the right specialist auto-dispatches.
2. **Iterate** — `/qa-fast` (a PostToolUse hook also auto-ruffs any `.py` you save).
3. **YAML pipeline change** — `/yaml-check` to confirm it builds.
4. **Ship** — `/qa-gate` → `/commit` → `/pr-prep`. Never bump a version by hand (CI does it).
5. **Review a diff** — the `code-review` workflow.
6. **Parallel / big work** — `/worktree` or the `feature-fanout` workflow.

**Full loop:**
```
You:  Use the compiler-yaml-specialist to fix input_mapping not overriding ambient input.
Agent: edits reference_resolver.py, adds a test, reports the diff.
You:  /qa-fast    → clean
You:  /qa-gate    → PASS (cov 74%)
You:  /pr-prep fix(compiler): honor input_mapping over ambient input
      → PR #NN opened to main.
```

---

## Guardrails baked in
- `settings.json` **denies** `--no-verify`, `--amend`, force-push, `git reset --hard`, and secret
  reads — the non-negotiable rules are enforced, not just documented.
- A **PostToolUse** hook auto-runs `ruff format` + `ruff check --fix` on any `.py` you edit.
- Every skill's "What NOT to do" section repeats the load-bearing rules (never lower the coverage
  gate, never touch the version/CHANGELOG, never cross the kernel boundary).

---

## Extending it
- **New skill** → `.claude/skills/<name>/SKILL.md`: frontmatter (`name` + trigger-rich
  `description`) + numbered steps + "What NOT to do".
- **New workflow** → `.claude/workflows/<name>.js`: starts with `export const meta` and uses
  `agent()` / `parallel()` / `pipeline()` / `phase()`.
- **New agent** → `.claude/agents/<name>.md`: `name` / trigger-rich `description` / `tools` / body =
  ownership + invariants + footgun ledger + "What NOT to do".
- Keep this README and `../CLAUDE.md` in sync when the roster changes.

See `../CLAUDE.md` for full conventions, the QA/coverage gates, and branch/version discipline.
