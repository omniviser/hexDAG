---
name: qa-fast
description: Fast inner-loop quality check — ruff check --fix + ruff format on changed files, mypy on hexdag/, the relevant hexDAG boundary checks, and last-failed tests only (pytest --lf -x). Use while iterating; use qa-gate before committing. Triggers "quick check", "qa-fast", "fast QA", "does it still pass", "/loop".
---

The tight convergence loop. Unlike `qa-gate`, this skips the full suite + coverage + doctests + interrogate/radon/vulture and runs only previously-failed tests, so each pass is cheap. Run in order; stop at the first failure.

## Why this skill exists
qa-gate is authoritative but slow (full pytest + coverage + docstring/complexity/dead-code + boundary scans). While editing you want sub-minute feedback: lint the files you touched, type-check, run the boundary checks that your change could trip, and re-run only what was red. Converge, then graduate to qa-gate.

## Pre-conditions
- Run from repo root; `uv` available.
- There must be uncommitted changes (else there's nothing to fast-check — suggest qa-gate).

## Step 0 — Scope to changed .py files
```bash
FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u )
echo "$FILES"
```

## Step 1 — Ruff check (auto-fix) on changed files
```bash
[ -n "$FILES" ] && uv run ruff check --fix --config=pyproject.toml $FILES
```

## Step 2 — Ruff format on changed files
```bash
[ -n "$FILES" ] && uv run ruff format --config=pyproject.toml $FILES
```

## Step 3 — Mypy (fine-grained cache makes repeat runs fast)
```bash
uv run mypy hexdag/
```

## Step 4 — Boundary checks likely relevant to the change
Pick by which paths changed (each scans the tree, but they're fast static checks):
```bash
echo "$FILES" | grep -q '^hexdag/kernel/'   && uv run python scripts/check_kernel_boundary.py && uv run python scripts/check_core_imports.py
echo "$FILES" | grep -q '^hexdag/kernel/ports/' && uv run python scripts/check_port_protocols.py
grep -lq 'async def' $FILES 2>/dev/null      && uv run python scripts/check_async_io.py
uv run python scripts/check_circular_imports.py   # cheap, catches import-order breakage early
```
When unsure, run `check_kernel_boundary.py` + `check_core_imports.py` — they catch the most common architectural regressions.

## Step 5 — Last-failed tests only
```bash
uv run pytest --lf -x -q
```
`--lf` re-runs only the tests that failed on the previous run. If the cache is empty, pytest runs the full set — narrow it manually to the file you're editing, e.g. `uv run pytest tests/hexdag/kernel/<path>/test_x.py -x -q`.

## Report back
PASS/FAIL per step. When all green: "qa-fast clean — run **qa-gate** before committing."

## What NOT to do
- Do NOT treat qa-fast green as commit-ready — it skips the coverage gate, doctests, and the full suite.
- Do NOT run the whole `tests/hexdag/` suite here; that belongs to qa-gate.
- Do NOT add `--no-verify` or lower any threshold.
