---
name: qa-gate
description: Full deterministic quality gate mirroring the pre-commit + pre-push hooks in .pre-commit-config.yaml. Runs ruff format/check, pyupgrade, mypy + pyright, bandit, the hexDAG boundary checks (kernel-boundary, core-imports, async-io, port-protocols, timer-usage, ...), doctests, pytest with the 70% coverage gate, interrogate/radon/vulture, and optionally integration tests. Use before every commit or PR, when asked to "run QA", "run the full quality gate", "check the codebase is clean", or "verify before pushing".
argument-hint: "[all] — force the full run (incl. pre-push integration tests) regardless of git diff"
---

Run the **full** quality gate, in order, stopping and reporting at the first hard failure. This mirrors `.pre-commit-config.yaml` exactly — **do not invent flags**. Where a step is scoped to changed paths, only run it when those paths changed; if you cannot determine the diff, or the user passed `all`, run everything (including the pre-push steps in Step 11).

## Why this skill exists
The pre-commit/pre-push hooks are the source of truth for "clean". Reproducing them verbatim locally means a green qa-gate → a commit/push that will not bounce. hexDAG's config adds ~15 custom boundary checks on top of the usual ruff/mypy/pytest; this skill runs them in hook order so nothing is missed.

## Pre-conditions (hard — refuse if unmet)
- Run from the repo root (`git rev-parse --show-toplevel`).
- `uv` is available (`uv --version`).
- Never pass `--no-verify`; never lower `--cov-fail-under` (70) or interrogate's `--fail-under` (95); never edit the `pyproject.toml` version or `CHANGELOG.md`.

## Step 0 — Detect what changed
```bash
CHANGED=$( { git diff --name-only HEAD; git diff --name-only --cached; git ls-files --others --exclude-standard; } | sort -u )
echo "$CHANGED"
echo "$CHANGED" | grep -qE '\.py$'                                  && PY=1     || PY=0
echo "$CHANGED" | grep -qE '(^schemas/|^hexdag/compiler/)'          && SCHEMA=1 || SCHEMA=0
```
If the argument is `all`, or `$CHANGED` is empty, treat every flag as `1` and also run Step 11.

## Step 1 — Ruff format
```bash
uv run ruff format --config=pyproject.toml hexdag/
```

## Step 2 — Ruff check with auto-fix
```bash
uv run ruff check --fix --config=pyproject.toml hexdag/
```

## Step 3 — Pyupgrade (only changed .py, excludes tests/examples per hook)
```bash
[ "$PY" = 1 ] && uv run pyupgrade --py312-plus $(echo "$CHANGED" | grep -E '\.py$' | grep -vE '^(tests|examples)/') || true
```

## Step 4 — Type checking (mypy + pyright)
```bash
uv run mypy hexdag/
uv run pyright
```
Both exclude `tests/`, `examples/`, `hexdag/studio/`, `hexdag/cli/` via config — do not point them there manually.

## Step 5 — Security scan
```bash
uv run bandit -ll -c pyproject.toml -r hexdag
uv run safety check --output bare 2>&1 | grep -v "DEPRECATED\|pkg_resources" || true
```
`gitleaks` also runs in the real hook; if `pre-commit` is installed you may instead run `uv run pre-commit run gitleaks --all-files`.

## Step 6 — hexDAG boundary checks (only when .py changed — each scans the tree)
```bash
[ "$PY" = 1 ] && uv run python scripts/check_core_imports.py
[ "$PY" = 1 ] && uv run python scripts/check_kernel_boundary.py
[ "$PY" = 1 ] && uv run python scripts/check_async_io.py
[ "$PY" = 1 ] && uv run python scripts/check_exception_hierarchy.py
[ "$PY" = 1 ] && uv run python scripts/check_port_protocols.py
[ "$PY" = 1 ] && uv run python scripts/check_timer_usage.py
[ "$PY" = 1 ] && uv run python scripts/check_init_params.py
[ "$PY" = 1 ] && uv run python scripts/check_lazy_imports.py
[ "$PY" = 1 ] && uv run python scripts/check_circular_imports.py
[ "$PY" = 1 ] && uv run python scripts/check_deprecated_usage.py
[ "$PY" = 1 ] && uv run python scripts/check_test_structure.py
```

## Step 7 — Schema sync (only when schemas/ or hexdag/compiler/ changed)
```bash
[ "$SCHEMA" = 1 ] && uv run python scripts/check_yaml_schema_sync.py
[ "$SCHEMA" = 1 ] && uv run python scripts/check_schemas.py
```

## Step 8 — Doctests
```bash
uv run pytest --doctest-modules hexdag/ --ignore=hexdag/cli/ --doctest-continue-on-failure
```

## Step 9 — Unit tests + coverage gate (verbatim from the pytest hook)
```bash
uv run pytest tests/hexdag/ --ignore=tests/hexdag/cli -x --tb=short --cov=hexdag --cov-config=pyproject.toml --cov-fail-under=70 --cov-report=term-missing:skip-covered
```

## Step 10 — Docstring coverage, complexity, dead code
```bash
uv run interrogate hexdag/ --fail-under 95 --quiet --exclude hexdag/studio/
uv run radon cc hexdag/ --min B --show-complexity --exclude hexdag/studio/
uv run vulture hexdag/ .vulture_whitelist.py --min-confidence 90 --exclude hexdag/studio/
```

## Step 11 — Pre-push parity (only on `all`, or before an actual push)
```bash
uv run pytest tests/integration/ -x --tb=short
uv run licensecheck --zero
```

## Report back
```
qa-gate — changed: py[✓/–] schema[✓/–]
| Step                | Status    | Notes                          |
|---------------------|-----------|--------------------------------|
| ruff format         | PASS/FAIL | N files reformatted            |
| ruff check --fix    | PASS/FAIL | N fixed / M remaining          |
| pyupgrade           | PASS/FAIL |                                |
| mypy + pyright      | PASS/FAIL | errors                         |
| bandit / safety     | PASS/FAIL | findings                       |
| boundary checks     | PASS/FAIL | first failing script           |
| schema sync         | PASS/FAIL |                                |
| doctests            | PASS/FAIL |                                |
| pytest (cov gate)   | PASS/FAIL | passed/failed, cov% vs 70      |
| interrogate/radon   | PASS/FAIL | docstring% vs 95 / complexity  |
| integration (all)   | PASS/–    |                                |
```
When all green: "qa-gate clean — safe to commit / run pr-prep." On any FAIL, show the failing output inline and stop.

## What NOT to do
- Do NOT invent ruff/pytest/mypy flags — use the ones above verbatim from `.pre-commit-config.yaml`.
- Do NOT lower `--cov-fail-under` (70) or interrogate `--fail-under` (95), and NEVER add `--no-verify`.
- Do NOT run mypy/pyright on the excluded dirs (`hexdag/cli`, `hexdag/studio`, `tests`, `examples`).
- Do NOT touch the `pyproject.toml` version or `CHANGELOG.md` — CI owns versioning.
