---
name: yaml-check
description: Validate and lint changed hexDAG pipeline YAML with the project's own CLI (hexdag validate + hexdag lint), and confirm generated schemas are in sync. The hexDAG-native analog of a build-safety check for declarative pipelines. Triggers "validate my pipeline", "check the YAML", "lint the pipeline", "does this pipeline build", "yaml-check".
argument-hint: "[path/to/pipeline.yaml] — omit to auto-detect changed YAML"
---

Validate every changed pipeline YAML through hexDAG's real build/validation path, then lint it for best-practice warnings. Catches dangling refs, expression-var/node collisions, unknown `kind`s, and schema drift **before** runtime.

## Why this skill exists
hexDAG is YAML-first: a pipeline that builds cleanly is the declarative equivalent of a passing type-check. `hexdag validate` runs the same include-expansion + alias/custom-type registration + validator the builder uses, so a green validate means the pipeline will build. `hexdag lint` layers best-practice rules (unused nodes, missing deps, etc.) on top.

## Pre-conditions
- Run from repo root; `uv` available.

## Step 0 — Find changed pipeline YAML (or use the argument)
```bash
if [ -n "$1" ]; then
  YAMLS="$1"
else
  YAMLS=$( { git diff --name-only HEAD -- '*.yaml' '*.yml'; git ls-files --others --exclude-standard -- '*.yaml' '*.yml'; } | sort -u )
fi
echo "$YAMLS"
```
Filter to actual pipeline/system manifests (skip config like `.pre-commit-config.yaml`, CI files) — look for `kind: Pipeline` / `kind: System`.

## Step 1 — Validate each pipeline
```bash
for f in $YAMLS; do
  echo "== validate $f =="
  uv run hexdag validate "$f" --verbose
done
```
A non-zero exit = a build-blocking error (dangling ref, unknown `kind`, collision). Read the message; it usually names the node + field.

## Step 2 — Lint each pipeline
```bash
for f in $YAMLS; do
  echo "== lint $f =="
  uv run hexdag lint "$f"
done
```
Lint emits warnings (e.g. unused node, `{{name}}` near-miss of a real node, incomplete explicit `dependencies`). Treat `error`-severity lint findings as blocking; `warning`/`info` as advisory.

## Step 3 — Schema sync (only if you touched schemas/ or hexdag/compiler/)
```bash
uv run python scripts/check_yaml_schema_sync.py
uv run python scripts/check_schemas.py
```
If these fail, regenerate: `uv run python scripts/generate_schemas.py`, then re-run the check.

## Report back
```
yaml-check — N pipeline(s)
| File                    | validate | lint (err/warn) |
|-------------------------|----------|-----------------|
| pipelines/foo.yaml      | PASS     | 0 / 2           |
| pipelines/bar.yaml      | FAIL     | —               |
schema-sync: PASS/FAIL
```
On a validate FAIL, quote the node/field + the error and stop. Summarize lint warnings but don't block on them unless severity=error.

## What NOT to do
- Do NOT hand-edit generated files under `schemas/` — regenerate with `scripts/generate_schemas.py`.
- Do NOT silence a validate error by deleting the offending ref without understanding the data flow — check whether the node should exist or the ref is a typo (lint's "did you mean?" helps).
- Do NOT run these against non-pipeline YAML (CI/pre-commit configs will just error as "not a pipeline").
