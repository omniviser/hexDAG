---
name: hexdag-validate
description: Validate and lint hexDAG pipeline YAML with the hexdag CLI before running or committing. The build-safety check for declarative pipelines. Triggers "validate my pipeline", "check the YAML", "lint the pipeline", "does this pipeline build", "hexdag-validate".
argument-hint: "[path/to/pipeline.yaml] — omit to auto-detect changed YAML"
---

Run every changed pipeline YAML through hexDAG's real build/validation path, then lint it. Catches dangling refs, expression-var/node collisions, unknown `kind`s, and missing deps **before** runtime.

## Why this skill exists
`hexdag validate` runs the same include-expansion + alias/custom-type registration + validator that the builder uses, so a green validate means the pipeline will build. `hexdag lint` layers best-practice rules on top. Running these before you execute (or commit) turns runtime surprises into fast, precise build errors.

## Step 0 — Find pipelines (or use the argument)
```bash
if [ -n "$1" ]; then
  YAMLS="$1"
else
  YAMLS=$( { git diff --name-only HEAD -- '*.yaml' '*.yml'; git ls-files --others --exclude-standard -- '*.yaml' '*.yml'; } | sort -u )
fi
echo "$YAMLS"
```
Filter to actual manifests — look for `kind: Pipeline` / `kind: System`. Skip config YAML.

## Step 1 — Validate each pipeline
```bash
for f in $YAMLS; do echo "== validate $f =="; hexdag validate "$f" --verbose; done
```
A non-zero exit is a build-blocking error (dangling ref, unknown `kind`, var/node collision). The message names the node + field.

## Step 2 — Lint each pipeline
```bash
for f in $YAMLS; do echo "== lint $f =="; hexdag lint "$f"; done
```
Treat `error`-severity findings as blocking; `warning`/`info` as advisory (unused node, `{{name}}` near-miss of a real node, incomplete explicit `dependencies`).

## Report back
```
hexdag-validate — N pipeline(s)
| File                | validate | lint (err/warn) |
|---------------------|----------|-----------------|
| pipelines/foo.yaml  | PASS     | 0 / 2           |
```
On a validate FAIL, quote the node/field + the error and stop.

## What NOT to do
- Do NOT silence a validate error by deleting the offending ref without understanding the data flow — check whether the node should exist or the ref is a typo (lint's "did you mean?" helps).
- Do NOT run these against non-pipeline YAML (config/CI files just error as "not a pipeline").
