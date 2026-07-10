---
name: commit
description: Create a conventional-commit-style semantic commit (feat|fix|refactor|test|docs|ci|experiment|chore). Stages files by name, enforces the shape, appends the Co-Authored-By trailer, and never uses --no-verify or --amend after a hook failure. Triggers "commit this", "make a commit", "semantic commit", "commit my changes".
argument-hint: "[type] [short subject]"
---

Create a well-formed semantic commit that passes every pre-commit hook on the first try.

## Why this skill exists
hexDAG has **no commitizen** — commit messages are not tool-enforced — but the branch-name regex (`feat|fix|refactor|test|docs|ci|experiment`) and the team convention are conventional-commit style, and CI/readers rely on it. This skill produces a clean message and, crucially, lets the full pre-commit hook chain run so the commit doesn't bounce.

## Pre-conditions (hard)
- qa-gate (or at least qa-fast) is green for the changes being committed.
- Not committing `.env`, `*.key`, `*.pem`, `secrets.json`, `*.pyc`, `__pycache__/`, or anything gitleaks would flag.
- On a valid feature branch (branch-name regex is enforced by a pre-commit hook — check early).

## Step 1 — Assess changes
```bash
git status --short
git diff --stat HEAD
```

## Step 2 — Stage files by name (never `git add -A` / `git add .`)
```bash
git add <explicit> <paths> <here>
```

## Step 3 — Choose type + optional scope
Types (match the branch regex + common extras): `feat` (new capability), `fix`, `refactor`, `test`, `docs`, `ci`, `experiment`, `chore`, `build`. A trailing `!` marks a breaking change (`feat!:`).
Suggested scopes (hexDAG layers): `kernel`, `stdlib`, `drivers`, `api`, `compiler`, `cli`, `nodes`, `adapters`, `ports`, `docs`, `tests`.
Subject: imperative mood, no trailing period, ≤ ~72 chars. Format: `<type>(<optional scope>): <subject>`.

## Step 4 — Commit (let all hooks run)
```bash
git commit -m "$(cat <<'EOF'
feat(compiler): deep-scan dependency inference across all spec strings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

## If a pre-commit hook fails
1. Read the failure. Fix the code (ruff/mypy/pytest/boundary-check/secrets) or the message.
2. Re-stage the affected files by name (ruff auto-fixes may have modified files — re-`git add` them).
3. Create a **NEW** commit. Never `git commit --amend` after a hook failure, and **never** `--no-verify`.

## Report back
- Commit SHA + one-line summary, and which hooks ran/passed.
- If blocked: which hook failed and the exact fix applied before the retry.

## What NOT to do
- NEVER `--no-verify`, NEVER `--amend` to paper over a failed hook.
- NEVER `git add -A` / `git add .`.
- NEVER bump the `pyproject.toml` version or touch `CHANGELOG.md` — CI owns that.
- Do NOT invent commit types outside the set above (keeps history parseable and matches the branch regex).
