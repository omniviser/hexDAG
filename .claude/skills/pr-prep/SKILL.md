---
name: pr-prep
description: End-to-end release prep — run qa-gate, verify the branch name matches the enforced regex, make a semantic commit, push, and open a PR to main via the gh CLI (GitHub). No version bump, no CHANGELOG edit, no CI watch. Triggers "open a PR", "prep a PR", "ship this", "raise a pull request", "release this".
argument-hint: "[PR title] — omit to derive from the last commit"
---

Orchestrate a safe "release" = QA gate → branch check → semantic commit → push → PR to `main`. Versioning/CHANGELOG are **CI-owned**. Stop after the PR is open — **no CI watch, no version bump.**

## Why this skill exists
hexDAG is on **GitHub** (`github.com/omniviser/hexDAG`), so PRs go through the `gh` CLI, not any MCP. The branch-name regex is enforced by a pre-commit hook; the pre-push hooks add integration tests + license check. This skill guarantees the tree is green, the branch name is legal, and the PR is well-formed.

## Pre-conditions (hard — refuse and explain if unmet)
1. **Not on `main`** and **not detached**. Current branch must match:
   `^(ci|dependabot/pip|docs|experiment|feat|fix|refactor|test)/[A-Za-z0-9._-]+$`
   If currently on `main`, create a compliant branch first (see Step 1).
2. `gh` is authenticated (`gh auth status`).
3. qa-gate is **green** (including the Step 11 pre-push parity, since push runs those hooks).

## Step 1 — Verify / create the branch
```bash
BRANCH=$(git symbolic-ref --short HEAD)
if [ "$BRANCH" = "main" ]; then
  echo "On main — create a feature branch, e.g.: git switch -c feat/<slug>"; exit 1
fi
echo "$BRANCH" | grep -qE '^(ci|dependabot/pip|docs|experiment|feat|fix|refactor|test)/[A-Za-z0-9._-]+$' \
  && echo "branch OK: $BRANCH" || { echo "INVALID BRANCH NAME: $BRANCH"; exit 1; }
```

## Step 2 — Run the full gate
Invoke the **qa-gate** skill with `all` (so the pre-push integration tests + license check run too). Do not proceed unless every step is PASS.

## Step 3 — Semantic commit
Invoke the **commit** skill for any staged/unstaged work (skip if the tree is clean and the branch is already ahead of `main`).

## Step 4 — Push the branch
```bash
git push -u origin "$BRANCH"
```
The **pre-push** hooks run here (branch-name, integration tests, license-check). If they bounce, fix and re-push — never `--no-verify`.

## Step 5 — Open the PR to `main` via gh
Derive the title from the argument or the last commit subject; it MUST be a conventional-commit line (e.g. `feat(compiler): deep-scan dependency inference`).

First check for an existing open PR (avoid duplicates):
```bash
gh pr list --head "$BRANCH" --state open
```
If one exists, report its URL and stop. Otherwise:
```bash
gh pr create --base main --head "$BRANCH" \
  --title "<type>(<scope>): <subject>" \
  --body "$(cat <<'EOF'
## Summary
<what changed and why>

## Test evidence
- qa-gate: PASS (cov ≥ 70%, all boundary checks green)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Step 6 — Report and STOP
Return the PR URL. **Do NOT** watch CI, poll build status, or merge — reviewers merge; CI owns versioning.

## Report back
```
pr-prep
- branch:  <BRANCH>  (regex OK)
- qa-gate: PASS
- commit:  <sha> <subject>
- push:    origin/<BRANCH> (pre-push hooks PASS)
- PR:      #<id>  <title>  ->  main   <url>
- next:    reviewer merges; CI handles versioning. No action from us.
```

## What NOT to do
- Do NOT open the PR against a branch other than `main`.
- Do NOT bump the `pyproject.toml` version or append to `CHANGELOG.md`.
- Do NOT auto-merge the PR or bypass branch protection.
- Do NOT watch or re-run CI.
- Do NOT push with `--no-verify` or from `main`/detached HEAD.
