---
name: worktree
description: Multi-agent worktree hygiene — report git status and stash list, surface other sessions' untracked files, and create/destroy isolated git worktrees under .claude/worktrees on demand. Triggers "give me an isolated worktree", "is the tree clean", "set up a worktree", "clean up my worktree", "claim a clean tree".
argument-hint: "add <name> <branch> [base] | remove <name> | status"
---

Manage a clean, isolated working tree so parallel agents don't clobber each other. Default action is a **read-only status report**; worktree create/remove happens only on explicit request.

## Why this skill exists
Multiple Claude sessions share one checkout. The dangerous shared surfaces are (1) the **stash stack** — `git stash` is global, so one agent's `stash pop` can grab another's — and (2) **untracked files** another session created but hasn't committed. Isolated `git worktree`s give each parallel task its own checkout + branch on the same object store, sidestepping both.

## Action: `status` (default, read-only)
```bash
git status --short --branch
git stash list
git worktree list
git ls-files --others --exclude-standard
```
Report: current branch, ahead/behind, staged/unstaged, **every** stash entry (warn if >0), existing worktrees, and untracked files. If untracked files or stashes exist that you did not create, **warn loudly** and do not stash/pop.

## Action: `add <name> <branch> [base]`
`base` defaults to `main`. Branch name must match the enforced regex.
```bash
BASE=${3:-main}
git fetch origin "$BASE"
echo "$2" | grep -qE '^(ci|dependabot/pip|docs|experiment|feat|fix|refactor|test)/[A-Za-z0-9._-]+$' \
  || { echo "INVALID branch: $2"; exit 1; }
git worktree add ".claude/worktrees/$1" -b "$2" "origin/$BASE"
```
Report the new path and remind the agent to work inside it. Ensure `.claude/worktrees/` is git-ignored (add it to `.gitignore` if not).

## Action: `remove <name>`
```bash
git -C ".claude/worktrees/$1" status --short   # refuse if uncommitted changes, unless caller says --force
git worktree remove ".claude/worktrees/$1"
git worktree prune
```

## Verify a worktree's base before merging its work (multi-agent trap)
The harness creates a worktree from the main tree's current HEAD. If a sibling flipped the branch, a new worktree can inherit the wrong lineage. Before trusting/merging any worktree's commits:
```bash
git -C .claude/worktrees/<name> log --oneline -3
git -C .claude/worktrees/<name> merge-base --is-ancestor origin/main HEAD && echo "based on main: OK" || echo "WRONG BASE"
```

## The multi-agent stash hazard (call out every time)
- `git stash`/`stash pop` operate on a **single global stack** shared by all sessions on this checkout. Never `stash pop`/`drop` blindly — you may destroy another agent's WIP.
- Prefer a **worktree** over stashing to switch context.
- If you must stash, use a labeled entry (`git stash push -m "session-<id>: <desc>"`) and only pop **that** entry by `stash@{n}` after confirming the label.

## Report back
```
worktree (<action>)
- branch:    <br> (ahead N / behind M)
- staged:    <n>   unstaged: <n>
- untracked: <list — flag if not yours>
- stashes:   <count>  [WARN if >0: shared global stack]
- worktrees: <list>
- action:    <added/removed/none>
```

## What NOT to do
- Do NOT `git stash pop`/`drop` without a confirmed matching label — it's a shared stack.
- Do NOT create worktrees outside `.claude/worktrees/`, and do NOT commit that directory.
- Do NOT `git checkout`/`switch` in the shared checkout to change another agent's branch — use a worktree.
- Do NOT `worktree remove --force` over uncommitted changes without explicit confirmation.
