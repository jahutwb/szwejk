# Project Agent Workflow

This repository uses a local Codex stack.

## Primary Workflow

Use GSD as the main workflow layer:
- start a greenfield project with `$gsd-new-project`
- for existing code, start with `$gsd-map-codebase`
- treat GSD planning/state files as the source of truth

Expected state files:
- `PROJECT.md`
- `REQUIREMENTS.md`
- `ROADMAP.md`
- `STATE.md`
- `.planning/`

## Supporting Skills

Use ECC skills as support, not as a replacement for GSD:
- `coding-standards`
- `tdd-workflow`
- `verification-loop`
- `security-review`
- `backend-patterns`
- `frontend-patterns`
- `api-design`
- `eval-harness`
- `e2e-testing`
- `strategic-compact`

## Working Rules

- Plan before major edits.
- Prefer test-first changes.
- Run verification before calling work complete.
- Do security review for auth, input validation, secrets, file access, and external integrations.
- Keep global Codex assumptions out of this repo; use only local project config and skills.

## Optional Tools

- If Axon is configured in this repo, use it for code graph context and impact analysis.
- Use Shannon only at the end for runnable web applications.
