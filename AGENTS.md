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

## Known Backlog / Future Work

- **Phrase-to-token alignment (compound expansion candidates)**:
  The enrichment pipeline produces only 1:1 token pairs and symmetric N:M span
  candidates (both sides ≥ 2 tokens).  Czech compound ordinals / numerals like
  `jedensedmdesátého` (1 token) that correspond to `siedemdziesiąty pierwszy`
  (2 PL tokens) are currently unaligned.  Fix: add `compound_expansion_candidate`
  type in `align/enrichment.py` with asymmetric source_span / target_span.
  Cross-refs: `align/subtree.py::align_tree_units`, `align/enrichment.py`.

## Optional Tools

- If Axon is configured in this repo, use it for code graph context and impact analysis.
- Use Shannon only at the end for runnable web applications.
