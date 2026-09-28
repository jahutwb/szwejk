# Szwejk

An alignment-first NLP and editorial pipeline for building a Polish-Czech hybrid edition of *The Good Soldier Svejk*.

The project combines corpus ingestion and normalization, hierarchical text alignment, hybridization policy, review tooling, and deterministic generation. It is also used as a practical environment for structured AI-assisted software development with local coding agents.

## What is in this repository

- `src/szwejk/ingest` - source ingestion
- `src/szwejk/normalize` - corpus normalization
- `src/szwejk/align` - alignment logic
- `src/szwejk/policy` - hybridization policy
- `src/szwejk/review` and `review_bundle.py` - review and QA tooling
- `src/szwejk/generate` - output generation
- `tests/unit` and `tests/integration` - automated verification
- `docs/` - corpus, alignment and hybridization decisions

## AI-assisted engineering workflow

This repository is intentionally structured for a local Codex workflow. The agent instructions in [AGENTS.md](AGENTS.md) require planning before major edits, test-first changes where practical, verification before completion, security review for sensitive surfaces, and project-local context.

The bootstrap guide is in [START_HERE.md](START_HERE.md).

The goal is not to let an agent modify code unchecked, but to keep AI-assisted changes inside a reviewable workflow with explicit requirements, tests and verification.

## Technology

- Python 3.12+
- package layout under `src/`
- unit and integration tests
- local Codex skills and agent instructions

## Project context

The repository focuses on the engineering and editorial problems involved in aligning Polish and Czech literary text and turning the resulting structured representation into a controlled hybrid edition. Supporting documents describe corpus layout, hybridization policy, editorial decisions and review contracts.

See [docs/](docs/) for the project documentation.
