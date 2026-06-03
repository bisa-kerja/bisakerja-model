# Documentation Index

Cross-workspace documentation for Bisakerja Model Workspace.

This repository documents the model side only. The Backend API is a separate repository: <https://github.com/bisa-kerja/bisakerja-api>.

## Core Docs

| Document                             | Purpose                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `project-structure.md`               | Model-repo layout, folder ownership, and safe restructuring rules.       |
| `architecture/service-boundaries.md` | Backend API, Model API, training, artifacts, and legacy boundaries.      |
| `runbooks/local-development.md`      | Local setup for Model API, training notebooks, tests, and release gates. |
| `runbooks/release-gates.md`          | Production evidence gates, expected checks, and failure handling.        |

## Workspace Docs

| Workspace          | README                            |
| ------------------ | --------------------------------- |
| Root workspace     | `../README.md`                    |
| Training           | `../training/README.md`           |
| Training notebooks | `../training/notebooks/README.md` |
| Model API          | `../model_api/README.md`          |
| Scripts            | `../scripts/README.md`            |
| Tests              | `../tests/README.md`              |
| Artifacts          | `../artifacts/README.md`          |
| Reports            | `../reports/README.md`            |
| Legacy snapshot    | `../legacy/README.md`             |

## Backend Contract Input

Backend contracts should come from <https://github.com/bisa-kerja/bisakerja-api> or generated fixtures copied from that repository into explicit model-release artifacts. Backend source is not part of this repository.

## Documentation Rules

- Write durable docs in English.
- Keep implementation paths exact.
- Update docs when behavior, env vars, API contracts, artifacts, or release gates change.
- Do not document speculative features as available behavior.
- Do not include secrets, raw CV text, DB URLs, tokens, or unrelated PII.
