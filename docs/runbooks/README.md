# Runbooks

Operational guides for local development, staging validation, release gates, and rollback.

## Files

| File                                          | Purpose                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| `local-development.md`                        | Local Model API, training runtime, notebook kernel, and targeted tests.          |
| `hugging-face-docker-deployment.md`           | Docker deployment runbook for Hugging Face Spaces.                               |
| `ai-cv-analyzer-staging-runtime.md`           | AI CV Analyzer warmup, readiness, latency, cache, smoke, and rollback.           |
| `ai-cv-analyzer-product-copy-localization.md` | AI CV Analyzer public-copy language policy, templates, safety, and verification. |
| `release-gates.md`                            | Production evidence gates, expected failure modes, and release notes.            |

Keep runbooks in English and update them when env vars, runtime versions, gates, or service boundaries change.
