# Reports

Generated evidence reports for notebook phases, release gates, audits, and contract validation.

## Purpose

Reports make model readiness reviewable without reopening every notebook. They capture:

- training decisions and metrics
- validation and slice-risk evidence
- TensorFlow delivery evidence
- Model API contract and smoke-test results
- requirement coverage
- release-gate decisions

## Naming

Most files follow this pattern:

```text
phase_<number>_<topic>.json
phase_<number>_<topic>.md
```

JSON reports are machine-readable evidence. Markdown reports are human-readable summaries.

## Rules

- Prefer generated writes through `scripts/` or notebooks.
- Do not hand-edit generated reports unless the report explicitly has no generator.
- Keep reports in English.
- Keep paths stable because release manifests and tests reference them.
- Do not include secrets, raw CV text, service tokens, DB URLs, or unrelated PII.
