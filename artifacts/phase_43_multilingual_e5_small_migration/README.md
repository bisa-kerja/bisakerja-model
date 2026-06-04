# Phase 43 multilingual-E5-small Migration Baseline

Purpose: freeze current `intfloat/e5-base-v2` runtime and quality evidence before any multilingual-E5-small implementation.

Files:

- `e5_base_runtime_baseline.json` — E5-base runtime, endpoint, resource, command, git, and artifact-hash evidence.
- `e5_base_quality_baseline.json` — Phase 25 metrics, calibration, distribution, reranking examples, and API handoff evidence.
- `migration_decision_risk_register.json` — migration hypothesis, risk register, go/no-go thresholds, namespace, version naming, and staging-only decision.

Rules:

- Do not overwrite Phase 25 artifacts.
- Do not change staging/production embedding defaults in this phase.
- Phase 44 must measure embedding drift before retraining or runtime swap.
