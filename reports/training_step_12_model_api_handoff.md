# Training Step 12 Model API Handoff

Generated at: `2026-06-04T07:55:00+00:00`
Final decision: **pass**

## Acceptance

- PASS `training_exports_only_handoff_fixtures`
- PASS `candidate_reranking_uses_backend_candidate_ids`
- PASS `training_does_not_invent_or_hydrate_jobs`
- PASS `model_core_handoff_matches_contract`
- PASS `backend_behavior_remains_outside_training`

## Checks

### phase_21_candidate_boundary

Status: **PASS**
Report: `reports/phase_21_backend_candidate_reranking.json`

Confirms candidate reranking requires Backend-provided candidate job IDs, does not require a production static job index, and has zero candidate constraint violations.

### phase_23_contract_validation

Status: **PASS**
Report: `reports/phase_23_model_api_contract_validation.json`

Positive fixtures have zero violations; all eight negative fixtures are rejected before persistence or user response.

### phase_23_contract_artifacts

Status: **PASS**
Artifact directory: `artifacts/phase_23_model_api_contract_validation/`

Required contract fixtures, fallback validation, model-core contract, and validation result artifacts are present.

### phase_25_handoff_fixtures

Status: **PASS**
Report: `reports/phase_25_model_api_handoff_fixtures.json`
Fixtures: `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json`
Validation: `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json`

Positive recommendations are members of the Backend candidate set, contain no Backend-wrapper-owned fields, and keep scores inside integer 0-100 bounds.

### backend_source_boundary

Status: **PASS**

Backend source remains external to this model repository. Versioned contract snapshots and fixtures are limited to `artifacts/backend_model_api_contract/` and `references/docs/generated/openapi.json`.
