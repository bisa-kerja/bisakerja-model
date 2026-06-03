# AI CV Analyzer Staging Readiness Gate

Final decision: `go`

## Test commands

- `model_api`: `python -m unittest tests.test_phase_33_cv_analyzer_request_compatibility tests.test_phase_34_cv_analyzer_response_compatibility tests.test_phase_31_release_gate`
- `backend`: `cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer tests/unit/shared/model-api.schema.test.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts tests/integration/contracts/fixture-contracts.test.ts`
- `phase36_report`: `python scripts/verify_phase_36_ai_cv_analyzer_staging_gate.py`

## Checks

- [x] `model_api_tests_cover_phase_36_scope`
- [x] `backend_tests_cover_phase_36_scope`
- [x] `cross_repo_fixture_path_declared`
- [x] `openapi_public_success_contract_frozen`
- [x] `openapi_error_statuses_cover_downstream_failures`
- [x] `backend_downstream_errors_fail_closed`
- [x] `model_api_rejects_unsafe_input_deterministically`
- [x] `security_privacy_boundary_verified`
- [x] `language_behavior_verified`
- [x] `failure_matrix_covers_required_cases`
- [x] `previous_release_and_contract_gates_passed`

## Failure coverage

- invalid file type
- file too large
- no active CV
- job not found
- bookmark not owned
- empty candidates
- Model API 422
- Model API invalid response
- timeout
- model not ready
- GenAI invalid JSON
- GenAI timeout

## Fallback coverage

- `deterministic_fallback`: Backend wrapper fallback creates English OpenAPI-compatible prose and generatedCv.available=false
- `invalid_genai_output`: Unsafe/invalid GenAI output is rejected before persistence/frontend response
- `score_integrity`: Wrapper/fallback path preserves model scores, IDs, order, and model metadata

## Remaining risks

- none
