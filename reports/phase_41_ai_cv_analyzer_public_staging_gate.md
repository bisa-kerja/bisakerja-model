# Phase 41 AI CV Analyzer Public Staging Smoke

Decision: `go`

Phase 38.6 public Backend-to-Model smoke is superseded by this public staging smoke evidence.

## Checks
- PASS `public_smoke_script_exists`
- PASS `sanitized_pdf_fixture_exists`
- PASS `deterministic_candidate_fixture_documented`
- PASS `public_response_contract_validated`
- PASS `persistence_and_latency_recorded`
- PASS `private_field_leak_checks_present`
- PASS `multipart_excludes_cv_storage_metadata`
- PASS `payload_hygiene_test_exists`
- PASS `phase_38_6_superseded_by_phase_41`

## Commands
- `model_api_warmup`: `python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url ${MODEL_API_URL:-http://127.0.0.1:8000} --token ${MODEL_API_SERVICE_TOKEN} --latency-budget-ms 30000 --output reports/ai_cv_analyzer_warmup_staging.json`
- `backend_seed`: `cd references && bun run prisma:seed`
- `public_staging_smoke`: `python scripts/smoke_ai_cv_analyzer_public_staging.py --backend-api-url ${BACKEND_API_URL} --user-access-token ${USER_ACCESS_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --job-role 'Backend Developer' --language en --input-mode UPLOAD --compare-source JOB_SEARCH --latency-budget-ms 5000 --output reports/phase_41_ai_cv_analyzer_public_staging_smoke.json`
- `backend_payload_hygiene_tests`: `cd references && bun test --preload ./tests/preload-env.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts`
- `phase41_gate`: `python scripts/verify_phase_41_ai_cv_analyzer_public_staging_smoke.py --write`
