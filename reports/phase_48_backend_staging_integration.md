# Phase 48 Backend/Staging Integration, Shadow Comparison, and Rollback Plan

Status: `ready_for_staging_execution`
Generated at: 2026-06-04T09:21:54Z

Repo-side staging harness is ready. Live deployment, Backend token smoke, old-vs-new shadow run, and rollback drill still require operator credentials and approval.

## Checks

- PASS `phase47_runtime_support_complete`
- PASS `phase46_artifact_selected_for_staging`
- PASS `phase25_rollback_artifact_preserved`
- PASS `staging_deploy_env_documented`
- PASS `readiness_metadata_gate_documented`
- PASS `warmup_and_direct_model_smoke_supported`
- PASS `backend_public_smoke_supported`
- PASS `shadow_compare_and_allowed_deltas_defined`
- PASS `failure_behavior_matrix_defined`
- PASS `rollback_commands_frozen`
- PASS `secrets_redacted_in_evidence`

## Required Live Evidence

- Deploy staging Model API revision with Phase 46 artifact env.
- Run readiness probes and record /model-info metadata.
- Run direct Model API multipart smoke.
- Run Backend public AI CV Analyzer smoke with non-production token.
- Run old-vs-new shadow comparison over frozen fixtures.
- Run rollback drill to Phase 25 E5-base artifact path.

## Commands

- `phase48_gate`: `python scripts/verify_phase_48_backend_staging_integration.py --write`
- `phase48_new_model_api_smoke`: `python scripts/run_phase_48_backend_staging_shadow.py --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --output reports/phase_48_backend_staging_shadow_report.json`
- `phase48_backend_public_smoke`: `python scripts/run_phase_48_backend_staging_shadow.py --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --backend-api-url ${BACKEND_API_URL} --user-access-token ${USER_ACCESS_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --output reports/phase_48_backend_staging_shadow_report.json`
- `phase48_shadow_compare`: `python scripts/run_phase_48_backend_staging_shadow.py --old-model-api-url ${OLD_MODEL_API_URL} --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --score-delta-threshold 5 --output reports/phase_48_backend_staging_shadow_report.json`
- `rollback_to_phase25`: `MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2 docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api`
