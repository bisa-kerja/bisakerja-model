# Phase 38 AI CV Analyzer Runtime Gate

Decision: `go`

## Checks
- PASS `warmup_command_exists_and_uses_sanitized_fixture`
- PASS `readiness_requires_model_api_ready_and_warmup_policy`
- PASS `latency_budget_evidence_records_breakdown`
- PASS `timeout_alignment_documented`
- PASS `hf_e5_cache_guidance_documented`
- PASS `staging_smoke_and_rollback_documented`

## Commands
- `model_api_warmup`: `python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url http://127.0.0.1:8000 --token ${MODEL_API_SERVICE_TOKEN} --latency-budget-ms 30000 --output reports/ai_cv_analyzer_warmup_staging.json`
- `phase38_gate`: `python scripts/verify_phase_38_ai_cv_analyzer_runtime_gate.py --write`
