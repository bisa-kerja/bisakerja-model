# Staging Promotion and Production Guardrails

Use this decision record before making multilingual-E5-small default for broader staging/demo traffic. This document does not approve production rollout.

## Decision

Current decision: `staging-experiment-only`.

Reasons:

- Phase 43 approved multilingual-E5-small for staging experiment only, with Phase 25 E5-base preserved as rollback.
- Phase 44 rejected direct embedding replacement and required retraining plus recalibration.
- Phase 45 trained multilingual-E5-small and selected it for staging shadow validation.
- Phase 46 produced self-contained calibrated artifacts and handoff fixtures.
- Phase 47 verified Model API runtime support for both Phase 46 and Phase 25 artifact roots.
- Phase 48 produced repo-side staging smoke, shadow comparison, and rollback commands, but live staging smoke and rollback drill still need operator credentials and approval.

The model must not be marked production-ready until live staging evidence, rollback drill, and human/reviewer validation gates pass.

## Default staging env

Use Phase 46 only for staging experiment traffic:

```env
MODEL_API_ENV=production
MODEL_API_ARTIFACT_ROOT=artifacts/phase_46_calibration_model_card_manifest_handoff_refresh
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/multilingual-e5-small
MODEL_API_WARMUP_ON_STARTUP=true
MODEL_API_WARMUP_REQUIRED=true
SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers
```

Rollback env:

```env
MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2
```

## VPS Docker deployment

```bash
cd /opt/bisakerja-model-api
MODEL_API_IMAGE=ghcr.io/bisa-kerja/bisakerja-model:sha-<candidate-sha> \
MODEL_API_ENV_FILE=.env.production \
COMPOSE_PROJECT_NAME=bisakerja-model-api \
  docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api

curl -fsS http://127.0.0.1:3004/live
curl -fsS http://127.0.0.1:3004/health
curl -fsS http://127.0.0.1:3004/ready
curl -fsS -H "Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:3004/model-info
```

Do not write service tokens, `DATABASE_URL`, `OPENROUTER_API_KEY`, raw CV text, uploaded bytes, or auth headers into reports.

## Nginx checks

- Proxy only Backend-approved routes to Model API.
- Keep `/model-info` and internal inference endpoints protected by `Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}`.
- Set request body limit high enough for approved CV uploads and low enough to reject abusive files.
- Set proxy timeout below Backend user-facing timeout budget.
- Verify 401 on missing token and no private header leakage in access/error logs.

## Cloud Run checks

If Cloud Run is used:

- Set the same env vars as the VPS Docker revision.
- Allocate memory and CPU for TensorFlow plus sentence-transformers cache warmup.
- Keep min instances or warmup policy aligned with `MODEL_API_WARMUP_REQUIRED=true`.
- Mount no writable volume over `SENTENCE_TRANSFORMERS_HOME` unless it already contains `intfloat/multilingual-e5-small`.
- Confirm `/ready.embeddingModel=intfloat/multilingual-e5-small` and artifact hash after each revision.

## Hugging Face staging caveats

HF Spaces can be used only as staging/demo support, not production evidence, unless resource, auth, persistence boundary, cold-start, and network timeout behavior match the target runtime.

- Do not call HF demo URLs from production Backend.
- Record cold-start and warm latency separately.
- Ensure protected endpoints are not public.
- Keep Phase 25 rollback on the production-like runtime.

## Monitoring checklist

Monitor these before any broader traffic:

- timeout rate
- `MODEL_NOT_READY` count
- internal inference latency p50/p95/p99
- Backend public route latency p50/p95/p99
- memory RSS and OOM restarts
- CPU saturation
- score distribution drift versus Phase 25 and Phase 46 fixtures
- recommendation count and empty-candidate rate
- backend downstream errors
- model metadata mismatch
- private field leakage alerts

## Production blockers

Production rollout remains blocked until all pass:

- live staging Backend public AI CV Analyzer smoke
- old-vs-new shadow comparison on frozen fixtures
- rollback drill to Phase 25 E5-base with evidence
- human/reviewer validation at useful scale
- slice coverage for language, role, seniority, score band, pair type, CV quality
- calibration confidence for job-fit, ATS, and recommendation ranking
- public response contract safety and private leakage audit
- privacy review for CV handling, logs, and report artifacts
- cost/resource monitoring with staging traffic
- operator-owned deploy and rollback approvals

## Runtime constant safety

Do not change runtime constants or hardcoded embedding defaults directly. Promote by changing `MODEL_API_ARTIFACT_ROOT` and `MODEL_API_EXPECTED_EMBEDDING_MODEL`, then verify `/ready` and `/model-info` metadata. Future agents must complete migration validation before changing default env examples.

## Rejected artifact policy

If multilingual-E5-small is later rejected:

- Keep reports and artifacts for audit history.
- Do not reference Phase 46 artifact paths in default env examples.
- Mark the model card and promotion report as `rejected`.
- Keep Phase 25 E5-base rollback/default env visible.
