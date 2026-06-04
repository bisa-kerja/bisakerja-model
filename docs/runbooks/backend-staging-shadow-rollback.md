# Backend Staging Shadow and Rollback Runbook

Run this before routing broader staging/demo traffic to the multilingual-E5-small Model API artifact.

## Scope

- Backend public response shape remains `cv-analysis-v2`.
- Backend keeps database persistence, public recommendation hydration, auth, and wrapper copy.
- Model API runs model-core inference only.
- New rollout starts in staging only.

## Deploy staging Model API revision

Use versioned env vars and image-baked E5 cache. Do not change Backend public contracts.

```env
MODEL_API_ENV=production
MODEL_API_ARTIFACT_ROOT=artifacts/phase_46_calibration_model_card_manifest_handoff_refresh
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/multilingual-e5-small
MODEL_API_WARMUP_ON_STARTUP=true
MODEL_API_WARMUP_REQUIRED=true
SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers
```

Cache requirements:

- Keep `SENTENCE_TRANSFORMERS_HOME` pointed at the image-baked cache path.
- Do not mount a Docker volume over `SENTENCE_TRANSFORMERS_HOME`; it hides the model predownloaded during image build.
- Do not write service tokens, `DATABASE_URL`, `OPENROUTER_API_KEY`, raw CV text, or uploaded bytes into reports.

Docker Compose deploy example:

```bash
cd /opt/bisakerja-model-api
MODEL_API_IMAGE=ghcr.io/bisa-kerja/bisakerja-model:sha-<candidate-sha> \
MODEL_API_ENV_FILE=.env.production \
COMPOSE_PROJECT_NAME=bisakerja-model-api \
  docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api
```

## Readiness and metadata gate

Probe internal Model API before Backend traffic:

```bash
curl -fsS http://127.0.0.1:3004/live
curl -fsS http://127.0.0.1:3004/health
curl -fsS http://127.0.0.1:3004/ready
curl -fsS -H "Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:3004/model-info
```

Required metadata:

- `/ready.ready=true`
- `/ready.embeddingModel=intfloat/multilingual-e5-small`
- `/ready.artifactPhase=phase_46_calibration_model_card_manifest_handoff_refresh`
- `/ready.artifactHash` present
- `/model-info.embeddingPolicy.embeddingModel=intfloat/multilingual-e5-small`
- `/model-info.model.artifactPhase=phase_46_calibration_model_card_manifest_handoff_refresh`
- `/model-info.model.artifact.sha256` present

## Warm runtime

Warm TensorFlow, multilingual-E5-small, PDF parser, and in-process caches before Backend traffic.

```bash
python scripts/run_phase_48_backend_staging_shadow.py \
  --new-model-api-url "${MODEL_API_URL:-http://127.0.0.1:3004}" \
  --model-api-token "${MODEL_API_SERVICE_TOKEN}" \
  --fixture-pdf artifacts/smoke/sanitized-cv.pdf \
  --output reports/phase_48_backend_staging_shadow_report.json
```

The command posts sanitized multipart data to `/internal/model/cv-analysis` with `x-model-api-include-observability=true` and records `parseLatencyMs`, `embeddingLatencyMs`, `tensorflowLatencyMs`, and `totalLatencyMs`.

Latency gates:

- first internal CV analysis <= `30000 ms`
- warm internal CV analysis <= `5000 ms`
- Backend public smoke <= `5000 ms` unless staging infra sets stricter budget

## Direct Model API smoke

Direct smoke must validate model-core response only:

- `schemaVersion=model-core-cv-analysis-v1`
- `jobFitAlignment.score` present
- `candidateReranking.recommendations` present
- `atsFriendliness.score` present
- no Backend-owned fields such as `topActionables`, `sectionReviews`, `companyName`, `nextStep`
- no private field leakage: `storageKey`, `rawCv`, `cvText`, `authorization`, `MODEL_API_SERVICE_TOKEN`, `DATABASE_URL`, `OPENROUTER_API_KEY`, artifact paths

## Backend public smoke

Seed or select deterministic non-production Backend data, then run public route smoke. Use `persistResult=false` by default.

```bash
python scripts/run_phase_48_backend_staging_shadow.py \
  --new-model-api-url "${MODEL_API_URL}" \
  --model-api-token "${MODEL_API_SERVICE_TOKEN}" \
  --backend-api-url "${BACKEND_API_URL}" \
  --user-access-token "${USER_ACCESS_TOKEN}" \
  --fixture-pdf artifacts/smoke/sanitized-cv.pdf \
  --compare-source JOB_SEARCH \
  --output reports/phase_48_backend_staging_shadow_report.json
```

For direct candidate scenario:

```bash
python scripts/run_phase_48_backend_staging_shadow.py \
  --new-model-api-url "${MODEL_API_URL}" \
  --model-api-token "${MODEL_API_SERVICE_TOKEN}" \
  --backend-api-url "${BACKEND_API_URL}" \
  --user-access-token "${USER_ACCESS_TOKEN}" \
  --fixture-pdf artifacts/smoke/sanitized-cv.pdf \
  --compare-source DIRECT_JOB_DETAIL \
  --direct-job-id "${DIRECT_JOB_ID}" \
  --output reports/phase_48_backend_staging_shadow_report.json
```

Required Backend public checks:

- HTTP 200
- response envelope contains `{ success, message, data, meta }`
- `data.analysisResult.schemaVersion=cv-analysis-v2`
- `jobRecommendations` hydrated by Backend, length <= 5
- public response includes model metadata
- no private field leakage
- persistence behavior matches `persistResult=false` or explicit `--persist-result`

## Shadow compare old vs new

Run old E5-base and new multilingual-E5-small services against same sanitized fixture set.

```bash
python scripts/run_phase_48_backend_staging_shadow.py \
  --old-model-api-url "${OLD_MODEL_API_URL}" \
  --new-model-api-url "${MODEL_API_URL}" \
  --model-api-token "${MODEL_API_SERVICE_TOKEN}" \
  --fixture-pdf artifacts/smoke/sanitized-cv.pdf \
  --score-delta-threshold 5 \
  --output reports/phase_48_backend_staging_shadow_report.json
```

Allowed deltas and review triggers:

- Review any job-fit score delta > 5 points.
- Review any top recommendation rank swap, especially when top score delta > 5 points.
- Review `matchLevel` changes, `matchedSkills` changes, and `missingSkills` changes.
- Review ATS stability: score delta > 5 points, new parsing issue, or parse quality downgrade.
- Review public copy changes because Backend wrapper consumes model-core evidence and rank order.
- Block rollout when high-fit recall evidence regresses below Phase 43 threshold or language regression appears.

The script uses `compare_shadow_outputs` and records `score_delta_threshold`, score deltas, ranked job IDs, top recommendation rank swap, matched skills, missing skills, and ATS stability flags.

## Failure behavior matrix

Validate these before broader traffic:

| Scenario               | Expected behavior                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| Model not ready        | Model API returns 503 `MODEL_NOT_READY`; Backend handles retry/fallback without private leakage.                |
| Timeout                | Request fails within configured timeout; no partial private payload is returned.                                |
| Invalid PDF            | 4xx validation error; no raw file bytes or parser internals leak.                                               |
| Empty candidates       | Backend/Model API rejects invalid candidate set or returns safe empty recommendation result; never invent jobs. |
| Invalid token          | Protected `/model-info` and inference endpoints return 401.                                                     |
| Rollback artifact path | Phase 25 artifact env starts and reports E5-base metadata.                                                      |

## Freeze rollback commands

Rollback to E5-base without changing image tag:

```bash
cd /opt/bisakerja-model-api
# edit .env.production
MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2

MODEL_API_ENV_FILE=.env.production \
COMPOSE_PROJECT_NAME=bisakerja-model-api \
  docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api

curl -fsS http://127.0.0.1:3004/live
curl -fsS http://127.0.0.1:3004/health
curl -fsS http://127.0.0.1:3004/ready
curl -fsS -H "Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:3004/model-info
```

Rollback is complete only when:

- `/ready.embeddingModel=intfloat/e5-base-v2`
- `/ready.artifactPhase=phase_25_tensorflow_training_delivery`
- `/model-info.embeddingPolicy.embeddingModel=intfloat/e5-base-v2`
- direct Model API smoke passes with Phase 25 artifact metadata

## Evidence files

Generate repo-side gate evidence:

```bash
python scripts/verify_phase_48_backend_staging_integration.py --write
```

Generate live staging evidence after credentials and deploy approval are available:

```bash
python scripts/run_phase_48_backend_staging_shadow.py \
  --old-model-api-url "${OLD_MODEL_API_URL}" \
  --new-model-api-url "${MODEL_API_URL}" \
  --model-api-token "${MODEL_API_SERVICE_TOKEN}" \
  --backend-api-url "${BACKEND_API_URL}" \
  --user-access-token "${USER_ACCESS_TOKEN}" \
  --fixture-pdf artifacts/smoke/sanitized-cv.pdf \
  --score-delta-threshold 5 \
  --output reports/phase_48_backend_staging_shadow_report.json
```

The generated report redacts tokens as `<redacted>` and must stay free of raw CV text, auth headers, DB URLs, OpenRouter keys, prompts, and artifact internals beyond approved metadata.
