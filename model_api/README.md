# Bisakerja Model API Package

Production serving package for Phase 26. It is separate from `legacy/api` and consumes Phase 25 artifacts without retraining.

## Canonical references

- Backend payload/response: `references/docs/generated/openapi.json`
- Backend DB shape/enums: `references/prisma/schema.prisma`
- Phase 25 handoff: `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json`
- GenAI wrapper boundary: `artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json`
- OpenRouter base URL for OpenAI-compatible wrapper calls: `https://openrouter.ai/api/v1`
- OpenRouter model catalog: `https://openrouter.ai/models`

## Layout

- `app.py` — FastAPI app factory and HTTP route wiring boundary.
- `config.py` — environment/config/path resolution for Phase 25 artifacts, OpenAPI/Prisma refs, and disabled-by-default OpenRouter wrapper config.
- `contracts.py` — read-only OpenAPI/Prisma contract extraction helpers.
- `schemas.py` — backend-aligned model-core request/response shape names.
- `artifacts.py` — artifact manifest loading plus SHA-256/byte-size verification for runtime-required Phase 25 files.
- `custom_objects.py` — Phase 25 Keras custom-object registration boundary.
- `features.py` — approved six-feature vector order and normalization helpers.
- `pdf_parser.py` — deterministic request-scoped PDF parsing, section/contact/date evidence, and ATS penalties.
- `inference.py` — loaded-model service facade.
- `validators.py` — model-core response contract validators.
- `errors.py` — deterministic exception/error payload types.

## Runbook

### Runtime versions

Use Python `3.13.x` with TensorFlow `2.21.x`, Keras `3.14.x`, FastAPI, Uvicorn, NumPy, and SentenceTransformers. Phase 25 reproducibility records Python `3.13.11`; smoke fixture records TensorFlow `2.21.0` and Keras `3.14.1`.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow fastapi uvicorn sentence-transformers numpy
```

Serving-only dependency pinning lives in root `requirements.txt`; notebook/training-only packages stay in `training/requirements.txt` unless inference needs them.

### Required artifact paths

Defaults resolve from repo root:

- `MODEL_API_MODEL_PATH=artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras`
- `MODEL_API_TENSORFLOW_FEATURE_CONFIG=artifacts/phase_25_tensorflow_training_delivery/tensorflow_feature_config.json`
- `MODEL_API_FEATURE_CONFIG=artifacts/phase_25_tensorflow_training_delivery/feature_config.json`
- `MODEL_API_SCORE_CALIBRATION=artifacts/phase_25_tensorflow_training_delivery/score_calibration.json`
- `MODEL_API_MODEL_CARD=artifacts/phase_25_tensorflow_training_delivery/model_card.json`
- `MODEL_API_ARTIFACT_MANIFEST=artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`
- `MODEL_API_OPENAPI=references/docs/generated/openapi.json`
- `MODEL_API_PRISMA_SCHEMA=references/prisma/schema.prisma`

Startup verifies manifest SHA-256 and byte-size metadata before loading the Keras model.

### Environment knobs

```bash
export MODEL_API_ENV=staging
export MODEL_API_SERVICE_NAME=bisakerja-model-api
export MODEL_API_SERVICE_TOKEN=replace-with-internal-token
export MODEL_API_MAX_RECOMMENDATIONS=5
export MODEL_API_TIMEOUT_MS=30000
export MODEL_API_MAX_PDF_BYTES=5000000
export MODEL_API_MAX_PDF_PAGES=10
export MODEL_API_ENABLE_GENAI_WRAPPER=false
```

Set `MODEL_API_ENV=local` only for local experiments. Staging/production require `MODEL_API_SERVICE_TOKEN` for inference routes and reject TF-IDF, local-hash, and fallback embedding backends.

### Run API

```bash
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

`/health.ready=true` means artifacts verified and model loaded. Loader failure keeps inference unavailable and returns deterministic `model_not_ready` or `model_load_error` payloads.

### Example inference request

```bash
curl -X POST http://127.0.0.1:8000/inference/cv-analysis \
  -H 'content-type: application/json' \
  -d '{
    "requestId": "req_cv_001",
    "inputVersion": "cv-analyzer-v1",
    "language": "en",
    "inputMode": "UPLOAD",
    "compareSource": "JOB_SEARCH",
    "profile": {
      "profileText": "Backend developer building REST APIs with TypeScript and PostgreSQL",
      "normalizedSkills": ["typescript", "postgresql", "rest api"],
      "targetRoles": ["Backend Developer"],
      "detectedCvSectionNames": ["experience", "skills"]
    },
    "rankingPolicy": {
      "maxRecommendations": 5,
      "requireCandidateJobIds": true,
      "deduplicateByJobId": true,
      "backendOwnsHydration": true
    },
    "jobCandidates": [
      {
        "jobId": "job-1",
        "scoringInput": {
          "titleText": "Backend Developer",
          "requirementSummary": "Build REST APIs",
          "requiredSkills": ["typescript", "postgresql"],
          "roleFamily": "backend"
        }
      }
    ]
  }'
```

Response `data` is model-core only: bounded integer scores, evidence keys, candidate IDs from request, and no backend-owned hydrated fields.

### Smoke and contract tests

```bash
python -m unittest tests.model_api.test_phase_26_layout.Phase26LayoutTest.test_real_phase25_keras_artifact_matches_inference_smoke_fixture_predictions
python -m unittest tests.model_api.test_phase_26_layout.Phase26LayoutTest.test_phase25_handoff_fixtures_match_recorded_contract_validation_behavior
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_phase_29_model_api_hardening
```

Smoke coverage loads the real Phase 25 `.keras` artifact, runs rows from `inference_smoke_fixture.json`, checks prediction bounds, and compares positive/negative handoff validation with `model_api_handoff_validation.json`.

Final production smoke gate:

```bash
python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live
```

Run this only inside a Python `3.13.x` serving environment after `python -m pip install -r requirements.txt`. The gate fails if Phase 26 tests skip TensorFlow/Keras coverage, the real `.keras` custom-object loader does not run, or `/health`, `/model-info`, and `/inference/cv-analysis` are not hit with real runtime dependencies.

### Deployment notes

- Build image from repo root so relative artifact and reference paths resolve.
- Mount Phase 25 artifact directory read-only.
- Set env vars explicitly in staging/production.
- Warm up `/health` after process start before routing inference traffic.
- Keep OpenRouter disabled for core inference; wrapper GenAI remains backend-owned.
- Do not mutate artifacts at runtime.

## Internal multipart CV analysis contract

Production Backend integration uses one internal route:

```text
POST /internal/model/cv-analysis
content-type: multipart/form-data
```

Backend sends:

- `requestId`: required trace id.
- `language`: public/model enum `id | en`; Backend maps Prisma `ID | EN` at persistence boundary.
- `inputMode`: `UPLOAD | REFERENCE`; Backend resolves ownership before forwarding PDF bytes.
- `compareSource`: `BOOKMARK | JOB_SEARCH | DIRECT_JOB_DETAIL`.
- `jobRoles[]`: backend-selected target roles.
- `cvFile`: one PDF file part only; magic bytes, configured size limit, and configured page limit are enforced.
- `jobCandidates`: JSON array of backend-selected candidate jobs.
- `rankingPolicy`: JSON policy with `maxRecommendations <= 5`, `requireCandidateJobIds=true`, `deduplicateByJobId=true`, and `backendOwnsHydration=true`.

Model API returns `model-core-cv-analysis-v1` only:

- `parsedCv`: parser status, page count, text length, detected sections, and extraction evidence. Empty/scanned PDFs use deterministic low-confidence issues; no OCR or hallucinated CV text is generated.
- `jobFitAlignment`: bounded score plus matched/missing signals and evidence.
- `atsFriendliness`: bounded score plus parser/format issues and evidence.
- `overallImpression`: score/evidence for Backend wrapper copy.
- `candidateReranking.recommendations[]`: supplied `jobId`, bounded `matchScore`, `matchLevel`, skill evidence, max five items.
- `model` and timestamps.

Model-core output must exclude Backend/public fields: `title`, `companyName`, `reason`, `nextStep`, `topActionables`, `sectionReviews`, `generatedCv`, auth, persistence, DB objects, and hydrated job data.

Durable contract evidence:

- `artifacts/backend_model_api_contract/internal_contract_fixtures.json`
- `artifacts/backend_model_api_contract/openapi_prisma_owner_matrix.json`

## Request schema boundary

Model API requests are strict model-core inputs:

- `requestId`: required trace id.
- `language`: `id | en` from backend OpenAPI; DB `ID | EN` maps at backend boundary.
- CV/profile evidence: sanitized `cvText`/`profileText` or extracted signals such as `normalizedSkills`, `targetRoles`, `roleFamily`, `experienceYears`, `experienceBand`, `embeddingTextHash`, `detectedCvSectionNames`.
- `jobCandidates[]`: required backend candidate set, max `50`, unique `jobId`.
- `jobCandidates[].scoringInput`: model-owned evidence only, such as text summaries, required skills, role/experience signals, finite numeric signals.
- `jobCandidates[].backendMetadata`: optional backend-owned hydration context only. Top-level hydrated fields such as `title` or `companyName` are rejected.
- `rankingPolicy`: must require backend candidate IDs and deduplication. Public recommendation cap is `5`.

## Feature builder

Runtime feature construction uses Phase 25 contract only:

- feature order: `e5_cosine`, `skill_overlap`, `requirement_coverage`, `role_match`, `experience_match`, `experience_gap_years_clipped`
- E5 backend: `sentence-transformers` + `intfloat/e5-base-v2`
- prefixes: `query:` for profile/CV text and `passage:` for job text
- normalization: train-split `mean`/`std` from `tensorflow_feature_config.json`
- staging/production: TF-IDF, local-hash, and fallback embedding backends are rejected

## Error behavior

Errors use deterministic JSON shape:

```json
{
  "success": false,
  "message": "Invalid model-core request/response",
  "data": null,
  "error": {
    "errorCode": "contract_validation_error",
    "message": "$.language must be one of ['en', 'id']",
    "errors": ["$.language must be one of ['en', 'id']"]
  }
}
```

Stable error codes include:

- `contract_validation_error`: invalid language, empty candidate set, duplicate candidate IDs, missing text/signals, malformed payload, or response-contract violation.
- `unsupported_artifact_version`: artifact manifest schema or phase does not match Phase 25 runtime handoff.
- `artifact_error`: missing/stale hash, size mismatch, missing runtime artifact, or invalid artifact JSON.
- `feature_build_error`: invalid E5 backend, missing E5 text/signals, embedding failure, non-finite feature, or normalization failure.
- `model_not_ready`: model not loaded or startup loader failed.
- `model_load_error`: TensorFlow/Keras prediction output is malformed or prediction call fails.
- `inference_timeout`: configured inference timeout is exceeded.

Staging/production never falls back to rule-based scores when TensorFlow inference fails.

## Boundary

Model API returns model-core payloads only:

- `schemaVersion`: `model-core-cv-analysis-v1` or `model-core-candidate-reranking-v1`
- `language`: OpenAPI enum `id | en`
- scores: integer `0..100`
- recommendations: max `5`, unique, only backend-provided `jobId`

Backend owns final `cv-analysis-v2` response fields from OpenAPI: `id`, prose summaries, `topActionables`, `sectionReviews`, hydrated `jobRecommendations[].title/companyName/reason/nextStep`, `generatedCv`, auth, persistence, and DB writes.

## OpenRouter

OpenRouter config exists only for wrapper-owned GenAI prose. Core model inference must not call external GenAI by default.

Environment knobs:

- `MODEL_API_ENABLE_GENAI_WRAPPER=false`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=~openai/gpt-latest`
- `OPENROUTER_API_KEY` (required only when wrapper GenAI is enabled)

## Non-goals

- No backend DB access.
- No auth or persistence.
- No final job detail hydration.
- No retraining.
- No external GenAI call for core inference.
