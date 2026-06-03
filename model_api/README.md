# Model API Workspace

FastAPI serving package for Bisakerja model-core inference.

Model API consumes exported training artifacts, validates strict model-core contracts, parses CV evidence, builds approved features, runs TensorFlow/Keras inference, and returns deterministic JSON. It is internal-only and is called by Backend API, not by Frontend UI.

Backend API is a separate repository: <https://github.com/bisa-kerja/bisakerja-api>.

## Boundary

Model API owns:

- artifact verification and model loading
- request-scoped PDF parsing signals
- E5 feature construction and normalization
- TensorFlow inference
- model-core schema validation
- deterministic errors and safe observability metadata

Model API does not own:

- Backend auth or user ownership
- PostgreSQL/Prisma access
- persisted CV analysis records
- public `CvAnalysis` response formatting
- hydrated job details
- final prose fields such as `topActionables`, `sectionReviews`, `reason`, `nextStep`, or `generatedCv`
- retraining

Frontend UI must never call Model API directly. Backend API is the public REST boundary.

## Canonical External Source

Backend contracts and public API ownership come from:

```text
https://github.com/bisa-kerja/bisakerja-api
```

When contract validation is needed, use OpenAPI/Prisma snapshots or handoff fixtures exported from that Backend repository. Backend source is not part of this model repository.

## Layout

```text
model_api/
|-- README.md
|-- .env.example
|-- __init__.py
|-- app.py             # FastAPI app factory and route wiring
|-- config.py          # Environment and artifact path resolution
|-- contracts.py       # Backend contract extraction helpers from exported snapshots
|-- schemas.py         # Strict model-core request/response schemas
|-- artifacts.py       # Manifest loading and hash/size verification
|-- custom_objects.py  # Keras custom-object registration
|-- features.py        # Feature vector construction and normalization
|-- pdf_parser.py      # Deterministic PDF parsing and ATS evidence
|-- inference.py       # Loaded model service facade
|-- validators.py      # Response contract validators
|-- errors.py          # Stable error payloads
`-- observability.py   # Safe operational metadata allowlist
```

## Runtime

Target runtime:

- Python `3.13.x`
- TensorFlow `2.21.x`
- Keras `3.14.x`
- FastAPI
- Uvicorn
- NumPy
- SentenceTransformers with `intfloat/e5-base-v2`

Install from repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required Artifact Paths

Defaults resolve from repository root:

| Env var                               | Default                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `MODEL_API_MODEL_PATH`                | `artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras` |
| `MODEL_API_TENSORFLOW_FEATURE_CONFIG` | `artifacts/phase_25_tensorflow_training_delivery/tensorflow_feature_config.json`          |
| `MODEL_API_FEATURE_CONFIG`            | `artifacts/phase_25_tensorflow_training_delivery/feature_config.json`                     |
| `MODEL_API_SCORE_CALIBRATION`         | `artifacts/phase_25_tensorflow_training_delivery/score_calibration.json`                  |
| `MODEL_API_MODEL_CARD`                | `artifacts/phase_25_tensorflow_training_delivery/model_card.json`                         |
| `MODEL_API_ARTIFACT_MANIFEST`         | `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`                  |

Startup verifies manifest SHA-256 and byte-size metadata before serving inference.

## Backend Contract Snapshots

Model API may validate against Backend OpenAPI/Prisma snapshots exported from <https://github.com/bisa-kerja/bisakerja-api>.

Optional env vars:

| Env var                        | Purpose                                                                   |
| ------------------------------ | ------------------------------------------------------------------------- |
| `MODEL_API_OPENAPI_PATH`       | Path to generated Backend OpenAPI JSON snapshot.                          |
| `MODEL_API_PRISMA_SCHEMA_PATH` | Path to Backend Prisma schema snapshot when DB enum validation is needed. |

These snapshots are release inputs, not Backend source ownership. Keep Backend repo outside this model repository unless a specific generated fixture is intentionally copied into `artifacts/`.

## Environment

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

Use `MODEL_API_ENV=local` only for local experiments. Staging and production require `MODEL_API_SERVICE_TOKEN`, verified artifacts, TensorFlow model loading, E5 backend configuration, PDF parser availability, and no fallback embedding backend.

## Run API

```bash
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

`/health.ready=true` means artifacts are verified and runtime dependencies are available. Loader failure keeps inference unavailable and returns deterministic `model_not_ready` or `model_load_error` payloads.

## Internal Multipart CV Analysis Contract

Backend integration uses one internal route:

```text
POST /internal/model/cv-analysis
content-type: multipart/form-data
```

Backend sends:

- `requestId`: required trace id
- `language`: public/model enum `id | en`; Backend maps persistence enums at its own boundary
- `inputMode`: `UPLOAD | REFERENCE`; Backend resolves ownership before forwarding PDF bytes
- `compareSource`: `BOOKMARK | JOB_SEARCH | DIRECT_JOB_DETAIL`
- `jobRoles[]`: Backend-selected target roles
- `cvFile`: one PDF file part only; magic bytes, size limit, and page limit are enforced
- `jobCandidates`: JSON array of Backend-selected candidate jobs
- `rankingPolicy`: JSON policy with `maxRecommendations <= 5`, `requireCandidateJobIds=true`, `deduplicateByJobId=true`, and `backendOwnsHydration=true`

Model API returns model-core only:

- `parsedCv`
- `jobFitAlignment`
- `atsFriendliness`
- `overallImpression`
- `candidateReranking.recommendations[]`
- model metadata and timestamps

## Example JSON Inference Request

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

Response `data` is model-core only: bounded integer scores, evidence keys, candidate IDs from request, and no Backend-owned hydrated fields.

## Feature Builder

Runtime feature construction uses the Phase 25 contract only:

- feature order: `e5_cosine`, `skill_overlap`, `requirement_coverage`, `role_match`, `experience_match`, `experience_gap_years_clipped`
- E5 backend: `sentence-transformers` + `intfloat/e5-base-v2`
- prefixes: `query:` for CV/profile text and `passage:` for job text
- normalization: train-split `mean`/`std` from `tensorflow_feature_config.json`
- staging/production: TF-IDF, local-hash, and fallback embedding backends are rejected

## Error Behavior

Errors use deterministic JSON:

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

- `contract_validation_error`
- `unsupported_artifact_version`
- `artifact_error`
- `feature_build_error`
- `model_not_ready`
- `model_load_error`
- `inference_timeout`

Staging and production never fall back to rule-based scores when TensorFlow inference fails.

## Safe Observability

Logs must include operational metadata only:

- `requestId`
- `modelVersion`
- `artifactHash`
- `candidateCount`
- `parseQuality`
- `parseLatencyMs`
- `embeddingLatencyMs`
- `tensorflowLatencyMs`
- `wrapperLatencyMs`
- `totalLatencyMs`
- `errorCode`
- `fallbackReason`

Logs must exclude raw CV text, service-token values, DB URLs, auth headers, uploaded file bytes, and unrelated PII.

Security/privacy review items: service-token rotation guidance, raw CV log exclusion, upload cleanup, retention, path traversal defense, and non-public Model API routing.

## Smoke and Contract Tests

```bash
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_phase_29_model_api_hardening
python -m unittest tests.test_phase_31_release_gate
```

Final production smoke gate:

```bash
python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live
```

Run this only inside a Python `3.13.x` serving environment after `python -m pip install -r requirements.txt`. The gate fails if TensorFlow/Keras coverage is skipped, the real `.keras` custom-object loader does not run, or `/health`, `/model-info`, and `/inference/cv-analysis` are not hit with real runtime dependencies.

## Deployment Notes

- Build images from repo root so relative artifact paths resolve.
- Mount Phase 25 artifact directory read-only.
- Set env vars explicitly in staging/production.
- Keep Model API private behind Backend/internal routing.
- Warm up `/health` after process start before routing inference traffic.
- Keep OpenRouter disabled for core inference; wrapper GenAI remains Backend-owned.
- Do not mutate artifacts at runtime.

## OpenRouter

OpenRouter config exists only for wrapper-owned GenAI prose experiments. Core model inference must not call external GenAI by default.

Environment knobs:

```bash
MODEL_API_ENABLE_GENAI_WRAPPER=false
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=~openai/gpt-latest
OPENROUTER_API_KEY=replace-with-key-only-if-wrapper-enabled
```

## Non-Goals

- No Backend DB access.
- No auth or persistence ownership.
- No final job detail hydration.
- No retraining.
- No external GenAI call for core inference.
