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

- Python `3.13.11`
- TensorFlow `2.21.x`
- Keras `3.14.x`
- FastAPI
- Uvicorn
- NumPy `2.1.x`
- SentenceTransformers with artifact-declared E5 model (`intfloat/e5-base-v2` rollback or `intfloat/multilingual-e5-small`)

Install from repo root with the same Python version used by the TensorFlow notebook runtime:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv .venv
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not use `pyenv`, make sure `python -V` prints `3.13.11` before creating `.venv`. Do not use Python `3.14` for TensorFlow/Model API smoke checks.

## Required Artifact Paths

Defaults resolve from repository root. Set `MODEL_API_ARTIFACT_ROOT` to switch a known versioned package; explicit env paths can still pin every file for rollback.

| Env var                               | Default                                                                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `MODEL_API_ARTIFACT_ROOT`             | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh`                                                               |
| `MODEL_API_MODEL_PATH`                | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/export/selected_jobfit_tf_phase46_multilingual_e5_small.keras` |
| `MODEL_API_TENSORFLOW_FEATURE_CONFIG` | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/tensorflow_feature_config.json`                                |
| `MODEL_API_FEATURE_CONFIG`            | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/feature_config.json`                                           |
| `MODEL_API_SCORE_CALIBRATION`         | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/score_calibration.json`                                        |
| `MODEL_API_MODEL_CARD`                | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/model_card.json`                                               |
| `MODEL_API_ARTIFACT_MANIFEST`         | `artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/artifact_manifest.json`                                        |
| `MODEL_API_EXPECTED_EMBEDDING_MODEL`  | optional startup assertion, e.g. `intfloat/multilingual-e5-small`                                                                  |

For rollback to Phase 25 E5-base, set:

```bash
export MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery
export MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2
```

Startup verifies manifest SHA-256 and byte-size metadata before serving inference. It also verifies the embedding model, prefixes, and normalization policy agree across feature config, model card, manifest, env expectation, and runtime backend.

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
export MODEL_API_MAX_RECOMMENDATIONS=10
export MODEL_API_TIMEOUT_MS=30000
export MODEL_API_WARMUP_REQUIRED=true
export MODEL_API_WARMUP_ON_STARTUP=true
export MODEL_API_MAX_PDF_BYTES=5000000
export MODEL_API_MAX_PDF_PAGES=10
export MODEL_API_ENABLE_GENAI_WRAPPER=false
```

Use `MODEL_API_ENV=local` only for local experiments. Staging and production require `MODEL_API_SERVICE_TOKEN`, verified artifacts, TensorFlow model loading, E5 backend configuration, PDF parser availability, and no fallback embedding backend. TF-IDF, local-hash, and undeclared embedding model swaps are rejected in every environment.

## Run API

```bash
export MODEL_API_ENV=local
export MODEL_API_SERVICE_TOKEN=replace-with-local-service-token
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Liveness/readiness checks:

```bash
curl http://127.0.0.1:8000/live
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -H "authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:8000/model-info
```

`/live` is a lightweight process liveness endpoint. Runtime model loading starts in the background so `/live`, `/`, and `/health` can respond during cold start. `/ready.ready=true` means artifacts are verified, TensorFlow model is loaded, the artifact-declared E5 backend is configured, PDF parser is available, service token is configured when required, and warmup is complete when `MODEL_API_WARMUP_REQUIRED=true`. `/ready` and `/model-info` expose `embeddingModel`, artifact phase, model version, artifact hash, and readiness metadata for deployment verification. `/model-info` uses the same bearer token in staging/production. Loader failure keeps inference unavailable and returns deterministic `model_not_ready` or `model_load_error` payloads.

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
- `rankingPolicy`: JSON policy with `maxRecommendations <= 10`, `requireCandidateJobIds=true`, `deduplicateByJobId=true`, and `backendOwnsHydration=true`

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
      "maxRecommendations": 10,
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

Runtime feature construction uses the versioned artifact contract:

- feature order: `e5_cosine`, `skill_overlap`, `requirement_coverage`, `role_match`, `experience_match`, `experience_gap_years_clipped`
- E5 backend: `sentence-transformers` with the embedding model declared by the loaded artifact
- prefixes: `query:` for CV/profile text and `passage:` for job text
- normalization: train-split `mean`/`std` from `tensorflow_feature_config.json`
- all environments: TF-IDF, local-hash, fallback backends, and undeclared embedding swaps are rejected

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

Logs must include operational metadata only. Runtime warmup and staging smoke evidence should record cold-start latency and warm inference latency for `/internal/model/cv-analysis` with this breakdown:

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

See `docs/runbooks/hugging-face-docker-deployment.md` for Hugging Face Spaces and `docs/runbooks/vps-docker-deployment.md` for VPS Docker Compose deployment.

- Build images from repo root so relative artifact paths resolve.
- Include Phase 46 artifact directory in the image; keep Phase 25 artifact directory available only for rollback.
- Set env vars explicitly in staging/production.
- Keep Model API private behind Backend/internal routing.
- Keep `SENTENCE_TRANSFORMERS_HOME` on the image-baked cache path; do not mount over it or the build-time E5 predownload is hidden.
- Use `/live` for uptime checks that must not wait for TensorFlow/E5 readiness; use `/ready` as the traffic gate.
- Run `python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url "$MODEL_API_URL" --token "$MODEL_API_SERVICE_TOKEN" --latency-budget-ms 30000` before routing inference traffic.
- Keep `MODEL_API_TIMEOUT_MS` greater than measured cold-start latency, or make warmup mandatory before traffic.
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
