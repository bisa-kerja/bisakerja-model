# Service Boundaries

This document aligns the model workspace with Backend API responsibilities. Backend source is not part of this repository. Backend API lives at <https://github.com/bisa-kerja/bisakerja-api>.

## Platform Components

| Component          | Responsibility                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Frontend UI        | Presents job discovery, CV analysis, recommendations, and application tracking.                                  |
| Backend API        | External public REST boundary, auth, persistence, product workflows, wrapper copy, hydration, and error mapping. |
| Model API          | Internal inference service for model-core CV analysis and candidate scoring.                                     |
| Training Workspace | Notebook-first model training, evaluation, calibration, and artifact export.                                     |
| PostgreSQL         | Backend-owned durable source of truth in the Backend API system.                                                 |

Frontend UI must never call Model API directly. Backend API is the only public API owner.

## Backend API Owns

- authentication, authorization, and ownership checks
- user profile, preference, bookmark, application, and CV-file workflows
- database access through Prisma/PostgreSQL
- public API response shape and error mapping
- job candidate retrieval and hydration
- GenAI wrapper copy for product-facing summaries and recommendations
- persistence of CV analysis and recommendation results

Backend repo: <https://github.com/bisa-kerja/bisakerja-api>

## Model API Owns

- loading verified exported TensorFlow/Keras artifacts
- building approved feature vectors
- request-scoped PDF parsing signals
- sanitized wrapper evidence helper contracts for Backend-owned Analyzer prose
- strict model-core request and response validation
- deterministic inference errors
- operational metadata logging without secrets or raw CV text

Model API owns no DB credentials and does not persist user data.

## Training Owns

- notebook-first training workflow
- weak-label and human-label governance
- feature generation and calibration evidence
- TensorBoard release logs
- model card, manifest, and handoff fixtures
- reproducibility and production-readiness reports

Training output becomes serving input only after exported artifacts are recorded with hashes and release evidence.

## Model-Core Output

Model API may return:

- `schemaVersion`
- `language`
- `parsedCv`
- `jobFitAlignment`
- `atsFriendliness`
- `overallImpression`
- `candidateReranking.recommendations[]` with Backend-provided `jobId` values
- model metadata and timestamps

Model API helper contract `ai-cv-analyzer-wrapper-evidence-v1` defines allowlisted Backend wrapper input fields: parsed-CV status, detected section evidence, requirement coverage, ATS issue evidence, role evidence, quantified-impact flags, and candidate job context. This helper context must not include raw CV text, contact data, file bytes, prompts, provider payloads, tokens, storage keys, DB data, or hydrated job fields. Experience years, education, certification, location/work-type, and generic requirements remain separate from skills and must not be emitted as `missingSkills`.

Model API default response shape remains unchanged and must not return Backend/public fields such as `title`, `companyName`, `reason`, `nextStep`, `topActionables`, `sectionReviews`, `generatedCv`, auth state, DB objects, or hydrated job data.

## Integration Flow

```text
User uploads or selects CV
  -> Backend API validates auth, ownership, file limits, and candidate jobs
  -> Backend API sends model-core request to internal Model API
  -> Model API parses CV, builds features, classifies requirements, runs TensorFlow inference, validates output
  -> Backend API uses model-core scores plus allowlisted wrapper evidence to build public CvAnalysis prose
  -> Backend API persists result and hydrates job recommendations
```

## Failure Handling

- invalid PDF -> deterministic validation error
- parse failure -> deterministic low-confidence parse result, no fabricated CV text
- empty candidates -> Backend no-recommendation policy before Model API call
- Model API timeout -> deterministic unavailable/timeout mapping
- TensorFlow load failure -> readiness failure and inference unavailable
- E5 failure -> readiness or inference failure, no rule-based fallback in staging/production
- GenAI wrapper failure -> Backend fallback copy without changing model scores or ordering
