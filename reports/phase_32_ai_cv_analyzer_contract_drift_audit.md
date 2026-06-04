# AI CV Analyzer Contract Drift Audit and Canonical Schema Freeze

Final decision: `implemented`

## Canonical decision

- `internalResponseShape`: POST /internal/model/cv-analysis returns raw model-core JSON, not success/data envelope
- `modelCoreOwner`: Model API owns parsedCv, jobFitAlignment evidence, atsFriendliness evidence, overallImpression evidence, candidate reranking scores, model name/version, createdAt
- `backendOwner`: Backend owns public cv-analysis-v2 envelope, prose wrapper/fallback, hydration, generatedCv unavailable object, persistence, auth/user/CV ownership
- `languagePolicy`: Request language remains explicit id|en; product-facing staging default is English; fallback/wrapper must not silently switch to Indonesian
- `securityPolicy`: Model API remains internal-only; no DB credentials, tokens, raw CV text, storage keys, artifact paths, or public hydrated job fields in public response/logs
- `aiCvGenerateScope`: AI CV Generate compatibility is separate audit scope and must not block AI CV Analyzer closure
- `reviewGate`: Canonical schema reviewed and implemented for Model API internal route

## Checks

- [x] `public_openapi_contract_extracted`
- [x] `backend_client_contract_extracted`
- [x] `model_api_actual_contract_extracted`
- [x] `drift_matrix_complete`
- [x] `canonical_internal_shape_chosen`
- [x] `language_policy_frozen`
- [x] `ai_cv_generate_marked_separate`
- [x] `ownership_boundary_clear`
- [x] `internal_route_returns_raw_model_core`
- [x] `backend_request_shape_supported`
- [x] `internal_response_metadata_minimized`

## Public OpenAPI contract

- Source: `references/docs/generated/openapi.json#/paths/~1api~1v1~1ai~1cv-analyzer/post`
- Route: `POST /api/v1/ai/cv-analyzer`
- Request required: `jobRoles, language, inputMode`
- Success envelope required: ``
- CvAnalysis required: `jobRoles, language, analysisResult`
- AnalysisResult required: `id, schemaVersion, jobFitAlignment, atsFriendliness, overallImpression, topActionables, sectionReviews, jobRecommendations, generatedCv, model, analyzedAt`
- Error statuses: `401, 404, 413, 422, 502, 503`

## Drift matrix

| ID | Current drift | Canonical shape |
| --- | --- | --- |
| `response-envelope` | Implemented: internal route returns raw model-core JSON | Internal route returns raw model-core JSON; no data wrapper |
| `timestamp-createdAt-vs-analyzedAt` | Implemented: internal route emits createdAt | Model-core uses createdAt; Backend maps to public analyzedAt |
| `parsedCv-status-detectedSections-vs-sectionNames` | Implemented: parsedCv uses status, textLength, detectedSections, extractionEvidence | Model-core parsedCv = status/pageCount/textLength/detectedSections/extractionEvidence |
| `parseQuality-enum` | Implemented: parser qualities are mapped before internal response | Map parser qualities to high\|medium\|low\|failed before response |
| `requirements-object-vs-string` | Implemented: accepts strings or requirement objects and uses value for scoring | Model API accepts Backend requirement objects and derives scoring text from value |
| `numericSignals-vs-numericFeatures` | Implemented: accepts approved numericSignals or numericFeatures | Accept numericSignals as approved numeric feature source or remove from Backend fixture explicitly |
| `backendMetadata-locationDisplay` | Implemented: locationDisplay is accepted as safe backend metadata | Allow title/companyName/locationDisplay/sourceUpdatedAt only for trace/hydration hints |
| `jobRoles-repeated-form-fields` | Implemented: all repeated jobRoles values are read | Model API reads all jobRoles form values; min 1 max 10 |
| `strict-extra-fields` | Implemented: internal route removes debug summary/confidence/observability fields | Model-core response contains only Backend schema fields |
| `model-artifact-metadata-exposure` | Implemented: internal response returns model name/version only | Model-core HTTP response exposes only name/version unless explicitly allowed |
| `wrapper-output-ownership` | Implemented: internal route returns evidence arrays and scores only | Model API returns evidence/signals only; Backend wrapper owns prose |
| `error-envelope-mapping` | Documented: Model API internal errors stay internal for Backend mapping | Backend maps Model API 4xx/5xx/timeout/invalid schema to public ErrorEnvelope |
| `language-default-policy` | Documented: request language remains explicit; Backend wrapper owns prose language | English default for wrapper/fallback; no silent Indonesian switch |
| `security-privacy-fields` | Implemented: request parser rejects unsafe backend metadata and internal response omits artifacts/observability | Raw CV/storage keys stay internal request only; public response excludes them; logs use allowlist |

## Sources

- `openapi`: `references\docs\generated\openapi.json`
- `backend_fixtures`: `artifacts\backend_model_api_contract\internal_contract_fixtures.json`
- `owner_matrix`: `artifacts\backend_model_api_contract\openapi_prisma_owner_matrix.json`
- `model_api_app`: `model_api\app.py`
- `model_api_schemas`: `model_api\schemas.py`
- `model_api_validators`: `model_api\validators.py`
