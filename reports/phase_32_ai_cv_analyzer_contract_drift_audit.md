# AI CV Analyzer Contract Drift Audit and Canonical Schema Freeze

Final decision: `review_required`

## Canonical decision

- `internalResponseShape`: POST /internal/model/cv-analysis returns raw model-core JSON, not success/data envelope
- `modelCoreOwner`: Model API owns parsedCv, jobFitAlignment evidence, atsFriendliness evidence, overallImpression evidence, candidate reranking scores, model name/version, createdAt
- `backendOwner`: Backend owns public cv-analysis-v2 envelope, prose wrapper/fallback, hydration, generatedCv unavailable object, persistence, auth/user/CV ownership
- `languagePolicy`: Request language remains explicit id|en; product-facing staging default is English; fallback/wrapper must not silently switch to Indonesian
- `securityPolicy`: Model API remains internal-only; no DB credentials, tokens, raw CV text, storage keys, artifact paths, or public hydrated job fields in public response/logs
- `aiCvGenerateScope`: AI CV Generate compatibility is separate audit scope and must not block AI CV Analyzer closure
- `reviewGate`: No request/response implementation phase should start until this canonical schema and drift matrix are reviewed

## Checks

- [x] `public_openapi_contract_extracted`
- [x] `backend_client_contract_extracted`
- [x] `model_api_actual_contract_extracted`
- [x] `drift_matrix_complete`
- [x] `canonical_internal_shape_chosen`
- [x] `language_policy_frozen`
- [x] `ai_cv_generate_marked_separate`
- [x] `ownership_boundary_clear`

## Public OpenAPI contract

- Source: `references/docs/generated/openapi.json#/paths/~1api~1v1~1ai~1cv-analyzer/post`
- Route: `POST /api/v1/ai/cv-analyzer`
- Request required: `jobRoles, language, inputMode`
- Success envelope required: `success, message, data, meta`
- CvAnalysis required: `jobRoles, language, analysisResult`
- AnalysisResult required: `id, schemaVersion, jobFitAlignment, atsFriendliness, overallImpression, topActionables, sectionReviews, jobRecommendations, generatedCv, model, analyzedAt`
- Error statuses: `401, 404, 413, 422, 502, 503`

## Drift matrix

| ID | Current drift | Canonical shape |
| --- | --- | --- |
| `response-envelope` | Model API returns success/message/data/error envelope | Internal route returns raw model-core JSON; no data wrapper |
| `timestamp-createdAt-vs-analyzedAt` | Model API emits analyzedAt | Model-core uses createdAt; Backend maps to public analyzedAt |
| `parsedCv-status-detectedSections-vs-sectionNames` | Model API emits sectionNames and no status | Model-core parsedCv = status/pageCount/textLength/detectedSections/extractionEvidence |
| `parseQuality-enum` | Parser emits text_ok or parser-specific quality | Map parser qualities to high\|medium\|low\|failed before response |
| `requirements-object-vs-string` | Model API accepts string[] | Model API accepts Backend requirement objects and derives scoring text from value |
| `numericSignals-vs-numericFeatures` | Model API scoringInput.numericFeatures | Accept numericSignals as approved numeric feature source or remove from Backend fixture explicitly |
| `backendMetadata-locationDisplay` | Model API allows nested location, not locationDisplay | Allow title/companyName/locationDisplay/sourceUpdatedAt only for trace/hydration hints |
| `jobRoles-repeated-form-fields` | Model API JSON-parses form.get('jobRoles') | Model API reads all jobRoles form values; min 1 max 10 |
| `strict-extra-fields` | Model API emits summarySignals/confidenceNotes/summary/evidenceKeys/fallback/observability/nested metadata | Model-core response contains only Backend schema fields |
| `model-artifact-metadata-exposure` | Model API may include artifact path/hash | Model-core HTTP response exposes only name/version unless explicitly allowed |
| `wrapper-output-ownership` | Model API core has summary-style fields | Model API returns evidence/signals only; Backend wrapper owns prose |
| `error-envelope-mapping` | Model API internal envelope differs | Backend maps Model API 4xx/5xx/timeout/invalid schema to public ErrorEnvelope |
| `language-default-policy` | OpenAPI example uses id and fallback copy may mix language | English default for wrapper/fallback; no silent Indonesian switch |
| `security-privacy-fields` | Request carries cv.storageKey to client serializer and response may expose observability/artifact metadata | Raw CV/storage keys stay internal request only; public response excludes them; logs use allowlist |

## Sources

- `openapi`: `references/docs/generated/openapi.json`
- `backend_schema`: `references/src/shared/integrations/model-api.schema.ts`
- `backend_client`: `references/src/shared/integrations/model-api.client.ts`
- `backend_service`: `references/src/modules/ai-cv-analyzer/ai-cv-analyzer.service.ts`
- `model_api_app`: `model_api/app.py`
- `model_api_schemas`: `model_api/schemas.py`
- `model_api_validators`: `model_api/validators.py`
