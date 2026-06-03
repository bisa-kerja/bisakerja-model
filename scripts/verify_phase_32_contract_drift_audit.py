from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPENAPI_PATH = ROOT / "references/docs/generated/openapi.json"
MODEL_SCHEMA_PATH = ROOT / "references/src/shared/integrations/model-api.schema.ts"
MODEL_CLIENT_PATH = ROOT / "references/src/shared/integrations/model-api.client.ts"
AI_CV_SERVICE_PATH = ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.service.ts"
MODEL_APP_PATH = ROOT / "model_api/app.py"
MODEL_API_SCHEMA_PATH = ROOT / "model_api/schemas.py"
MODEL_VALIDATORS_PATH = ROOT / "model_api/validators.py"
REPORT_JSON_PATH = ROOT / "reports/phase_32_ai_cv_analyzer_contract_drift_audit.json"
REPORT_MD_PATH = ROOT / "reports/phase_32_ai_cv_analyzer_contract_drift_audit.md"

DRIFT_IDS = (
    "response-envelope",
    "timestamp-createdAt-vs-analyzedAt",
    "parsedCv-status-detectedSections-vs-sectionNames",
    "parseQuality-enum",
    "requirements-object-vs-string",
    "numericSignals-vs-numericFeatures",
    "backendMetadata-locationDisplay",
    "jobRoles-repeated-form-fields",
    "strict-extra-fields",
    "model-artifact-metadata-exposure",
    "wrapper-output-ownership",
    "error-envelope-mapping",
    "language-default-policy",
    "security-privacy-fields",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def schema_at(openapi: dict[str, Any], ref: str) -> dict[str, Any]:
    return openapi["components"]["schemas"][schema_ref_name(ref)]


def field_summary(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "required": schema.get("required", []),
        "additionalProperties": schema.get("additionalProperties"),
        "properties": sorted(schema.get("properties", {}).keys()),
    }


def build_public_contract(openapi: dict[str, Any]) -> dict[str, Any]:
    operation = openapi["paths"]["/api/v1/ai/cv-analyzer"]["post"]
    request_schema = schema_at(
        openapi,
        operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"],
    )
    success_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    cv_schema = openapi["components"]["schemas"]["CvAnalysis"]
    analysis_result = cv_schema["properties"]["analysisResult"]

    return {
        "source": "references/docs/generated/openapi.json#/paths/~1api~1v1~1ai~1cv-analyzer/post",
        "method": "POST",
        "path": "/api/v1/ai/cv-analyzer",
        "requestContentType": "multipart/form-data",
        "request": {
            "required": request_schema["required"],
            "fields": {
                name: {
                    key: value[key]
                    for key in ("type", "format", "enum", "minItems", "maxItems", "default", "oneOf")
                    if key in value
                }
                for name, value in request_schema["properties"].items()
            },
        },
        "successEnvelope": field_summary(success_schema),
        "publicDataSchema": field_summary(cv_schema),
        "analysisResult": {
            "schemaVersion": analysis_result["properties"]["schemaVersion"]["const"],
            "required": analysis_result["required"],
            "scoreRange": "integer 0-100 for jobFitAlignment.score, atsFriendliness.score, jobRecommendations[].matchScore",
            "topActionables": "1-3 strings",
            "sectionReviews": "array of strict objects; actionPoints min 1",
            "jobRecommendations": "max 5; strict objects; jobId/companyName nullable; matchScore 0-100",
            "generatedCv": "strict { available, note }",
            "timestamp": "analyzedAt date-time",
            "model": "strict { name, version }",
        },
        "errorEnvelope": {
            "schema": "ErrorEnvelope",
            "statuses": sorted(k for k in operation["responses"] if k != "200"),
            "required": openapi["components"]["schemas"]["ErrorEnvelope"]["required"],
        },
    }


def build_backend_contract() -> dict[str, Any]:
    return {
        "sources": [
            "references/src/shared/integrations/model-api.schema.ts",
            "references/src/shared/integrations/model-api.client.ts",
            "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.service.ts",
        ],
        "modelApiEndpoint": "POST /internal/model/cv-analysis",
        "multipartRequest": {
            "fields": ["requestId", "language", "inputMode", "compareSource", "jobRoles[]", "cv", "jobCandidates", "rankingPolicy", "cvFile"],
            "cvFile": "Blob/PDF from Backend storage; field name cvFile",
            "jobRoles": "Backend appends repeated form fields, not JSON array string",
            "jobCandidates": "JSON string; max 50; scoringInput.requirements is object array { type, value, priority }[]",
            "numericSignals": "optional numeric record under scoringInput",
            "backendMetadata": "strict { title, companyName, locationDisplay, sourceUpdatedAt }",
        },
        "expectedModelCoreResponse": {
            "shape": "raw strict cvAnalyzerModelResponseSchema JSON, not { success, message, data, error } envelope",
            "schemaVersion": "model-core-cv-analysis-v1",
            "requiredRoot": ["schemaVersion", "parsedCv", "jobFitAlignment", "atsFriendliness", "overallImpression", "candidateReranking", "model", "createdAt"],
            "forbiddenPublicFields": ["topActionables", "sectionReviews", "generatedCv", "jobRecommendations[].title", "jobRecommendations[].companyName", "reason", "nextStep"],
        },
        "backendResponsibilities": [
            "auth, CV ownership, storage, and candidate DB retrieval",
            "hydrate final job recommendations from Backend DB only",
            "map model-core evidence into public CvAnalysis.analysisResult",
            "own public envelope { success, message, data, meta }",
            "own persistence snapshot when persistResult=true",
            "map Model API non-2xx/invalid JSON/Zod errors into Backend errors",
        ],
    }


def build_model_api_actual_contract() -> dict[str, Any]:
    return {
        "sources": ["model_api/app.py", "model_api/schemas.py", "model_api/validators.py"],
        "route": "POST /internal/model/cv-analysis",
        "contentType": "multipart/form-data only",
        "auth": "Bearer service token unless local/test unauthenticated bypass allowed",
        "multipartBehavior": {
            "cvFile": "requires exactly one cvFile; content-type application/pdf or application/octet-stream; PDF magic bytes required",
            "jobCandidates": "form field parsed as JSON array",
            "rankingPolicy": "form field parsed as JSON object/null",
            "jobRoles": "form.get('jobRoles') then JSON parse; repeated fields from Backend are not accepted as array",
        },
        "parsedPayload": {
            "inputVersion": "cv-analyzer-v1 forced internally",
            "profile": "built from parsed PDF text, targetRoles, normalized skills, detected section names",
            "candidateRequirements": "currently array of strings only",
            "numericFeatures": "currently accepted; numericSignals unknown",
            "backendMetadata": "title/companyName/location/workType/experienceLevel/postedAt/sourceUpdatedAt/source; no locationDisplay",
        },
        "responseBehavior": {
            "envelope": "returns { success: true, message, data, error: null }",
            "timestamp": "uses analyzedAt in model-core payload",
            "parsedCv": "injects { textLength, pageCount, parseQuality, sectionNames }",
            "atsFriendliness": "returns score, detectedIssues, fallback, evidence object; parseQuality nested in evidence/observability",
            "overallImpression": "returns score, summary, evidenceKeys, confidenceNotes",
            "candidateReranking": "nested full model-core reranking with requestId/schemaVersion/candidateSetId/language/recommendations/model/rankedAt",
            "model": "includes artifact metadata via dataclass identity",
            "observability": "included in response data",
        },
        "validation": {
            "scoreRange": "integer 0-100",
            "recommendations": "max requested; jobId must belong to request candidate set and be unique",
            "forbiddenFields": "backend/public owner keys rejected anywhere in model-core payload",
        },
    }


def build_drift_matrix() -> list[dict[str, str]]:
    return [
        {"id": "response-envelope", "sourceOfTruth": "Backend expects raw cvAnalyzerModelResponseSchema", "current": "Model API returns success/message/data/error envelope", "impact": "Backend Zod parses envelope root and rejects", "canonical": "Internal route returns raw model-core JSON; no data wrapper"},
        {"id": "timestamp-createdAt-vs-analyzedAt", "sourceOfTruth": "Backend model response requires createdAt; public OpenAPI uses analyzedAt after mapping", "current": "Model API emits analyzedAt", "impact": "Strict response validation fails and timestamp ownership drifts", "canonical": "Model-core uses createdAt; Backend maps to public analyzedAt"},
        {"id": "parsedCv-status-detectedSections-vs-sectionNames", "sourceOfTruth": "Backend requires parsedCv.status and detectedSections", "current": "Model API emits sectionNames and no status", "impact": "Parsed CV evidence rejected/misread", "canonical": "Model-core parsedCv = status/pageCount/textLength/detectedSections/extractionEvidence"},
        {"id": "parseQuality-enum", "sourceOfTruth": "Backend allows high|medium|low|failed", "current": "Parser emits text_ok or parser-specific quality", "impact": "Enum mismatch", "canonical": "Map parser qualities to high|medium|low|failed before response"},
        {"id": "requirements-object-vs-string", "sourceOfTruth": "Backend sends { type, value, priority }[]", "current": "Model API accepts string[]", "impact": "Request validation fails", "canonical": "Model API accepts Backend requirement objects and derives scoring text from value"},
        {"id": "numericSignals-vs-numericFeatures", "sourceOfTruth": "Backend scoringInput.numericSignals", "current": "Model API scoringInput.numericFeatures", "impact": "Strict request rejects numericSignals", "canonical": "Accept numericSignals as approved numeric feature source or remove from Backend fixture explicitly"},
        {"id": "backendMetadata-locationDisplay", "sourceOfTruth": "Backend sends locationDisplay", "current": "Model API allows nested location, not locationDisplay", "impact": "Strict request rejects safe hydration hint", "canonical": "Allow title/companyName/locationDisplay/sourceUpdatedAt only for trace/hydration hints"},
        {"id": "jobRoles-repeated-form-fields", "sourceOfTruth": "Backend FormData.append('jobRoles', role)", "current": "Model API JSON-parses form.get('jobRoles')", "impact": "Repeated multipart roles fail", "canonical": "Model API reads all jobRoles form values; min 1 max 10"},
        {"id": "strict-extra-fields", "sourceOfTruth": "Backend strict Zod response", "current": "Model API emits summarySignals/confidenceNotes/summary/evidenceKeys/fallback/observability/nested metadata", "impact": "Strict response fails", "canonical": "Model-core response contains only Backend schema fields"},
        {"id": "model-artifact-metadata-exposure", "sourceOfTruth": "Backend expects model { name, version }", "current": "Model API may include artifact path/hash", "impact": "Strict response failure and unnecessary internal metadata exposure", "canonical": "Model-core HTTP response exposes only name/version unless explicitly allowed"},
        {"id": "wrapper-output-ownership", "sourceOfTruth": "Backend owns prose/topActionables/sectionReviews/hydrated recommendations", "current": "Model API core has summary-style fields", "impact": "Ownership boundary blurred", "canonical": "Model API returns evidence/signals only; Backend wrapper owns prose"},
        {"id": "error-envelope-mapping", "sourceOfTruth": "Frontend sees Backend ErrorEnvelope; Model API internal errors mapped by client", "current": "Model API internal envelope differs", "impact": "Backend must not leak Model API internal error shape", "canonical": "Backend maps Model API 4xx/5xx/timeout/invalid schema to public ErrorEnvelope"},
        {"id": "language-default-policy", "sourceOfTruth": "Staging product prose default English; request language explicit id|en", "current": "OpenAPI example uses id and fallback copy may mix language", "impact": "Inconsistent product-facing prose", "canonical": "English default for wrapper/fallback; no silent Indonesian switch"},
        {"id": "security-privacy-fields", "sourceOfTruth": "Model API internal-only; no raw CV logs/tokens/storage keys/DB creds in public response", "current": "Request carries cv.storageKey to client serializer and response may expose observability/artifact metadata", "impact": "PII/internal metadata leak risk", "canonical": "Raw CV/storage keys stay internal request only; public response excludes them; logs use allowlist"},
    ]


def build_report() -> dict[str, Any]:
    openapi = load_json(OPENAPI_PATH)
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (MODEL_SCHEMA_PATH, MODEL_CLIENT_PATH, AI_CV_SERVICE_PATH, MODEL_APP_PATH, MODEL_API_SCHEMA_PATH, MODEL_VALIDATORS_PATH)
    )
    public_contract = build_public_contract(openapi)
    drift_matrix = build_drift_matrix()
    canonical = {
        "internalResponseShape": "POST /internal/model/cv-analysis returns raw model-core JSON, not success/data envelope",
        "modelCoreOwner": "Model API owns parsedCv, jobFitAlignment evidence, atsFriendliness evidence, overallImpression evidence, candidate reranking scores, model name/version, createdAt",
        "backendOwner": "Backend owns public cv-analysis-v2 envelope, prose wrapper/fallback, hydration, generatedCv unavailable object, persistence, auth/user/CV ownership",
        "languagePolicy": "Request language remains explicit id|en; product-facing staging default is English; fallback/wrapper must not silently switch to Indonesian",
        "securityPolicy": "Model API remains internal-only; no DB credentials, tokens, raw CV text, storage keys, artifact paths, or public hydrated job fields in public response/logs",
        "aiCvGenerateScope": "AI CV Generate compatibility is separate audit scope and must not block AI CV Analyzer closure",
        "reviewGate": "No request/response implementation phase should start until this canonical schema and drift matrix are reviewed",
    }
    checks = {
        "public_openapi_contract_extracted": public_contract["analysisResult"]["schemaVersion"] == "cv-analysis-v2",
        "backend_client_contract_extracted": all(token in source_text for token in ["requestMultipartModelApi", "cvAnalyzerModelResponseSchema", "buildPublicCvAnalysisResponse"]),
        "model_api_actual_contract_extracted": all(token in source_text for token in ["/internal/model/cv-analysis", "_build_cv_payload_from_multipart", "validate_model_core_payload"]),
        "drift_matrix_complete": {item["id"] for item in drift_matrix} == set(DRIFT_IDS),
        "canonical_internal_shape_chosen": canonical["internalResponseShape"].startswith("POST /internal/model/cv-analysis returns raw"),
        "language_policy_frozen": "English" in canonical["languagePolicy"] and "id|en" in canonical["languagePolicy"],
        "ai_cv_generate_marked_separate": "separate audit scope" in canonical["aiCvGenerateScope"],
        "ownership_boundary_clear": "Backend owns public cv-analysis-v2" in canonical["backendOwner"] and "Model API owns" in canonical["modelCoreOwner"],
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase-32-ai-cv-analyzer-contract-drift-audit-v1",
        "final_decision": "review_required" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "publicOpenApiContract": public_contract,
        "backendModelApiClientContract": build_backend_contract(),
        "currentModelApiContract": build_model_api_actual_contract(),
        "driftMatrix": drift_matrix,
        "canonicalContractDecision": canonical,
        "sources": {
            "openapi": str(OPENAPI_PATH.relative_to(ROOT)),
            "backend_schema": str(MODEL_SCHEMA_PATH.relative_to(ROOT)),
            "backend_client": str(MODEL_CLIENT_PATH.relative_to(ROOT)),
            "backend_service": str(AI_CV_SERVICE_PATH.relative_to(ROOT)),
            "model_api_app": str(MODEL_APP_PATH.relative_to(ROOT)),
            "model_api_schemas": str(MODEL_API_SCHEMA_PATH.relative_to(ROOT)),
            "model_api_validators": str(MODEL_VALIDATORS_PATH.relative_to(ROOT)),
        },
    }


def md_cell(value: str) -> str:
    return value.replace("|", "\\|")


def write_all() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# AI CV Analyzer Contract Drift Audit and Canonical Schema Freeze",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        "## Canonical decision",
        "",
    ]
    for key, value in report["canonicalContractDecision"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(["", "## Public OpenAPI contract", ""])
    public = report["publicOpenApiContract"]
    lines.append(f"- Source: `{public['source']}`")
    lines.append(f"- Route: `{public['method']} {public['path']}`")
    lines.append(f"- Request required: `{', '.join(public['request']['required'])}`")
    lines.append(f"- Success envelope required: `{', '.join(public['successEnvelope']['required'])}`")
    lines.append(f"- CvAnalysis required: `{', '.join(public['publicDataSchema']['required'])}`")
    lines.append(f"- AnalysisResult required: `{', '.join(public['analysisResult']['required'])}`")
    lines.append(f"- Error statuses: `{', '.join(public['errorEnvelope']['statuses'])}`")
    lines.extend(["", "## Drift matrix", ""])
    lines.append("| ID | Current drift | Canonical shape |")
    lines.append("| --- | --- | --- |")
    for row in report["driftMatrix"]:
        lines.append(f"| `{row['id']}` | {md_cell(row['current'])} | {md_cell(row['canonical'])} |")
    lines.extend(["", "## Sources", ""])
    for key, value in report["sources"].items():
        lines.append(f"- `{key}`: `{value}`")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = write_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
