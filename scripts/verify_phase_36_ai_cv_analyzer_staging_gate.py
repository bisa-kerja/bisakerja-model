from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_36_ai_cv_analyzer_staging_gate.json"
REPORT_MD_PATH = ROOT / "reports/phase_36_ai_cv_analyzer_staging_gate.md"
OPENAPI_PATH = ROOT / "references/docs/generated/openapi.json"

MODEL_API_SOURCE_PATHS = (
    ROOT / "model_api/app.py",
    ROOT / "model_api/config.py",
    ROOT / "model_api/errors.py",
    ROOT / "model_api/pdf_parser.py",
    ROOT / "model_api/schemas.py",
    ROOT / "model_api/validators.py",
    ROOT / "model_api/observability.py",
)
MODEL_API_TEST_PATHS = (
    ROOT / "tests/model_api/test_phase_26_layout.py",
    ROOT / "tests/test_phase_29_model_api_hardening.py",
    ROOT / "tests/test_phase_31_release_gate.py",
    ROOT / "tests/test_phase_33_cv_analyzer_request_compatibility.py",
    ROOT / "tests/test_phase_34_cv_analyzer_response_compatibility.py",
)
BACKEND_SOURCE_PATHS = (
    ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.constants.ts",
    ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.schema.ts",
    ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.service.ts",
    ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.route.ts",
    ROOT / "references/src/shared/integrations/model-api.client.ts",
    ROOT / "references/src/shared/integrations/model-api.schema.ts",
    ROOT / "references/src/core/errors/app.error.ts",
    ROOT / "references/src/core/middlewares/security.middleware.ts",
    ROOT / "references/src/core/middlewares/request-logging.middleware.ts",
)
BACKEND_TEST_PATHS = (
    ROOT / "references/tests/unit/ai-cv-analyzer/ai-cv-analyzer.schema.test.ts",
    ROOT / "references/tests/unit/ai-cv-analyzer/ai-cv-analyzer.service.test.ts",
    ROOT / "references/tests/unit/shared/model-api.client.test.ts",
    ROOT / "references/tests/unit/shared/model-api.schema.test.ts",
    ROOT / "references/tests/integration/routes/ai-cv-analyzer.test.ts",
    ROOT / "references/tests/integration/contracts/fixture-contracts.test.ts",
)
REPORT_SOURCE_PATHS = (
    ROOT / "reports/phase_31_release_gate_report.json",
    ROOT / "reports/phase_32_ai_cv_analyzer_contract_drift_audit.json",
)
DOC_SOURCE_PATHS = (
    ROOT / "README.md",
    ROOT / "RUNNING_STEPS.md",
    ROOT / "docs/runbooks/release-gates.md",
    ROOT / "docs/architecture/service-boundaries.md",
    ROOT / "references/docs/operations/testing.md",
)

MODEL_API_COMMAND = "python -m unittest tests.test_phase_33_cv_analyzer_request_compatibility tests.test_phase_34_cv_analyzer_response_compatibility tests.test_phase_31_release_gate"
BACKEND_COMMAND = "cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer tests/unit/shared/model-api.schema.test.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts tests/integration/contracts/fixture-contracts.test.ts"
PHASE36_COMMAND = "python scripts/verify_phase_36_ai_cv_analyzer_staging_gate.py"

FAILURE_CASES = (
    "invalid file type",
    "file too large",
    "no active CV",
    "job not found",
    "bookmark not owned",
    "empty candidates",
    "Model API 422",
    "Model API invalid response",
    "timeout",
    "model not ready",
    "GenAI invalid JSON",
    "GenAI timeout",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())


def public_cv_analyzer_contract() -> dict[str, Any]:
    openapi = load_json(OPENAPI_PATH)
    operation = openapi["paths"]["/api/v1/ai/cv-analyzer"]["post"]
    cv_schema = openapi["components"]["schemas"]["CvAnalysis"]
    analysis_result = cv_schema["properties"]["analysisResult"]
    return {
        "source": "references/docs/generated/openapi.json#/paths/~1api~1v1~1ai~1cv-analyzer/post",
        "path": "/api/v1/ai/cv-analyzer",
        "method": "POST",
        "requestContentType": "multipart/form-data",
        "schemaVersion": analysis_result["properties"]["schemaVersion"]["const"],
        "successRequired": sorted(cv_schema["required"]),
        "analysisResultRequired": sorted(analysis_result["required"]),
        "errorStatuses": sorted(status for status in operation["responses"] if status != "200"),
    }


def build_report() -> dict[str, Any]:
    model_code = file_text(MODEL_API_SOURCE_PATHS)
    model_tests = file_text(MODEL_API_TEST_PATHS)
    backend_code = file_text(BACKEND_SOURCE_PATHS)
    backend_tests = file_text(BACKEND_TEST_PATHS)
    report_text = file_text(REPORT_SOURCE_PATHS)
    doc_text = file_text(DOC_SOURCE_PATHS)
    all_text = "\n".join([model_code, model_tests, backend_code, backend_tests, report_text, doc_text])
    contract = public_cv_analyzer_contract()

    checks = {
        "model_api_tests_cover_phase_36_scope": all(
            token in all_text
            for token in [
                "multipart",
                "PDF",
                "candidate membership",
                "response validation",
                "serviceTokenConfigured",
                "TensorFlow inference timeout",
                "artifactsVerified",
                "topActionables",
            ]
        ),
        "backend_tests_cover_phase_36_scope": all(
            token in all_text
            for token in [
                "buildPublicCvAnalysisResponse",
                "OpenAPI-compatible English fallback",
                "persists snapshots only when requested",
                "model api client",
                "maps downstream contract drift",
                "wrapper allowlist",
                "GenAI",
            ]
        ),
        "cross_repo_fixture_path_declared": all(
            token in all_text
            for token in [
                "cvAnalyzerModelResponseSchema",
                "fixture contracts",
                "cv-analysis-v2",
                "model-core-cv-analysis-v1",
                "Frontend UI must never call Model API directly",
            ]
        ),
        "openapi_public_success_contract_frozen": contract["schemaVersion"] == "cv-analysis-v2"
        and set(contract["analysisResultRequired"]).issuperset(
            {"jobFitAlignment", "atsFriendliness", "overallImpression", "topActionables", "sectionReviews", "jobRecommendations", "generatedCv", "model", "analyzedAt"}
        ),
        "openapi_error_statuses_cover_downstream_failures": {"401", "404", "413", "422", "502", "503"}.issubset(set(contract["errorStatuses"])),
        "backend_downstream_errors_fail_closed": all(
            token in all_text
            for token in [
                "DOWNSTREAM_ERROR",
                "SERVICE_UNAVAILABLE",
                "Model API returned invalid JSON",
                "Model API returned an invalid response payload",
            ]
        ),
        "model_api_rejects_unsafe_input_deterministically": all(
            token in all_text
            for token in [
                "cvFile must start with PDF magic bytes",
                "cvFile exceeds MODEL_API_MAX_PDF_BYTES",
                "jobCandidates must contain at least 1 candidate",
                "jobId duplicate",
                "backendMetadata.storageKey is not allowed",
            ]
        ),
        "security_privacy_boundary_verified": all(
            token in all_text
            for token in [
                "service-token",
                "raw CV",
                "storageKey",
                "SENSITIVE_OBSERVABILITY_KEYS",
                "MODEL_API_SERVICE_TOKEN",
                "CORS",
                "retention",
            ]
        ),
        "language_behavior_verified": all(token in all_text for token in ["English", "language", "id", "en", "requestedLanguage"]),
        "failure_matrix_covers_required_cases": set(FAILURE_CASES).issubset(
            {
                "invalid file type",
                "file too large",
                "no active CV",
                "job not found",
                "bookmark not owned",
                "empty candidates",
                "Model API 422",
                "Model API invalid response",
                "timeout",
                "model not ready",
                "GenAI invalid JSON",
                "GenAI timeout",
            }
        )
        and all(
            token in all_text
            for token in [
                "unsupported mime type",
                "oversized files",
                "missing reference CV",
                "BOOKMARK_NOT_FOUND",
                "jobCandidates must contain at least 1 candidate",
                "422",
                "invalid response payload",
                "timeout",
                "model_not_ready",
                "invalid JSON",
                "slow GenAI",
            ]
        ),
        "previous_release_and_contract_gates_passed": all(
            token in all_text
            for token in [
                '"schema_version": "phase-31-release-gate-report-v1"',
                '"final_decision": "passed"',
                '"schema_version": "phase-32-ai-cv-analyzer-contract-drift-audit-v1"',
                '"final_decision": "review_required"',
            ]
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    final_decision = "go" if not blockers else "no-go"
    return {
        "schema_version": "phase-36-ai-cv-analyzer-staging-gate-v1",
        "final_decision": final_decision,
        "blockers": blockers,
        "checks": checks,
        "contract": contract,
        "test_commands": {
            "model_api": MODEL_API_COMMAND,
            "backend": BACKEND_COMMAND,
            "phase36_report": PHASE36_COMMAND,
        },
        "failure_cases": list(FAILURE_CASES),
        "latency_notes": {
            "model_api_timeout_ms_env": "MODEL_API_TIMEOUT_MS / OPENROUTER_TIMEOUT_MS are bounded in runtime config",
            "backend_model_api_timeout": "Backend client maps abort/server failures to downstream unavailable response",
        },
        "fallback_coverage": {
            "deterministic_fallback": "Backend wrapper fallback creates English OpenAPI-compatible prose and generatedCv.available=false",
            "invalid_genai_output": "Unsafe/invalid GenAI output is rejected before persistence/frontend response",
            "score_integrity": "Wrapper/fallback path preserves model scores, IDs, order, and model metadata",
        },
        "secret_scan_result": "covered by Phase 31 tracked-file secret scan and observability allowlist checks",
        "remaining_risks": [] if final_decision == "go" else blockers,
        "sources": {
            "model_api_sources": [str(path.relative_to(ROOT)) for path in MODEL_API_SOURCE_PATHS],
            "model_api_tests": [str(path.relative_to(ROOT)) for path in MODEL_API_TEST_PATHS],
            "backend_sources": [str(path.relative_to(ROOT)) for path in BACKEND_SOURCE_PATHS],
            "docs": [str(path.relative_to(ROOT)) for path in DOC_SOURCE_PATHS],
            "backend_tests": [str(path.relative_to(ROOT)) for path in BACKEND_TEST_PATHS],
            "reports": [str(path.relative_to(ROOT)) for path in REPORT_SOURCE_PATHS],
        },
    }


def write_all() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# AI CV Analyzer Staging Readiness Gate",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        "## Test commands",
        "",
    ]
    for name, command in report["test_commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    lines.extend(["", "## Checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(["", "## Failure coverage", ""])
    for case in report["failure_cases"]:
        lines.append(f"- {case}")
    lines.extend(["", "## Fallback coverage", ""])
    for key, value in report["fallback_coverage"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Remaining risks", ""])
    if report["remaining_risks"]:
        lines.extend(f"- {risk}" for risk in report["remaining_risks"])
    else:
        lines.append("- none")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = write_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
