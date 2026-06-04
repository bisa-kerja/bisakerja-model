from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "artifacts/backend_model_api_contract/internal_contract_fixtures.json"
MATRIX_PATH = ROOT / "artifacts/backend_model_api_contract/openapi_prisma_owner_matrix.json"
OPENAPI_PATH = ROOT / "references/docs/generated/openapi.json"
PRISMA_PATH = ROOT / "references/prisma/schema.prisma"
MODEL_API_README_PATH = ROOT / "model_api/README.md"
ROOT_README_PATH = ROOT / "README.md"
SERVICE_BOUNDARY_DOC_PATH = ROOT / "docs/architecture/service-boundaries.md"
REPORT_JSON_PATH = ROOT / "reports/phase_28_backend_model_contract_report.json"
REPORT_MD_PATH = ROOT / "reports/phase_28_backend_model_contract_report.md"

PUBLIC_WRAPPER_FIELDS = {
    "title",
    "companyName",
    "reason",
    "nextStep",
    "topActionables",
    "sectionReviews",
    "generatedCv",
    "auth",
    "persistence",
    "hydratedJob",
    "jobRecommendations",
}
REQUIRED_COMPARE_SOURCES = {"BOOKMARK", "JOB_SEARCH", "DIRECT_JOB_DETAIL"}
REQUIRED_INPUT_MODES = {"UPLOAD", "REFERENCE"}
REQUIRED_OWNER_ENTITIES = {
    "CvAnalysis.analysisResult",
    "CvAnalysisResult",
    "JobRecommendationRun",
    "JobRecommendationItem",
    "JobListing",
    "JobRequirement",
    "JobSkill",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def candidate_ids_from_fixture(fixture: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for case in fixture["positiveCases"]:
        ids.extend(candidate["jobId"] for candidate in case["form"]["jobCandidates"])
    return ids


def build_report() -> dict[str, Any]:
    fixture = load_json(FIXTURE_PATH)
    matrix = load_json(MATRIX_PATH)
    openapi = load_json(OPENAPI_PATH)
    docs = {
        "model_api_readme": MODEL_API_README_PATH.read_text(encoding="utf-8"),
        "root_readme": ROOT_README_PATH.read_text(encoding="utf-8"),
        "service_boundaries": SERVICE_BOUNDARY_DOC_PATH.read_text(encoding="utf-8"),
    }

    positive_cases = fixture["positiveCases"]
    negative_cases = fixture["negativeCases"]
    positive_compare_sources = {case["form"]["compareSource"] for case in positive_cases}
    positive_input_modes = {case["form"]["inputMode"] for case in positive_cases}
    response = fixture["modelCoreResponseExample"]
    response_keys = nested_keys(response)
    response_candidate_ids = {
        item["jobId"] for item in response["candidateReranking"]["recommendations"]
    }
    request_candidate_ids = set(candidate_ids_from_fixture(fixture))
    owner_entities = {row["entity"] for row in matrix["ownerMatrix"]}
    docs_text = "\n".join(docs.values())

    checks = {
        "contract_route_is_internal_multipart": fixture["contractRoute"] == "POST /internal/model/cv-analysis"
        and fixture["requestContentType"] == "multipart/form-data",
        "positive_fixtures_cover_compare_sources": positive_compare_sources == REQUIRED_COMPARE_SOURCES,
        "positive_fixtures_cover_input_modes": positive_input_modes == REQUIRED_INPUT_MODES,
        "negative_fixtures_cover_duplicate_empty_missing_evidence": {
            case["caseId"] for case in negative_cases
        }
        == {"duplicate-job-ids", "empty-pdf-parse", "missing-candidate-evidence"},
        "model_core_response_schema_is_v1": response["schemaVersion"] == "model-core-cv-analysis-v1",
        "model_core_response_has_required_sections": {
            "parsedCv",
            "jobFitAlignment",
            "atsFriendliness",
            "overallImpression",
            "candidateReranking",
            "model",
            "createdAt",
        }.issubset(response),
        "model_core_response_excludes_public_wrapper_fields": response_keys.isdisjoint(PUBLIC_WRAPPER_FIELDS),
        "model_recommendations_are_backend_candidate_members": response_candidate_ids.issubset(request_candidate_ids),
        "owner_matrix_covers_required_entities": owner_entities == REQUIRED_OWNER_ENTITIES,
        "enum_mapping_is_frozen": matrix["enumMappings"]["language"]["mapping"] == {"id": "ID", "en": "EN"}
        and matrix["enumMappings"]["matchLevel"]["mapping"] == {
            "strong": "STRONG",
            "good": "GOOD",
            "stretch": "STRETCH",
        },
        "openapi_contains_public_cv_analysis_v2": "CvAnalysis" in openapi["components"]["schemas"]
        and "CvAnalysisJobRecommendation" in openapi["components"]["schemas"],
        "owner_matrix_contains_persistence_entities": all(
            name in owner_entities
            for name in [
                "CvAnalysisResult",
                "JobRecommendationRun",
                "JobRecommendationItem",
                "JobListing",
                "JobRequirement",
                "JobSkill",
            ]
        ),
        "docs_reference_contract_and_fixtures": all(
            token in docs_text
            for token in [
                "POST /internal/model/cv-analysis",
                "model-core-cv-analysis-v1",
                "backendOwnsHydration",
                "artifacts/backend_model_api_contract/internal_contract_fixtures.json",
                "artifacts/backend_model_api_contract/openapi_prisma_owner_matrix.json",
            ]
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase-28-contract-realignment-report-v1",
        "final_decision": "passed" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "fixtures": {
            "positive_case_ids": [case["caseId"] for case in positive_cases],
            "negative_case_ids": [case["caseId"] for case in negative_cases],
        },
        "sources": {
            "fixtures": str(FIXTURE_PATH.relative_to(ROOT)),
            "owner_matrix": str(MATRIX_PATH.relative_to(ROOT)),
            "openapi": str(OPENAPI_PATH.relative_to(ROOT)),
            "model_api_readme": str(MODEL_API_README_PATH.relative_to(ROOT)),
            "root_readme": str(ROOT_README_PATH.relative_to(ROOT)),
            "service_boundary_doc": str(SERVICE_BOUNDARY_DOC_PATH.relative_to(ROOT)),
        },
    }


def write_all() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 28 Backend/Model API Contract Realignment Report",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(["", "## Fixtures", ""])
    lines.append("Positive: " + ", ".join(report["fixtures"]["positive_case_ids"]))
    lines.append("Negative: " + ", ".join(report["fixtures"]["negative_case_ids"]))
    lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    report = write_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
