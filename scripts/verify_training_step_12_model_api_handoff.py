#!/usr/bin/env python3
"""Verify TODO stabilization Step 12 Model API handoff evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/training_step_12_model_api_handoff.json"
REPORT_MD_PATH = ROOT / "reports/training_step_12_model_api_handoff.md"

PHASE_21_REPORT_PATH = ROOT / "reports/phase_21_backend_candidate_reranking.json"
PHASE_23_REPORT_PATH = ROOT / "reports/phase_23_model_api_contract_validation.json"
PHASE_25_HANDOFF_REPORT_PATH = ROOT / "reports/phase_25_model_api_handoff_fixtures.json"
PHASE_25_FIXTURES_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json"
PHASE_25_VALIDATION_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json"
PHASE_23_ARTIFACT_DIR = ROOT / "artifacts/phase_23_model_api_contract_validation"
BACKEND_CONTRACT_FIXTURE_DIR = ROOT / "artifacts/backend_model_api_contract"

PHASE_23_REQUIRED_ARTIFACTS = (
    "contract_fixtures.json",
    "fallback_validation.json",
    "model_core_contract.json",
    "validation_results.json",
)
HANDOFF_EXPORT_KEYS = ("handoff_fixtures", "handoff_validation", "report")
HASH_CHECK_EXPORT_KEYS = ("handoff_fixtures", "handoff_validation")
FORBIDDEN_BACKEND_WRAPPER_FIELDS = {
    "auth",
    "availability",
    "company",
    "companyName",
    "cvFile",
    "cvFileId",
    "experienceLevel",
    "generatedCv",
    "hasApplied",
    "hydratedJob",
    "isBookmarked",
    "location",
    "nextStep",
    "nextSteps",
    "persistence",
    "reason",
    "sectionReviews",
    "title",
    "topActionables",
    "userId",
    "visibility",
    "workType",
}
DISALLOWED_BACKEND_SOURCE_DIRS = ("backend", "backend_api", "bisakerja-api", "references/src")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def path_from_repo(value: str | None) -> Path | None:
    if not value:
        return None
    return ROOT / value


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    return {
        "path": rel(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def status_from(blockers: list[str], warnings: list[str] | None = None) -> str:
    if blockers:
        return "BLOCKED"
    if warnings:
        return "WARN"
    return "PASS"


def values_all_true(values: dict[str, Any]) -> bool:
    return all(value is True for value in values.values())


def collect_forbidden_paths(value: Any, forbidden_fields: set[str], prefix: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in forbidden_fields:
                matches.append(child_path)
            matches.extend(collect_forbidden_paths(child, forbidden_fields, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(collect_forbidden_paths(child, forbidden_fields, f"{prefix}[{index}]"))
    return matches


def audit_phase_21_candidate_boundary() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_21_REPORT_PATH)
    acceptance = report.get("acceptance", {})
    ranking_policy = report.get("input_schema", {}).get("rankingPolicy", {})
    contract_boundary = report.get("contract_boundary", {})
    constraint_results = report.get("constraint_results", [])
    failed_constraints = [item for item in constraint_results if item.get("passed") is not True]

    required_acceptance = {
        "backend_owns_hydration_and_copy": acceptance.get("backend_owns_hydration_and_copy") is True,
        "model_never_invents_jobs": acceptance.get("model_never_invents_jobs") is True,
        "static_job_index_not_required": acceptance.get("static_job_index_not_required") is True,
    }
    if not values_all_true(required_acceptance):
        blockers.append("Phase 21 candidate reranking boundary acceptance is incomplete.")
    if ranking_policy.get("requireCandidateJobIds") is not True:
        blockers.append("Phase 21 ranking policy does not require Backend-provided candidate job IDs.")
    if ranking_policy.get("productionStaticJobIndexAllowed") is not False:
        blockers.append("Phase 21 ranking policy still allows a production static job index.")
    if report.get("constraint_violation_rate") != 0.0:
        blockers.append("Phase 21 candidate constraint violation rate is not zero.")
    if failed_constraints:
        blockers.append("Phase 21 candidate constraint checks include failed rows.")
    if contract_boundary.get("job_index_required_for_production_output") is not False:
        blockers.append("Phase 21 contract boundary still requires a production job index.")

    return {
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "report": file_record(PHASE_21_REPORT_PATH),
        "schema_version": report.get("schema_version"),
        "phase_id": report.get("phase_id"),
        "required_acceptance": required_acceptance,
        "candidate_count": report.get("candidate_count"),
        "candidate_set_count": report.get("candidate_set_count"),
        "constraint_violation_rate": report.get("constraint_violation_rate"),
        "require_candidate_job_ids": ranking_policy.get("requireCandidateJobIds"),
        "production_static_job_index_allowed": ranking_policy.get("productionStaticJobIndexAllowed"),
        "job_index_required_for_production_output": contract_boundary.get("job_index_required_for_production_output"),
    }


def audit_phase_23_contract_validation() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_23_REPORT_PATH)
    acceptance = report.get("acceptance", {})
    fixture_checks = report.get("fixtureChecks", {})
    validation_checks = report.get("validationChecks", {})
    fallback_checks = report.get("fallbackChecks", {})
    validation_summary = report.get("validationResults", {}).get("summary", {})
    contract = report.get("modelCoreContract", {})
    rerank_response = contract.get("candidateRerankingResponse", {})
    cv_response = contract.get("cvAnalysisResponse", {})

    if report.get("readinessStatus") != "ready_for_backend_integration_fixture_review":
        blockers.append("Phase 23 readiness status is not ready for Backend integration fixture review.")
    for group_name, checks in {
        "acceptance": acceptance,
        "fixtureChecks": fixture_checks,
        "validationChecks": validation_checks,
        "fallbackChecks": fallback_checks,
    }.items():
        if not values_all_true(checks):
            blockers.append(f"Phase 23 {group_name} has incomplete checks.")
    if validation_summary.get("positiveViolationCount") != 0:
        blockers.append("Phase 23 positive fixtures have validation violations.")
    if validation_summary.get("negativeRejectedCount") != validation_summary.get("negativeFixtureCount"):
        blockers.append("Phase 23 negative fixtures are not all rejected.")
    if "jobId" not in rerank_response.get("requiredRecommendationFields", ["jobId"]):
        blockers.append("Phase 23 reranking contract does not require recommendation jobId.")
    if not set(rerank_response.get("forbiddenFields", [])).intersection(FORBIDDEN_BACKEND_WRAPPER_FIELDS):
        blockers.append("Phase 23 reranking contract does not list wrapper/backend-owned forbidden fields.")
    if not set(cv_response.get("forbiddenFields", [])).intersection(FORBIDDEN_BACKEND_WRAPPER_FIELDS):
        blockers.append("Phase 23 CV contract does not list wrapper/backend-owned forbidden fields.")

    return {
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "report": file_record(PHASE_23_REPORT_PATH),
        "schema_version": report.get("schemaVersion"),
        "phase_id": report.get("phaseId"),
        "readiness_status": report.get("readinessStatus"),
        "acceptance": acceptance,
        "fixture_checks": fixture_checks,
        "validation_checks": validation_checks,
        "fallback_checks": fallback_checks,
        "validation_summary": validation_summary,
        "reranking_response_schema_version": rerank_response.get("schemaVersion"),
        "cv_response_schema_version": cv_response.get("schemaVersion"),
    }


def audit_phase_23_artifacts() -> dict[str, Any]:
    blockers: list[str] = []
    records = {name: file_record(PHASE_23_ARTIFACT_DIR / name) for name in PHASE_23_REQUIRED_ARTIFACTS}
    missing = [name for name, record in records.items() if not record["exists"]]
    if missing:
        blockers.append(f"Missing Phase 23 Model API contract artifacts: {missing}.")

    if not missing:
        validation = load_json(PHASE_23_ARTIFACT_DIR / "validation_results.json")
        summary = validation.get("summary", {})
        if summary.get("positiveViolationCount") != 0:
            blockers.append("Phase 23 artifact validation_results has positive fixture violations.")
        if summary.get("negativeRejectedCount") != summary.get("negativeFixtureCount"):
            blockers.append("Phase 23 artifact validation_results does not reject all negative fixtures.")

    return {
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "artifact_dir": rel(PHASE_23_ARTIFACT_DIR),
        "required_artifacts": records,
    }


def audit_phase_25_handoff_fixtures() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_25_HANDOFF_REPORT_PATH)
    fixtures = load_json(PHASE_25_FIXTURES_PATH)
    validation = load_json(PHASE_25_VALIDATION_PATH)
    acceptance = report.get("acceptance", {})
    export_paths = report.get("export_paths", {})
    export_hashes = report.get("export_hashes", {})

    if not values_all_true(acceptance):
        blockers.append("Phase 25 handoff report acceptance is incomplete.")
    if sorted(export_paths) != sorted(HANDOFF_EXPORT_KEYS):
        blockers.append("Phase 25 handoff report exports paths beyond the handoff fixture/report set.")

    export_records: dict[str, dict[str, Any]] = {}
    hash_mismatches: list[str] = []
    for key in HANDOFF_EXPORT_KEYS:
        path = path_from_repo(export_paths.get(key))
        if path is None:
            blockers.append(f"Phase 25 handoff export path is missing for {key}.")
            continue
        record = file_record(path)
        export_records[key] = record
        if not record["exists"]:
            blockers.append(f"Phase 25 handoff export file is missing for {key}.")
        elif key in HASH_CHECK_EXPORT_KEYS and export_hashes.get(key) != record["sha256"]:
            hash_mismatches.append(key)
    if hash_mismatches:
        blockers.append(f"Phase 25 handoff export hashes are stale: {hash_mismatches}.")

    positive = fixtures.get("positive", {})
    request = positive.get("candidateRerankingCoreRequest", {})
    response = positive.get("candidateRerankingCoreOutput", {})
    cv_output = positive.get("cvAnalysisCoreOutput", {})
    request_candidate_ids = {item.get("jobId") for item in request.get("jobCandidates", []) if item.get("jobId")}
    response_job_ids = [item.get("jobId") for item in response.get("recommendations", [])]
    unknown_job_ids = sorted({job_id for job_id in response_job_ids if job_id not in request_candidate_ids})
    duplicate_job_ids = sorted({job_id for job_id in response_job_ids if response_job_ids.count(job_id) > 1})
    forbidden_positive_paths = sorted(
        collect_forbidden_paths(response, FORBIDDEN_BACKEND_WRAPPER_FIELDS)
        + collect_forbidden_paths(cv_output, FORBIDDEN_BACKEND_WRAPPER_FIELDS)
    )
    score_violations = [
        item.get("jobId")
        for item in response.get("recommendations", [])
        if not isinstance(item.get("matchScore"), int) or not 0 <= item.get("matchScore") <= 100
    ]
    cv_score_violations = [
        field
        for field, value in {
            "jobFitAlignment.score": cv_output.get("jobFitAlignment", {}).get("score"),
            "atsFriendliness.score": cv_output.get("atsFriendliness", {}).get("score"),
            "overallImpression.score": cv_output.get("overallImpression", {}).get("score"),
        }.items()
        if value is not None and (not isinstance(value, int) or not 0 <= value <= 100)
    ]

    if unknown_job_ids:
        blockers.append(f"Phase 25 handoff recommendations include IDs not supplied by Backend candidates: {unknown_job_ids}.")
    if duplicate_job_ids:
        blockers.append(f"Phase 25 handoff recommendations include duplicate job IDs: {duplicate_job_ids}.")
    if len(response_job_ids) > validation.get("rules", {}).get("max_recommendations", 5):
        blockers.append("Phase 25 handoff recommendations exceed the max recommendation count.")
    if forbidden_positive_paths:
        blockers.append(f"Phase 25 positive handoff fixtures include Backend-wrapper-owned fields: {forbidden_positive_paths}.")
    if score_violations or cv_score_violations:
        blockers.append("Phase 25 positive handoff fixtures include scores outside integer 0-100 bounds.")
    if validation.get("status") != "complete":
        blockers.append("Phase 25 handoff validation status is not complete.")
    if not values_all_true(validation.get("acceptance", {})):
        blockers.append("Phase 25 handoff validation acceptance is incomplete.")
    if any(validation.get("positive_validation_errors", {}).values()):
        blockers.append("Phase 25 positive handoff validation has errors.")
    if not validation.get("negative_validation_errors"):
        blockers.append("Phase 25 negative handoff fixtures are missing rejection evidence.")

    return {
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "report": file_record(PHASE_25_HANDOFF_REPORT_PATH),
        "fixtures": file_record(PHASE_25_FIXTURES_PATH),
        "validation": file_record(PHASE_25_VALIDATION_PATH),
        "schema_version": report.get("schema_version"),
        "phase_id": report.get("phase_id"),
        "acceptance": acceptance,
        "export_records": export_records,
        "hash_mismatches": hash_mismatches,
        "candidate_membership": {
            "request_candidate_count": len(request_candidate_ids),
            "response_recommendation_count": len(response_job_ids),
            "unknown_job_ids": unknown_job_ids,
            "duplicate_job_ids": duplicate_job_ids,
        },
        "positive_fixture_forbidden_paths": forbidden_positive_paths,
        "score_violations": {
            "candidate_reranking": score_violations,
            "cv_analysis": cv_score_violations,
        },
        "negative_validation_case_count": len(validation.get("negative_validation_errors", {})),
        "positive_validation_errors": validation.get("positive_validation_errors", {}),
    }


def audit_backend_source_boundary() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    training_readme_text = (ROOT / "training/README.md").read_text(encoding="utf-8")
    disallowed_dirs = [path for path in DISALLOWED_BACKEND_SOURCE_DIRS if (ROOT / path).exists()]
    fixture_records = {
        "internal_contract_fixtures": file_record(BACKEND_CONTRACT_FIXTURE_DIR / "internal_contract_fixtures.json"),
        "openapi_prisma_owner_matrix": file_record(BACKEND_CONTRACT_FIXTURE_DIR / "openapi_prisma_owner_matrix.json"),
        "openapi_snapshot": file_record(ROOT / "references/docs/generated/openapi.json"),
    }

    if disallowed_dirs:
        blockers.append(f"Possible Backend source directories are present in this model repo: {disallowed_dirs}.")
    if not all(record["exists"] for record in fixture_records.values()):
        blockers.append("Expected versioned Backend contract snapshots/fixtures are missing.")
    if "Backend API is a separate repository" not in readme_text:
        blockers.append("Root README no longer documents Backend API as a separate repository.")
    if "Backend source is not part of this repository" not in training_readme_text:
        blockers.append("Training README no longer states that Backend source is outside this repository.")
    if "https://github.com/bisa-kerja/bisakerja-api" not in readme_text + training_readme_text:
        warnings.append("External Backend repository URL is not documented in the main boundary docs.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "disallowed_backend_source_dirs": disallowed_dirs,
        "allowed_versioned_contract_inputs": fixture_records,
        "root_readme_external_backend_boundary": "Backend API is a separate repository" in readme_text,
        "training_readme_source_boundary": "Backend source is not part of this repository" in training_readme_text,
    }


def build_report() -> dict[str, Any]:
    checks = {
        "phase_21_candidate_boundary": audit_phase_21_candidate_boundary(),
        "phase_23_contract_validation": audit_phase_23_contract_validation(),
        "phase_23_contract_artifacts": audit_phase_23_artifacts(),
        "phase_25_handoff_fixtures": audit_phase_25_handoff_fixtures(),
        "backend_source_boundary": audit_backend_source_boundary(),
    }
    blocking_checks = [check_id for check_id, check in checks.items() if check["status"] == "BLOCKED"]
    warning_checks = [check_id for check_id, check in checks.items() if check["status"] == "WARN"]
    acceptance = {
        "training_exports_only_handoff_fixtures": checks["phase_25_handoff_fixtures"]["status"] != "BLOCKED",
        "candidate_reranking_uses_backend_candidate_ids": checks["phase_21_candidate_boundary"]["status"] != "BLOCKED"
        and not checks["phase_25_handoff_fixtures"]["candidate_membership"]["unknown_job_ids"],
        "training_does_not_invent_or_hydrate_jobs": checks["phase_21_candidate_boundary"]["status"] != "BLOCKED"
        and not checks["phase_25_handoff_fixtures"]["positive_fixture_forbidden_paths"],
        "model_core_handoff_matches_contract": checks["phase_23_contract_validation"]["status"] != "BLOCKED"
        and checks["phase_23_contract_artifacts"]["status"] != "BLOCKED"
        and checks["phase_25_handoff_fixtures"]["status"] != "BLOCKED",
        "backend_behavior_remains_outside_training": checks["backend_source_boundary"]["status"] != "BLOCKED",
    }

    return {
        "schema_version": "training-step-12-model-api-handoff-v1",
        "phase_id": "training_step_12_model_api_handoff",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-12-siapkan-bukti-handoff-model-api",
        "policy": {
            "model_workspace_owns": [
                "model-core score/signals",
                "candidate reranking over Backend-provided job IDs",
                "versioned handoff fixtures and validation evidence",
            ],
            "backend_owns": [
                "auth",
                "persistence",
                "public REST envelope",
                "wrapper prose",
                "job visibility/availability",
                "job detail hydration",
            ],
            "forbidden_backend_wrapper_fields": sorted(FORBIDDEN_BACKEND_WRAPPER_FIELDS),
        },
        "checks": checks,
        "blocking_checks": blocking_checks,
        "warning_checks": warning_checks,
        "acceptance": acceptance,
        "final_decision": "blocked" if blocking_checks else "pass-with-documented-limitations" if warning_checks else "pass",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Step 12 Model API Handoff",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Acceptance",
        "",
    ]
    for name, passed in report["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(["", "## Checks", ""])
    for check_id, check in report["checks"].items():
        lines.extend([f"### {check_id}", "", f"Status: **{check['status']}**", ""])
        if check.get("report"):
            lines.append(f"Report: `{check['report']['path']}`")
            lines.append("")
        if check.get("fixtures"):
            lines.append(f"Fixtures: `{check['fixtures']['path']}`")
            lines.append("")
        if check.get("blockers"):
            lines.append("Blockers:")
            lines.extend(f"- {item}" for item in check["blockers"])
            lines.append("")
        if check.get("warnings"):
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in check["warnings"])
            lines.append("")
        if not check.get("blockers") and not check.get("warnings"):
            lines.append("No blockers or warnings.")
            lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 12 Model API handoff.")
    parser.add_argument("--write", action="store_true", help="Write JSON and Markdown audit reports.")
    args = parser.parse_args()

    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
