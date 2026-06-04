#!/usr/bin/env python3
"""Verify Phase 51 AI CV Analyzer benchmark, parser, copy, and quality gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_api.app import build_cv_analysis_response_payload
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.pdf_parser import ats_score_from_pdf_evidence, normalized_skills_from_text, parse_pdf_bytes
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)
from model_api.validators import validate_model_core_payload
FIXTURE_PATH = ROOT / "artifacts/phase_51_ai_cv_analyzer_benchmark_matrix/benchmark_matrix.json"
REPORT_JSON_PATH = ROOT / "reports/phase_51_ai_cv_analyzer_benchmark_output_quality.json"
REPORT_MD_PATH = ROOT / "reports/phase_51_ai_cv_analyzer_benchmark_output_quality.md"
APP_PATH = ROOT / "model_api/app.py"
PARSER_PATH = ROOT / "model_api/pdf_parser.py"
TEST_PATH = ROOT / "tests/test_phase_51_ai_cv_analyzer_benchmark_output_quality.py"
SCHEMA_VERSION = "phase-51-ai-cv-analyzer-benchmark-output-quality-v4"
BENCHMARK_LATENCY_BUDGET_MS = 30_000
FORBIDDEN_INDONESIAN_OUTPUT_PHRASES = (
    "laporan keuangan",
    "menganalisis",
    "pengambilan keputusan",
    "manajemen proyek",
    "analisis data",
    "kemampuan komunikasi",
    "pengalaman kerja",
    "pendidikan",
    "sertifikasi",
    "keahlian",
    "keterampilan",
    "kontak tidak terdeteksi",
    "sinyal dampak terukur",
)
FORBIDDEN_GENERIC_OUTPUT_PHRASES = (
    "model-core evidence prepared for backend genai wrapper",
    "generic improvement text",
    "hiring outcome",
    "raw cv",
)
FORBIDDEN_RAW_CV_KEYS = ("rawCvText", "raw_cv_text")
FORBIDDEN_BACKEND_OWNED_FIELDS = ("topActionables", "sectionReviews", "reason", "nextStep", "generatedCv", "companyName")


class FakeModel:
    name = "phase51_verify_model"

    def predict(self, values, verbose=0):
        return [[0.64] for _ in values]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(read(path)) if path.exists() else {}


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=60)
    return {"command": " ".join(command), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def feature_config() -> TensorFlowFeatureConfig:
    features = (
        "e5_cosine",
        "skill_overlap",
        "requirement_coverage",
        "role_match",
        "experience_match",
        "experience_gap_years_clipped",
    )
    return TensorFlowFeatureConfig(approved_features=features, mean={name: 0.0 for name in features}, std={name: 1.0 for name in features})


def inference_service() -> InferenceService:
    identity = ModelIdentity(
        name="phase51",
        version="verify",
        artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
    )
    return InferenceService(
        model=FakeModel(),
        state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
    )


def benchmark_request(
    fixture: dict[str, Any],
    case_id: str,
    *,
    target_role: str | None = None,
    evidence: Any | None = None,
) -> CvAnalysisModelCoreRequest:
    cv_case = next(item for item in fixture.get("benchmarkFiles", []) if item.get("caseId") == case_id)
    roles = [role for role in fixture.get("targetRoles", []) if target_role is None or role.get("role") == target_role]
    role_skills = [skill for role in roles for skill in ((role.get("scoringInput") or {}).get("requiredSkills") or [])]
    if evidence is None:
        evidence = parse_pdf_bytes((ROOT / str(cv_case.get("path"))).read_bytes(), max_bytes=5_000_000, max_pages=10)
    candidates = tuple(
        CandidateJobInput(
            jobId=str(role.get("candidateJobId")),
            scoringInput=CandidateScoringInput(**(role.get("scoringInput") or {})),
        )
        for role in roles
    )
    return CvAnalysisModelCoreRequest(
        requestId=f"req-51-{case_id}" + (f"-{target_role}" if target_role else ""),
        inputVersion="cv-analyzer-v1",
        language="en",
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=SanitizedProfileInput(
            cvText=evidence.text,
            profileText=evidence.text[:2000],
            normalizedSkills=normalized_skills_from_text(evidence.text, role_skills),
            targetRoles=tuple(str(role.get("role")) for role in roles),
            experienceYears=evidence.estimated_experience_years,
            detectedCvSectionNames=evidence.section_names,
        ),
        jobCandidates=candidates,
        maxRecommendations=min(3, len(candidates)),
    )


def role_specific_evidence_result(fixture: dict[str, Any]) -> dict[str, Any]:
    payload = build_cv_analysis_response_payload(
        benchmark_request(fixture, "salman-abdurrahman-ats"),
        service=inference_service(),
        feature_config=feature_config(),
        calibration_policy=ScoreCalibrationPolicy(),
        embedding_backend=FakeE5Backend(),
        environment="test",
    )
    recommendations = {item["jobId"]: item for item in payload["candidateReranking"]["recommendations"]}
    evidence_pairs = {
        job_id: {"matchedSkills": item.get("matchedSkills", []), "missingSkills": item.get("missingSkills", [])}
        for job_id, item in recommendations.items()
    }
    serialized = json.dumps(payload).casefold()
    distinct = len({json.dumps(value, sort_keys=True) for value in evidence_pairs.values()}) == len(evidence_pairs)
    return {
        "passed": distinct
        and "react" in recommendations.get("phase51-software-engineer", {}).get("matchedSkills", [])
        and "product management" in recommendations.get("phase51-product-manager", {}).get("missingSkills", [])
        and "sql" in recommendations.get("phase51-data-analyst", {}).get("matchedSkills", [])
        and "forecasting" not in serialized
        and "variance analysis" not in serialized,
        "evidenceByJobId": evidence_pairs,
    }


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for child in value.values():
            output.extend(collect_strings(child))
        return output
    if isinstance(value, (list, tuple)):
        output: list[str] = []
        for child in value:
            output.extend(collect_strings(child))
        return output
    return []


def english_only_output(payload: dict[str, Any]) -> bool:
    checked = {
        "overallImpression": payload.get("overallImpression"),
        "jobFitAlignment": payload.get("jobFitAlignment"),
        "atsFriendliness": payload.get("atsFriendliness"),
        "candidateReranking": payload.get("candidateReranking"),
    }
    text = "\n".join(collect_strings(checked)).casefold()
    return all(phrase not in text for phrase in FORBIDDEN_INDONESIAN_OUTPUT_PHRASES)


def raw_cv_leakage(payload: dict[str, Any], source_text: str) -> bool:
    serialized = json.dumps(payload, sort_keys=True)
    folded = serialized.casefold()
    if any(f'"{key}"'.casefold() in folded for key in (*FORBIDDEN_RAW_CV_KEYS, *FORBIDDEN_BACKEND_OWNED_FIELDS)):
        return True
    compact_source = " ".join(source_text.split())
    snippets = [compact_source[index : index + 120] for index in (0, max(0, len(compact_source) // 2 - 60))]
    return any(len(snippet) >= 80 and snippet.casefold() in folded for snippet in snippets)


def non_generic_copy(payload: dict[str, Any]) -> bool:
    text = "\n".join(collect_strings(payload)).casefold()
    evidence = ((payload.get("overallImpression") or {}).get("evidence") or []) if isinstance(payload.get("overallImpression"), dict) else []
    return (
        all(phrase not in text for phrase in FORBIDDEN_GENERIC_OUTPUT_PHRASES)
        and any(isinstance(item, str) and item.startswith("summary:") for item in evidence)
        and any(isinstance(item, str) and "next improvement:" in item for item in evidence)
    )


def ats_issue_precision(payload: dict[str, Any], parser_issues: list[str], has_contact_signal: bool) -> bool:
    if has_contact_signal and "contact signal not detected" in parser_issues:
        return False
    ats = payload.get("atsFriendliness") if isinstance(payload.get("atsFriendliness"), dict) else {}
    issues = ats.get("detectedIssues") if isinstance(ats, dict) else []
    return all(isinstance(issue, str) and ":" in issue and "Fix:" in issue and issue != "section evidence not provided" for issue in issues)


def parser_snapshot_results(fixture: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    examples: list[dict[str, Any]] = []
    blockers: list[str] = []
    role_skills = [skill for role in fixture.get("targetRoles", []) for skill in ((role.get("scoringInput") or {}).get("requiredSkills") or [])]
    for cv_case in fixture.get("benchmarkFiles", []):
        path = ROOT / str(cv_case.get("path", ""))
        if not path.exists():
            blockers.append(f"missing benchmark PDF: {path}")
            continue
        pdf_bytes = path.read_bytes()
        actual_sha = hashlib.sha256(pdf_bytes).hexdigest()
        if actual_sha != cv_case.get("sha256"):
            blockers.append(f"sha256 mismatch: {cv_case.get('caseId')}")
        if path.stat().st_size != cv_case.get("sizeBytes"):
            blockers.append(f"size mismatch: {cv_case.get('caseId')}")
        evidence = parse_pdf_bytes(pdf_bytes, max_bytes=5_000_000, max_pages=10)
        ats_score, issues, fallback = ats_score_from_pdf_evidence(evidence)
        snapshot = cv_case.get("parserSnapshot") or {}
        if evidence.page_count != snapshot.get("expectedPageCount"):
            blockers.append(f"page count drift: {cv_case.get('caseId')}")
        if evidence.parse_quality != snapshot.get("expectedParseQuality"):
            blockers.append(f"parse quality drift: {cv_case.get('caseId')}")
        if len(evidence.text) < int(snapshot.get("minTextLength", 0)):
            blockers.append(f"text length drift: {cv_case.get('caseId')}")
        word_range = snapshot.get("wordCountRange") or [0, 999999]
        if not (word_range[0] <= evidence.word_count <= word_range[1]):
            blockers.append(f"word count drift: {cv_case.get('caseId')}")
        for section in snapshot.get("expectedSections") or []:
            if section not in evidence.section_names:
                blockers.append(f"section drift: {cv_case.get('caseId')} missing {section}")
        for title in snapshot.get("expectedRoleTitlesInclude") or []:
            if title not in evidence.role_titles:
                blockers.append(f"role title drift: {cv_case.get('caseId')} missing {title}")
        for company in snapshot.get("expectedCompaniesInclude") or []:
            if company not in evidence.company_names:
                blockers.append(f"company drift: {cv_case.get('caseId')} missing {company}")
        for issue in snapshot.get("expectedIssuesInclude") or []:
            if issue not in issues:
                blockers.append(f"issue drift: {cv_case.get('caseId')} missing {issue}")
        score_range = snapshot.get("atsScoreRange") or [0, 100]
        if not (score_range[0] <= ats_score <= score_range[1]):
            blockers.append(f"ATS score drift: {cv_case.get('caseId')}")
        skills = normalized_skills_from_text(evidence.text, role_skills)
        for skill in cv_case.get("expectedSkillsInclude") or []:
            if skill not in skills:
                blockers.append(f"skill drift: {cv_case.get('caseId')} missing {skill}")
        examples.append(
            {
                "caseId": cv_case.get("caseId"),
                "path": cv_case.get("path"),
                "pageCount": evidence.page_count,
                "parseQuality": evidence.parse_quality,
                "textLength": len(evidence.text),
                "wordCount": evidence.word_count,
                "detectedSections": list(evidence.section_names),
                "atsScore": ats_score,
                "issueCount": len(issues),
                "issues": list(issues),
                "fallback": fallback,
                "roleTitles": list(evidence.role_titles),
                "companyNames": list(evidence.company_names),
                "hasContactSignal": evidence.has_contact_signal,
                "hasDateSignal": evidence.has_date_signal,
                "skills": list(skills[:12]),
            }
        )
    return examples, blockers


def non_benchmark_role_result() -> dict[str, Any]:
    request = CvAnalysisModelCoreRequest(
        requestId="req-51-non-benchmark-role",
        inputVersion="cv-analyzer-v1",
        language="en",
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=SanitizedProfileInput(
            cvText="Security analyst with Python SQL log review and incident triage experience.",
            profileText="Security analyst with Python SQL log review and incident triage experience.",
            normalizedSkills=("python", "sql", "log review", "incident triage"),
            targetRoles=("Cybersecurity Analyst",),
            experienceYears=2.0,
            detectedCvSectionNames=("summary", "experience", "skills"),
        ),
        jobCandidates=(
            CandidateJobInput(
                jobId="phase51-cybersecurity-analyst",
                scoringInput=CandidateScoringInput(
                    titleText="Cybersecurity Analyst",
                    requiredSkills=("python", "sql", "threat modeling", "incident response"),
                    requirements=("Review security logs", "Coordinate incident response"),
                    roleFamily="security",
                    experienceBand="junior",
                ),
            ),
        ),
        maxRecommendations=1,
    )
    payload = build_cv_analysis_response_payload(
        request,
        service=inference_service(),
        feature_config=feature_config(),
        calibration_policy=ScoreCalibrationPolicy(),
        embedding_backend=FakeE5Backend(),
        environment="test",
    )
    recommendation = ((payload.get("candidateReranking") or {}).get("recommendations") or [{}])[0]
    serialized = json.dumps(payload).casefold()
    try:
        validate_model_core_payload(payload, {"phase51-cybersecurity-analyst"}, 1)
        schema_valid = True
    except Exception:
        schema_valid = False
    passed = (
        schema_valid
        and "python" in recommendation.get("matchedSkills", [])
        and "sql" in recommendation.get("matchedSkills", [])
        and "threat modeling" in recommendation.get("missingSkills", [])
        and "roleFamily=security" in recommendation.get("rankingSignals", [])
        and "phase51-software-engineer" not in serialized
        and "phase51-product-manager" not in serialized
        and "phase51-data-analyst" not in serialized
        and english_only_output(payload)
        and non_generic_copy(payload)
    )
    return {
        "passed": passed,
        "jobId": recommendation.get("jobId"),
        "matchedSkills": recommendation.get("matchedSkills", []),
        "missingSkills": recommendation.get("missingSkills", []),
        "rankingSignals": recommendation.get("rankingSignals", []),
    }


def benchmark_quality_gate_results(fixture: dict[str, Any]) -> tuple[dict[str, bool], list[dict[str, Any]], list[str]]:
    roles = [str(role.get("role")) for role in fixture.get("targetRoles", [])]
    quality_checks = {
        "benchmark_english_only_outputs": True,
        "benchmark_role_specific_evidence_gate": True,
        "benchmark_non_generic_copy_gate": True,
        "benchmark_score_spread_gate": True,
        "benchmark_no_raw_cv_leakage_gate": True,
        "benchmark_ats_issue_precision_gate": True,
        "benchmark_schema_compatibility_gate": True,
        "benchmark_bounded_latency_gate": True,
        "benchmark_no_hardcoded_role_only_behavior_gate": True,
    }
    blockers: list[str] = []
    examples: list[dict[str, Any]] = []
    scores_by_case: dict[str, list[int]] = {}
    evidence_by_case: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {}
    all_scores: list[int] = []

    for cv_case in fixture.get("benchmarkFiles", []):
        case_id = str(cv_case.get("caseId"))
        evidence = parse_pdf_bytes((ROOT / str(cv_case.get("path"))).read_bytes(), max_bytes=5_000_000, max_pages=10)
        _ats_score, parser_issues, _fallback = ats_score_from_pdf_evidence(evidence)
        for role in roles:
            request = benchmark_request(fixture, case_id, target_role=role, evidence=evidence)
            started_at = perf_counter()
            payload = build_cv_analysis_response_payload(
                request,
                service=inference_service(),
                feature_config=feature_config(),
                calibration_policy=ScoreCalibrationPolicy(),
                embedding_backend=FakeE5Backend(),
                environment="test",
                include_observability=True,
            )
            latency_ms = round((perf_counter() - started_at) * 1000)
            recommendation = ((payload.get("candidateReranking") or {}).get("recommendations") or [{}])[0]
            schema_valid = True
            try:
                validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
            except Exception:
                schema_valid = False
            case_checks = {
                "english": english_only_output(payload),
                "nonGeneric": non_generic_copy(payload),
                "noRawCvLeakage": not raw_cv_leakage(payload, evidence.text),
                "atsIssuePrecision": ats_issue_precision(payload, list(parser_issues), evidence.has_contact_signal),
                "schemaCompatible": schema_valid,
                "boundedLatency": latency_ms <= BENCHMARK_LATENCY_BUDGET_MS,
            }
            for check_name, passed in case_checks.items():
                if not passed:
                    blockers.append(f"{check_name} failed: {case_id}:{role}")
            quality_checks["benchmark_english_only_outputs"] &= case_checks["english"]
            quality_checks["benchmark_non_generic_copy_gate"] &= case_checks["nonGeneric"]
            quality_checks["benchmark_no_raw_cv_leakage_gate"] &= case_checks["noRawCvLeakage"]
            quality_checks["benchmark_ats_issue_precision_gate"] &= case_checks["atsIssuePrecision"]
            quality_checks["benchmark_schema_compatibility_gate"] &= case_checks["schemaCompatible"]
            quality_checks["benchmark_bounded_latency_gate"] &= case_checks["boundedLatency"]

            score = int((payload.get("jobFitAlignment") or {}).get("score", 0))
            scores_by_case.setdefault(case_id, []).append(score)
            evidence_by_case.setdefault(case_id, []).append((tuple(recommendation.get("matchedSkills", [])), tuple(recommendation.get("missingSkills", []))))
            all_scores.append(score)
            examples.append(
                {
                    "caseId": case_id,
                    "role": role,
                    "jobFitScore": score,
                    "matchedSkills": recommendation.get("matchedSkills", [])[:6],
                    "missingSkills": recommendation.get("missingSkills", [])[:6],
                    "atsIssues": (payload.get("atsFriendliness") or {}).get("detectedIssues", []),
                    "overallEvidence": (payload.get("overallImpression") or {}).get("evidence", []),
                    "latencyMs": latency_ms,
                }
            )

    role_specific = all(len(set(value)) == len(roles) for value in evidence_by_case.values()) if roles else False
    score_spread = len(set(all_scores)) >= 2 and len(set(scores_by_case.get("salman-abdurrahman-ats", []))) > 1
    no_hardcoded_role = non_benchmark_role_result()
    quality_checks["benchmark_role_specific_evidence_gate"] = role_specific
    quality_checks["benchmark_score_spread_gate"] = score_spread
    quality_checks["benchmark_no_hardcoded_role_only_behavior_gate"] = bool(no_hardcoded_role.get("passed"))
    if not role_specific:
        blockers.append("role-specific evidence collapsed across benchmark target roles")
    if not score_spread:
        blockers.append("benchmark score spread collapsed across target roles")
    if not no_hardcoded_role.get("passed"):
        blockers.append("non-benchmark role requirement-driven gate failed")
    return quality_checks, examples, blockers


def build_report(*, run_tests: bool = False) -> dict[str, Any]:
    fixture = read_json(FIXTURE_PATH)
    app_text = read(APP_PATH)
    parser_text = read(PARSER_PATH)
    test_text = read(TEST_PATH)
    examples, parser_blockers = parser_snapshot_results(fixture)
    role_evidence = role_specific_evidence_result(fixture) if fixture else {"passed": False, "evidenceByJobId": {}}
    quality_checks, quality_examples, quality_blockers = benchmark_quality_gate_results(fixture) if fixture else ({}, [], ["missing benchmark fixture"])
    checks = {
        "benchmark_fixture_exists": FIXTURE_PATH.exists() and fixture.get("schemaVersion") == "phase-51-ai-cv-analyzer-benchmark-matrix-v1",
        "required_upload_matrix_locked": len(fixture.get("benchmarkFiles", [])) == 3
        and len(fixture.get("targetRoles", [])) == 3
        and fixture.get("matrixExpectations", {}).get("requiredCaseCount") == 9
        and fixture.get("inputMode") == "UPLOAD",
        "parser_snapshots_match_real_pdfs": not parser_blockers,
        "score_ranges_recorded": bool((fixture.get("matrixExpectations") or {}).get("jobFitScoreRanges"))
        and all("atsScoreRange" in (case.get("parserSnapshot") or {}) for case in fixture.get("benchmarkFiles", [])),
        "english_only_policy_declared": (fixture.get("languagePolicy") or {}).get("outputLanguage") == "en"
        and bool((fixture.get("languagePolicy") or {}).get("fields")),
        "english_output_normalizer_implemented": "CANONICAL_ENGLISH_TEXT_REPLACEMENTS" in app_text
        and "_canonical_english_text" in app_text
        and "contact signal not detected" in app_text
        and "business decision support" in app_text,
        "role_specific_job_fit_evidence": bool(role_evidence.get("passed")),
        "real_pdf_parser_repairs_implemented": "COMPACT_ROLE_TITLE_MARKERS" in parser_text
        and "SPACED_CONTACT_RE" in parser_text
        and "_repair_compact_cv_text" in parser_text,
        "benchmark_regression_tests_exist": "test_benchmark_matrix_locks_required_upload_cv_role_cases_and_parser_snapshots" in test_text
        and "test_model_core_outputs_canonical_english_for_indonesian_signal_labels" in test_text
        and "test_job_fit_evidence_changes_by_role_without_unrequested_finance_gaps" in test_text
        and "test_real_pdf_parser_repairs_compact_role_company_contact_and_confidence_evidence" in test_text,
        "grounded_ats_issue_copy_implemented": "def _grounded_ats_issue_text" in app_text
        and "Section structure weak" in app_text
        and "Impact evidence weak" in app_text
        and "Fix:" in app_text,
        "grounded_overall_impression_templates_implemented": "def _overall_impression_evidence" in app_text
        and "summary: For" in app_text
        and "parser confidence:" in app_text
        and "next improvement:" in app_text,
        "phase51_5_51_6_regression_tests_exist": "test_ats_issue_copy_is_grouped_grounded_and_actionable" in test_text
        and "test_overall_impression_uses_grounded_template_not_generic_placeholder" in test_text,
        "phase51_7_regression_tests_exist": "test_phase51_7_quality_gates_cover_full_benchmark_matrix" in test_text
        and "test_phase51_7_quality_gate_uses_requirement_driven_non_benchmark_role" in test_text,
        "no_raw_cv_text_in_fixture": "text head" not in json.dumps(fixture).casefold()
        and "rawCvText" not in json.dumps(fixture),
        **quality_checks,
    }
    test_result: dict[str, Any] | None = None
    if run_tests:
        test_result = run(["python", "-m", "unittest", "tests.test_phase_51_ai_cv_analyzer_benchmark_output_quality"])
        checks["unit_tests_pass"] = test_result["returncode"] == 0
    blockers = [name for name, passed in checks.items() if not passed] + parser_blockers + quality_blockers
    return {
        "schemaVersion": SCHEMA_VERSION,
        "phaseId": "phase_51_ai_cv_analyzer_benchmark_driven_output_quality",
        "scope": "steps_51_1_51_7",
        "generatedAt": now_utc(),
        "status": "complete" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "benchmarkExamples": examples,
        "benchmarkQualityGateExamples": quality_examples,
        "roleSpecificEvidence": role_evidence.get("evidenceByJobId", {}),
        "remainingPhase51Steps": [],
        "testResult": test_result,
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks_md = "\n".join(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["checks"].items())
    examples_md = "\n".join(
        f"- `{item['caseId']}`: parseQuality={item['parseQuality']}, atsScore={item['atsScore']}, "
        f"sections={item['detectedSections']}, issueCount={item['issueCount']}"
        for item in report["benchmarkExamples"]
    )
    quality_examples_md = "\n".join(
        f"- `{item['caseId']}` / `{item['role']}`: jobFitScore={item['jobFitScore']}, "
        f"matched={item['matchedSkills']}, missing={item['missingSkills'][:3]}, latencyMs={item['latencyMs']}"
        for item in report.get("benchmarkQualityGateExamples", [])
    )
    remaining = report.get("remainingPhase51Steps", [])
    remaining_md = "\n".join(f"- {step}" for step in remaining) if remaining else "- None"
    REPORT_MD_PATH.write_text(
        "# AI CV Analyzer Benchmark Output Quality\n\n"
        f"Status: {report['status']}\n\n"
        f"Scope: `{report['scope']}`\n\n"
        "## Checks\n\n"
        f"{checks_md}\n\n"
        "## Benchmark examples\n\n"
        f"{examples_md}\n\n"
        "## Benchmark quality gate examples\n\n"
        f"{quality_examples_md}\n\n"
        "## Remaining Phase 51 steps\n\n"
        f"{remaining_md}\n",
        encoding="utf-8",
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write JSON/Markdown reports")
    parser.add_argument("--run-tests", action="store_true", help="run phase unit tests")
    args = parser.parse_args()
    report = build_report(run_tests=args.run_tests)
    if args.write:
        write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
