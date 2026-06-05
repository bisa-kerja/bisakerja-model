#!/usr/bin/env python3
"""Verify Phase 52 AI CV Analyzer wrapper context, fallback copy, prompt rules, and invariants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_api.app import build_cv_analysis_response_payload
from model_api.features import E5_MODEL_NAME, TensorFlowFeatureConfig
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy
from model_api.schemas import (
    CandidateJobInput,
    CandidateScoringInput,
    CvAnalysisModelCoreRequest,
    ModelArtifactIdentity,
    ModelIdentity,
    SanitizedProfileInput,
)
from model_api.wrapper_context import build_deterministic_fallback_copy, build_wrapper_evidence_contract, genai_analyzer_prompt_rules, validate_wrapper_output

REPORT_JSON_PATH = ROOT / "reports/phase_52_ai_cv_analyzer_wrapper_context_quality.json"
REPORT_MD_PATH = ROOT / "reports/phase_52_ai_cv_analyzer_wrapper_context_quality.md"
SCHEMA_VERSION = "phase-52-ai-cv-analyzer-wrapper-context-quality-v1"


class FakeModel:
    name = "phase52_verify_model"

    def predict(self, values, verbose=0):
        return [[0.66] for _ in values]


class FakeE5Backend:
    model_name = E5_MODEL_NAME
    backend_name = "sentence-transformers"

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=120)
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


def service() -> InferenceService:
    identity = ModelIdentity(
        name="phase52",
        version="verify",
        artifact=ModelArtifactIdentity(path="artifacts/test.keras", sha256="0" * 64),
    )
    return InferenceService(
        model=FakeModel(),
        state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase="phase_25", message="ready"),
    )


def request() -> CvAnalysisModelCoreRequest:
    return CvAnalysisModelCoreRequest(
        requestId="req-52-verify",
        inputVersion="cv-analyzer-v1",
        language="en",
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=SanitizedProfileInput(
            cvText="Summary Backend developer. Experience 2021 to 2024. Skills Python REST APIs. Education S1 Informatics. Email user@example.com.",
            profileText="Backend developer with Python REST APIs and S1 Informatics.",
            normalizedSkills=("python", "rest api"),
            targetRoles=("Backend Engineer",),
            experienceYears=2.0,
            detectedCvSectionNames=("summary", "experience", "skills", "education"),
        ),
        jobCandidates=(
            CandidateJobInput(
                jobId="phase52-backend-engineer",
                scoringInput=CandidateScoringInput(
                    titleText="Backend Engineer",
                    requiredSkills=("python", "sql", "maximum 3 years of experience"),
                    requirements=("minimum 1 years of experience", "Bachelor degree", "Remote work in Jakarta"),
                    roleFamily="backend",
                    experienceBand="junior",
                ),
            ),
        ),
        maxRecommendations=1,
    )


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for child in value.values():
            output.extend(collect_strings(child))
        return output
    if isinstance(value, list):
        output: list[str] = []
        for child in value:
            output.extend(collect_strings(child))
        return output
    return []


def build_report(*, run_tests: bool) -> dict[str, Any]:
    payload = build_cv_analysis_response_payload(
        request(),
        service=service(),
        feature_config=feature_config(),
        calibration_policy=ScoreCalibrationPolicy(),
        embedding_backend=FakeE5Backend(),
        environment="test",
    )
    wrapper = build_wrapper_evidence_contract(
        request(),
        recommendations_payload=payload["candidateReranking"]["recommendations"],
        ats_issues=payload["atsFriendliness"]["detectedIssues"],
    )
    fallback = build_deterministic_fallback_copy(wrapper)
    prompt_rules = genai_analyzer_prompt_rules()
    missing_text = json.dumps(payload["candidateReranking"]["recommendations"][0]["missingSkills"]).casefold()
    fallback_text = "\n".join(collect_strings(fallback)).casefold()
    wrapper_text = json.dumps(wrapper, sort_keys=True).casefold()
    bad_wrapper = {
        "jobFitAlignment": {"score": payload["jobFitAlignment"]["score"] + 1},
        "atsFriendliness": {"score": payload["atsFriendliness"]["score"]},
        "model": {"name": "changed", "version": "changed"},
        "jobRecommendations": [{"jobId": "changed", "reason": "Recommendation 1: improve your CV using user@example.com and raw CV."}],
    }
    validator_errors = validate_wrapper_output(payload, bad_wrapper)
    gates = {
        "wrapper_evidence_schema_present": wrapper.get("schemaVersion") == "ai-cv-analyzer-wrapper-evidence-v1",
        "required_context_fields_present": all(
            field in wrapper
            for field in (
                "parsedCv",
                "sectionEvidence",
                "requirementCoverage",
                "roleEvidence",
                "projectEvidence",
                "experienceEvidence",
                "educationEvidence",
                "certificationEvidence",
                "quantifiedImpactEvidence",
                "atsIssueEvidence",
                "candidateJobContexts",
            )
        ),
        "years_not_missing_skills": "years" not in missing_text and "maximum" not in missing_text and "minimum" not in missing_text,
        "section_evidence_grounded": any(item.get("sectionName") == "experience" for item in wrapper.get("sectionEvidence", [])),
        "fallback_no_bad_years_copy": "prove maximum" not in fallback_text,
        "prompt_rules_cover_evidence_only_and_invariants": "preserve all model-core scores" in "\n".join(prompt_rules["rules"]).casefold(),
        "validator_rejects_bad_wrapper": len(validator_errors) >= 4,
        "no_contact_value_in_wrapper": "user@example.com" not in wrapper_text,
    }
    test_result = run([sys.executable, "-m", "unittest", "tests.test_phase_52_ai_cv_analyzer_wrapper_context_quality"]) if run_tests else None
    if test_result is not None:
        gates["unit_tests_passed"] = test_result["returncode"] == 0
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": now_utc(),
        "gates": gates,
        "passed": all(gates.values()),
        "badCopyBefore": "Add one measurable bullet or project example that proves maximum 3 years of experience",
        "fallbackAfterExamples": fallback["topActionables"],
        "sectionReviewExamples": fallback["sectionReviews"][:3],
        "requirementCoverageExamples": wrapper["requirementCoverage"],
        "validatorErrorsForBadWrapper": validator_errors,
        "testResult": test_result,
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gates = "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in report["gates"].items())
    REPORT_MD_PATH.write_text(
        "# Phase 52 AI CV Analyzer Wrapper Context Quality\n\n"
        f"Generated: `{report['generatedAt']}`\n\n"
        f"Passed: `{report['passed']}`\n\n"
        "## Gates\n\n"
        f"{gates}\n\n"
        "## Bad Copy Regression\n\n"
        f"Before: `{report['badCopyBefore']}`\n\n"
        "After examples:\n"
        + "\n".join(f"- {item}" for item in report["fallbackAfterExamples"])
        + "\n\n## Requirement Coverage Examples\n\n"
        + "\n".join(f"- `{item['requirement']}` -> `{item['type']}` / `{item['coverage']}`" for item in report["requirementCoverageExamples"])
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    report = build_report(run_tests=args.run_tests)
    if args.write:
        write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
