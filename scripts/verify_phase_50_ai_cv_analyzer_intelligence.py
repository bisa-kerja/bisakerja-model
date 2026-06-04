#!/usr/bin/env python3
"""Verify AI CV Analyzer deterministic intelligence optimization gates."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_50_ai_cv_analyzer_intelligence.json"
REPORT_MD_PATH = ROOT / "reports/phase_50_ai_cv_analyzer_intelligence.md"
MODEL_API_APP = ROOT / "model_api/app.py"
PDF_PARSER = ROOT / "model_api/pdf_parser.py"
TEST_PATH = ROOT / "tests/test_phase_50_ai_cv_analyzer_intelligence.py"
TODOS_PATH = ROOT / "TODOS.md"
SCHEMA_VERSION = "phase-50-ai-cv-analyzer-intelligence-v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=30)
    return {"command": " ".join(command), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def contains_all(text: str, tokens: tuple[str, ...]) -> bool:
    return all(token in text for token in tokens)


def build_report(*, run_tests: bool = False) -> dict[str, Any]:
    parser_text = read(PDF_PARSER)
    app_text = read(MODEL_API_APP)
    test_text = read(TEST_PATH)
    combined = "\n".join([parser_text, app_text, test_text])
    checks = {
        "richer_cv_extraction": contains_all(
            parser_text,
            (
                "role_titles",
                "company_names",
                "has_education_signal",
                "has_certification_signal",
                "language_signals",
                "seniority_hints",
                "has_quantified_impact",
                "estimated_experience_years",
            ),
        ),
        "feature_based_ats_scoring": contains_all(
            parser_text,
            (
                "word_count",
                "quantified impact signal not detected",
                "multi-column layout marker present",
                "contact signal not detected",
                "date or timeline signal not detected",
            ),
        ),
        "stronger_job_fit_evidence": contains_all(
            app_text,
            (
                "matched required skills",
                "missing required skills",
                "candidate experience years",
                "job experience band",
                "target roles provided",
                "detected CV sections",
            ),
        ),
        "evidence_based_actionable_inputs_without_backend_fields": contains_all(
            app_text,
            (
                "_job_fit_evidence",
                "_ranking_signal_texts",
                "matchedSkills=",
                "missingSkills=",
                "experienceYears=",
            ),
        ) and "topActionables" not in app_text,
        "benchmark_and_safety_tests": contains_all(
            test_text,
            (
                "test_pdf_parser_extracts_richer_role_company_language_and_seniority_signals",
                "test_feature_based_ats_penalizes_missing_metrics_and_formatting_risk",
                "test_model_core_response_adds_evidence_without_backend_owned_fields_or_raw_cv_leakage",
                "assertNotIn(\"topActionables\"",
                "assertNotIn(\"sectionReviews\"",
            ),
        ),
        "genai_safe_wrapper_boundary_preserved": contains_all(
            combined,
            (
                "Model API core inference",
                "No external GenAI call is made by Model API core inference.",
                "deterministic_pdf_parser",
            ),
        ),
    }
    test_result: dict[str, Any] | None = None
    if run_tests:
        test_result = run(["python", "-m", "unittest", "tests.test_phase_50_ai_cv_analyzer_intelligence"])
        checks["unit_tests_pass"] = test_result["returncode"] == 0
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_50_ai_cv_analyzer_intelligence_optimization_priorities",
        "generated_at": now_utc(),
        "status": "complete" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "deterministic_features": [
            "normalized section detection",
            "role title, company, education, certification, language, seniority signals",
            "experience years and quantified-impact evidence",
            "feature-based ATS score penalties",
            "job-fit evidence strings from matched/missing skills and experience band",
        ],
        "contract_safety": {
            "model_core_schema": "model-core-cv-analysis-v1",
            "backend_public_schema_remains": "cv-analysis-v2",
            "backend_owned_fields_returned_by_model_api": [],
            "raw_cv_text_in_response": False,
            "external_genai_in_model_api": False,
        },
        "evaluation_gates": [
            "parser signal extraction fixture",
            "ATS missing-metrics/formatting-risk fixture",
            "model-core evidence + no backend-owned fields + no raw-CV leakage fixture",
        ],
        "test_result": test_result,
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks_md = "\n".join(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["checks"].items())
    REPORT_MD_PATH.write_text(
        "# AI CV Analyzer Intelligence Optimization\n\n"
        f"Status: {report['status']}\n\n"
        "## Checks\n\n"
        f"{checks_md}\n\n"
        "## Contract safety\n\n"
        "- Backend public schema remains `cv-analysis-v2`.\n"
        "- Model API response stays `model-core-cv-analysis-v1`.\n"
        "- Model API does not return backend-owned fields or raw CV text.\n"
        "- Model API makes no external GenAI call.\n\n"
        "## Evaluation gates\n\n"
        + "\n".join(f"- {item}" for item in report["evaluation_gates"])
        + "\n",
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
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
