#!/usr/bin/env python3
"""Verify TODO stabilization Step 1 scope and baseline evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/training_step_1_scope_baseline.json"
REPORT_MD_PATH = ROOT / "reports/training_step_1_scope_baseline.md"

TRAINING_OWNED_OUTPUTS = [
    "jobFitAlignment",
    "atsFriendliness",
    "overallImpression",
    "candidate reranking scores for Backend-provided job IDs",
]
WRAPPER_OWNED_OUTPUTS = [
    "topActionables",
    "sectionReviews",
    "hydrated job details",
    "auth",
    "persistence",
    "request validation",
    "GenAI orchestration",
]
REQUIRED_EVIDENCE_PATHS = [
    ROOT / "training/README.md",
    ROOT / "training/notebooks/README.md",
    ROOT / "reports/phase_25_tensorflow_training_delivery.json",
    ROOT / "reports/phase_27_1_27_2_release_gate.json",
    ROOT / "reports/phase_27_3_27_4_notebook_label_gate.json",
    ROOT / "reports/phase_27_5_27_6_validation_expansion.json",
    ROOT / "reports/phase_27_7_clean_kernel_production_export.json",
    ROOT / "reports/phase_27_8_model_card_manifest_refresh.json",
    ROOT / "reports/phase_27_10_requirement_matrix.json",
]
BOUNDARY_EVIDENCE_PATHS = [
    ROOT / "reports/phase_25_training_only_genai_boundary.json",
    ROOT / "reports/phase_25_model_api_handoff_fixtures.json",
    ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json",
    ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False}
    return {
        "path": rel(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_git_status() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        return {
            "available": False,
            "error": status.stderr.strip() or status.stdout.strip(),
            "dirty": None,
            "dirty_paths": [],
        }
    dirty_paths = [line.strip() for line in status.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty_paths),
        "dirty_path_count": len(dirty_paths),
        "dirty_paths": dirty_paths,
        "production_note": (
            "Dirty state is documented for Step 1 baseline; production-ready release claims still require "
            "a clean or explicitly accepted release gate."
        ),
    }


def text_contains(path: Path, tokens: list[str]) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8", errors="ignore").lower() if path.exists() else ""
    return {token: token.lower() in text for token in tokens}


def status_from(blockers: list[str], warnings: list[str] | None = None) -> str:
    if blockers:
        return "BLOCKED"
    if warnings:
        return "WARN"
    return "PASS"


def audit_scope_boundary() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    readme_path = ROOT / "training/README.md"
    handoff_path = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json"
    wrapper_contract_path = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json"
    genai_boundary_path = ROOT / "reports/phase_25_training_only_genai_boundary.json"

    readme_checks = text_contains(
        readme_path,
        [
            "jobFitAlignment",
            "atsFriendliness",
            "overallImpression",
            "candidate reranking scores",
            "auth",
            "persistence",
            "request validation",
            "hydrated job details",
            "GenAI orchestration",
            "GenAI wrapper prose",
        ],
    )
    for name, passed in readme_checks.items():
        if not passed:
            blockers.append(f"training/README.md does not document boundary token: {name}.")

    handoff = load_json(handoff_path) if handoff_path.exists() else {}
    mapping = handoff.get("cv_analysis_v2_wrapper_mapping", {})
    wrapper_owned = set(mapping.get("wrapper_backend_owned_fields", []))
    required_wrapper_fields = {"auth", "persistence", "sectionReviews", "topActionables", "hydratedJob"}
    missing_wrapper_fields = sorted(required_wrapper_fields - wrapper_owned)
    if missing_wrapper_fields:
        blockers.append(f"Model API handoff fixture misses wrapper-owned fields: {missing_wrapper_fields}.")

    wrapper_contract = load_json(wrapper_contract_path) if wrapper_contract_path.exists() else {}
    wrapper_outputs = set(wrapper_contract.get("wrapper_outputs", {}).keys())
    if not {"sectionReviews", "topActionables"}.issubset(wrapper_outputs):
        blockers.append("GenAI wrapper handoff contract does not keep topActionables and sectionReviews wrapper-owned.")

    genai_boundary = load_json(genai_boundary_path) if genai_boundary_path.exists() else {}
    acceptance = genai_boundary.get("acceptance", {})
    if acceptance.get("training_only_genai_boundary_documented") is not True:
        blockers.append("Training-only GenAI boundary is not documented as accepted.")
    if acceptance.get("no_external_genai_calls_in_notebook_code") is not True:
        blockers.append("Training notebook GenAI call boundary did not pass.")
    if genai_boundary.get("status") != "complete":
        warnings.append("GenAI boundary report is not marked complete.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "training_owned_outputs": TRAINING_OWNED_OUTPUTS,
        "wrapper_owned_outputs": WRAPPER_OWNED_OUTPUTS,
        "readme_boundary_checks": readme_checks,
        "handoff_wrapper_owned_fields_present": sorted(required_wrapper_fields & wrapper_owned),
        "boundary_evidence": [file_record(path) for path in BOUNDARY_EVIDENCE_PATHS],
    }


def summarize_required_reports() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    report_summaries: dict[str, Any] = {}

    for path in REQUIRED_EVIDENCE_PATHS:
        record = file_record(path)
        records.append(record)
        if not record["exists"]:
            blockers.append(f"Missing required Step 1 evidence: {rel(path)}.")
            continue
        if path.suffix == ".json":
            data = load_json(path)
            summary = {
                "phase_id": data.get("phase_id"),
                "schema_version": data.get("schema_version"),
                "generated_at": data.get("generated_at"),
                "status": data.get("status"),
                "passed": data.get("passed"),
                "final_decision": data.get("final_decision"),
                "blocker_count": len(data.get("blockers", [])) if isinstance(data.get("blockers"), list) else None,
            }
            report_summaries[rel(path)] = {key: value for key, value in summary.items() if value is not None}

    phase25 = load_json(ROOT / "reports/phase_25_tensorflow_training_delivery.json")
    phase27_1 = load_json(ROOT / "reports/phase_27_1_27_2_release_gate.json")
    phase27_3 = load_json(ROOT / "reports/phase_27_3_27_4_notebook_label_gate.json")
    phase27_5 = load_json(ROOT / "reports/phase_27_5_27_6_validation_expansion.json")
    phase27_7 = load_json(ROOT / "reports/phase_27_7_clean_kernel_production_export.json")
    phase27_10 = load_json(ROOT / "reports/phase_27_10_requirement_matrix.json")

    if phase25.get("status") != "production-ready":
        warnings.append(f"Phase 25 remains {phase25.get('status')}; Step 1 records this baseline without claiming production readiness.")
    for label, report in [
        ("Phase 27.1/27.2 release gate", phase27_1),
        ("Phase 27.3/27.4 notebook/label gate", phase27_3),
        ("Phase 27.7 clean-kernel export", phase27_7),
        ("Phase 27.10 requirement matrix", phase27_10),
    ]:
        if report.get("final_decision") == "blocked":
            warnings.append(f"{label} is blocked in current evidence.")
    if phase27_5.get("final_decision") != "production-ready":
        warnings.append("Phase 27.5/27.6 validation expansion is not production-ready.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "required_evidence": records,
        "report_summaries": report_summaries,
    }


def build_report() -> dict[str, Any]:
    scope = audit_scope_boundary()
    evidence = summarize_required_reports()
    git_state = run_git_status()
    blockers = []
    warnings = []
    for section in [scope, evidence]:
        blockers.extend(section.get("blockers", []))
        warnings.extend(section.get("warnings", []))
    if git_state.get("dirty"):
        warnings.append("Current git state is dirty; dirty paths are recorded in this report.")
    if not git_state.get("available"):
        blockers.append("Git state could not be read.")

    return {
        "schema_version": "training-step-1-scope-baseline-v1",
        "phase_id": "training_step_1_scope_baseline",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-1-kunci-scope-dan-baseline",
        "scope_boundary": scope,
        "evidence_review": evidence,
        "git_state": git_state,
        "blockers": blockers,
        "warnings": warnings,
        "acceptance": {
            "training_scope_explicit": scope["status"] != "BLOCKED",
            "existing_reports_reviewed": evidence["status"] != "BLOCKED",
            "dirty_paths_documented": git_state.get("available") is True and "dirty_paths" in git_state,
        },
        "final_decision": "blocked" if blockers else "implemented-with-documented-warnings" if warnings else "pass",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Step 1 Scope and Baseline",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Scope Boundary",
        "",
        f"Status: **{report['scope_boundary']['status']}**",
        "",
        "Training-owned outputs:",
        *[f"- {item}" for item in report["scope_boundary"]["training_owned_outputs"]],
        "",
        "Wrapper/Backend-owned outputs:",
        *[f"- {item}" for item in report["scope_boundary"]["wrapper_owned_outputs"]],
        "",
        "## Evidence Review",
        "",
        f"Status: **{report['evidence_review']['status']}**",
        "",
        "Warnings:",
        *[f"- {item}" for item in report["warnings"]],
        "",
        "## Git State",
        "",
        f"Dirty: `{report['git_state'].get('dirty')}`",
        f"Dirty path count: `{report['git_state'].get('dirty_path_count')}`",
        "",
    ]
    if report["git_state"].get("dirty_paths"):
        lines.append("Dirty paths:")
        lines.extend(f"- `{item}`" for item in report["git_state"]["dirty_paths"])
        lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 1.")
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
