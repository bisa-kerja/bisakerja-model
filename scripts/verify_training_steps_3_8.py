#!/usr/bin/env python3
"""Verify TODO stabilization Steps 3-8 with durable training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "training/notebooks"
REPORT_JSON_PATH = ROOT / "reports/training_steps_3_8_audit.json"
REPORT_MD_PATH = ROOT / "reports/training_steps_3_8_audit.md"

REQUIRED_NOTEBOOK_SECTIONS = ("Purpose", "Required input", "Action", "Expected output", "Verification")
RETIRED_NOTEBOOKS = {
    "phase_13_data_snapshot_contract_freezing.ipynb": "Phase 13 executable cells are intentionally retired; durable evidence remains in reports/phase_13_*.json.",
}
REQUIRED_PAIR_TYPES = {
    "high_fit_positive",
    "medium_fit",
    "hard_negative",
    "random_negative",
    "same_role_different_seniority",
    "cross_role_confusing",
}
REQUIRED_SCORE_BANDS = {"low", "medium", "high"}
REQUIRED_LABEL_SLICE_DIMENSIONS = {"score_band", "role_family", "language", "experience_band", "pair_type"}
SECRET_PATTERNS = {
    "database_url": re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb)://", re.IGNORECASE),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "password_assignment": re.compile(r"\bpassword\b\s*[:=]\s*['\"][^'\"]+", re.IGNORECASE),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
}


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
    return {"path": rel(path), "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def phase_key(path: Path) -> tuple[int, int]:
    match = re.match(r"phase_(\d+)(?:_(\d+))?", path.stem)
    if not match:
        return (9999, 9999)
    return (int(match.group(1)), int(match.group(2) or 0))


def status_from(blockers: list[str], warnings: list[str] | None = None) -> str:
    if blockers:
        return "BLOCKED"
    if warnings:
        return "WARN"
    return "PASS"


def first_markdown_source(cells: list[dict[str, Any]]) -> str:
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            return "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            return ""
    return ""


def audit_step_3() -> dict[str, Any]:
    notebooks = sorted(NOTEBOOK_DIR.glob("phase_*.ipynb"), key=phase_key)
    blockers: list[str] = []
    records: list[dict[str, Any]] = []

    if notebooks != sorted(notebooks, key=phase_key):
        blockers.append("Notebook filenames are not in phase-number order.")

    for path in notebooks:
        notebook = load_json(path)
        cells = notebook.get("cells", [])
        retired = path.name in RETIRED_NOTEBOOKS
        error_outputs: list[dict[str, Any]] = []
        unexecuted_cells: list[int] = []
        code_count = 0

        for index, cell in enumerate(cells, start=1):
            if cell.get("cell_type") != "code":
                continue
            code_count += 1
            source = "".join(cell.get("source", [])).strip()
            if source and cell.get("execution_count") is None and not retired:
                unexecuted_cells.append(index)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    error_outputs.append({"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")})

        first_md = first_markdown_source(cells)
        missing_sections = [section for section in REQUIRED_NOTEBOOK_SECTIONS if section.lower() not in first_md.lower()]
        if error_outputs:
            blockers.append(f"{rel(path)} has saved error outputs.")
        if unexecuted_cells:
            blockers.append(f"{rel(path)} has unexecuted production code cells: {unexecuted_cells}.")
        if missing_sections:
            blockers.append(f"{rel(path)} is missing documentation sections: {missing_sections}.")
        if retired and code_count != 0:
            blockers.append(f"{rel(path)} is retired but still has executable code cells.")

        records.append(
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "retired": retired,
                "retirement_reason": RETIRED_NOTEBOOKS.get(path.name),
                "code_cell_count": code_count,
                "error_output_count": len(error_outputs),
                "unexecuted_code_cells": unexecuted_cells,
                "missing_documentation_sections": missing_sections,
            }
        )

    phase13_reports = sorted((ROOT / "reports").glob("phase_13_*.json"))
    if len(phase13_reports) < 3:
        blockers.append("Phase 13 retired evidence reports are missing or incomplete.")

    return {
        "step": "3",
        "name": "Notebook hygiene",
        "status": status_from(blockers),
        "blockers": blockers,
        "notebook_count": len(records),
        "notebooks": records,
        "phase13_retired_evidence": [file_record(path) for path in phase13_reports],
    }


def scan_selected_outputs_for_secrets(paths: list[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": rel(path), "pattern": name})
    return findings


def audit_step_4() -> dict[str, Any]:
    required = [
        ROOT / "reports/phase_13_data_snapshot_contract_freezing.json",
        ROOT / "reports/phase_13_snapshot_manifests.json",
        ROOT / "reports/phase_13_model_core_schema_contracts.json",
    ]
    generated_contracts = [
        ROOT / "references/docs/generated/openapi.json",
        ROOT / "artifacts/phase_23_model_api_contract_validation/model_core_contract.json",
    ]
    selected_outputs = [
        *required,
        ROOT / "reports/phase_23_model_api_contract_validation.json",
        ROOT / "reports/phase_25_model_api_handoff_fixtures.json",
        ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json",
    ]
    blockers = [f"Missing required input evidence: {rel(path)}" for path in required if not path.exists()]
    contract_records = [file_record(path) for path in generated_contracts]
    if not any(record.get("exists") for record in contract_records):
        blockers.append("No generated Backend contract snapshot or fixture evidence was found.")

    secret_findings = scan_selected_outputs_for_secrets(selected_outputs)
    if secret_findings:
        blockers.append("Potential secret or private connection string found in generated evidence.")

    return {
        "step": "4",
        "name": "Data snapshot and contract input",
        "status": status_from(blockers),
        "blockers": blockers,
        "required_evidence": [file_record(path) for path in required],
        "generated_contract_inputs": contract_records,
        "secret_scan_findings": secret_findings,
    }


def audit_step_5() -> dict[str, Any]:
    report_path = ROOT / "reports/phase_25_e5_embedding_contract.json"
    blockers: list[str] = []
    warnings: list[str] = []
    report = load_json(report_path) if report_path.exists() else {}
    contract = report.get("embedding_contract", {})
    cache_meta = report.get("cache", {}).get("metadata", {})
    upstream = report.get("upstream_evidence", {})

    checks = {
        "embedding_model": contract.get("embedding_model") == "intfloat/e5-base-v2",
        "profile_prefix": contract.get("profile_prefix") == "query:" and cache_meta.get("profile_prefix") == "query:",
        "job_prefix": contract.get("job_prefix") == "passage:" and cache_meta.get("job_prefix") == "passage:",
        "normalized_embeddings": contract.get("normalized_embeddings") is True and cache_meta.get("normalized_embeddings") is True,
        "production_eligible_e5": contract.get("production_eligible_e5") is True,
        "fallback_forbidden": "forbidden" in str(contract.get("fallback_policy", "")).lower(),
        "phase17_phase18_e5": all(item.get("production_eligible_e5") is True for key, item in upstream.items() if key.startswith(("phase17", "phase18"))),
        "report_passed": report.get("passed") is True,
    }
    blockers.extend([f"Embedding contract check failed: {name}" for name, passed in checks.items() if not passed])

    phase14 = load_json(ROOT / "reports/phase_14_embedding_manifest.json")
    if phase14.get("embedding_model_version") == "local-hash-embedding-v1":
        warnings.append("Phase 14 keeps historical local-hash embedding evidence; Phase 25 production contract forbids fallback backends.")

    return {
        "step": "5",
        "name": "Embedding and feature contract",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "evidence": file_record(report_path),
    }


def audit_step_6() -> dict[str, Any]:
    label_gate_path = ROOT / "reports/phase_27_3_27_4_notebook_label_gate.json"
    validation_gate_path = ROOT / "reports/phase_27_5_27_6_validation_expansion.json"
    phase16_guidelines = ROOT / "reports/phase_16_reviewer_guidelines.md"
    phase16_manifest = ROOT / "reports/phase_16_label_manifest.json"
    blockers: list[str] = []
    warnings: list[str] = []

    label_gate = load_json(label_gate_path) if label_gate_path.exists() else {}
    label_summary = label_gate.get("steps", {}).get("27.4", {}).get("label_gate", {})
    label_policy = label_gate.get("steps", {}).get("27.4", {}).get("production_label_policy", {})
    validation_gate = load_json(validation_gate_path) if validation_gate_path.exists() else {}
    phase16 = load_json(phase16_manifest) if phase16_manifest.exists() else {}
    release_policy = validation_gate.get("release_policy", {})
    required_ats_cases = set(release_policy.get("ats", {}).get("required_cases", []))
    ats_case_counts = validation_gate.get("steps", {}).get("27.5", {}).get("ats_gate", {}).get("case_counts", {})
    covered_ats_cases = {case for case, count in ats_case_counts.items() if count}
    slice_dimensions = label_summary.get("slice_dimensions", {})
    present_label_slices = {dimension for dimension, buckets in slice_dimensions.items() if buckets}
    missing_label_slices = sorted(REQUIRED_LABEL_SLICE_DIMENSIONS - present_label_slices)
    undercovered_label_slices = {
        dimension: buckets
        for dimension, buckets in slice_dimensions.items()
        if dimension in REQUIRED_LABEL_SLICE_DIMENSIONS
        and buckets
        and any(count < label_policy.get("minimum_release_validation_set", {}).get("minimum_items_per_required_slice_bucket", 0) for count in buckets.values())
    }

    if not label_summary.get("weak_labels_allowed_only_as_bootstrap_training_support"):
        blockers.append("Weak-label bootstrap/training-support policy is missing.")
    if label_summary.get("human_labels_are_model_inputs") is not False:
        blockers.append("Human validation labels must remain evaluation-only, not model input features.")
    if not phase16_guidelines.exists():
        blockers.append("Reviewer guidance is missing.")
    if not phase16.get("schema_version"):
        blockers.append("Label manifest schema version is missing.")
    if validation_gate.get("final_decision") != "production-ready":
        blockers.append("Validation expansion gate is not production-ready.")
    if missing_label_slices:
        blockers.append(f"Label validation slice dimensions are missing: {missing_label_slices}.")
    missing_ats_cases = sorted(required_ats_cases - covered_ats_cases)
    if missing_ats_cases:
        blockers.append(f"ATS validation required cases are missing: {missing_ats_cases}.")

    if label_summary.get("production_score_claims_allowed") is False:
        warnings.extend(label_summary.get("blockers", []))
    if undercovered_label_slices:
        warnings.append("Label validation slices are present but remain under release-scale coverage thresholds.")

    return {
        "step": "6",
        "name": "Label governance",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "production_score_claims_allowed": label_summary.get("production_score_claims_allowed"),
        "production_claim_gate": {
            "policy_id": label_policy.get("policy_id"),
            "frozen_human_or_recruiter_reviewed_evidence_required": "frozen human or recruiter-reviewed validation labels"
            in label_policy.get("label_sources", {}).get("required_for_production_score_claims", []),
            "frozen_label_path": label_summary.get("label_path"),
            "frozen_label_sha256": label_summary.get("label_sha256"),
            "review_queue_path": label_summary.get("review_queue_path"),
            "review_queue_sha256": label_summary.get("review_queue_sha256"),
            "minimum_reviewers_per_item": label_summary.get("minimum_reviewers_per_item"),
            "production_score_claims_blocked_until_thresholds_pass": label_summary.get("production_score_claims_allowed") is False,
        },
        "validation_slice_gate": {
            "required_jobfit_slice_dimensions": sorted(REQUIRED_LABEL_SLICE_DIMENSIONS),
            "present_jobfit_slice_dimensions": sorted(present_label_slices),
            "missing_jobfit_slice_dimensions": missing_label_slices,
            "undercovered_jobfit_slice_dimensions": undercovered_label_slices,
            "required_ats_cases": sorted(required_ats_cases),
            "covered_ats_cases": sorted(covered_ats_cases),
            "missing_ats_cases": missing_ats_cases,
        },
        "unique_review_items": label_summary.get("unique_review_items"),
        "reviewer_guidelines": file_record(phase16_guidelines),
        "label_manifest": file_record(phase16_manifest),
        "label_gate": file_record(label_gate_path),
        "validation_expansion_gate": file_record(validation_gate_path),
    }


def audit_step_7() -> dict[str, Any]:
    pair_report = load_json(ROOT / "reports/phase_15_balanced_pair_generation_splits.json")
    leakage_report = load_json(ROOT / "reports/phase_15_leakage_report.json")
    diagnostics = load_json(ROOT / "reports/phase_15_pair_distribution_diagnostics.json")
    blockers: list[str] = []

    configured_types = set(pair_report.get("pair_generation_config", {}).get("required_pair_types", []))
    type_counts = pair_report.get("coverage", {}).get("pair_type_counts", {})
    score_counts = pair_report.get("coverage", {}).get("score_band_counts", {})
    missing_types = sorted(REQUIRED_PAIR_TYPES - configured_types)
    missing_count_types = sorted(pair_type for pair_type in REQUIRED_PAIR_TYPES if not type_counts.get(pair_type))
    missing_bands = sorted(score_band for score_band in REQUIRED_SCORE_BANDS if not score_counts.get(score_band))

    if missing_types:
        blockers.append(f"Required pair types missing from config: {missing_types}.")
    if missing_count_types:
        blockers.append(f"Required pair types missing from distribution counts: {missing_count_types}.")
    if missing_bands:
        blockers.append(f"Required score bands missing from distribution counts: {missing_bands}.")
    if leakage_report.get("passed") is not True or leakage_report.get("leaking_profile_count") != 0:
        blockers.append("Profile leakage report did not pass.")
    if pair_report.get("coverage", {}).get("passed") is not True:
        blockers.append("Pair coverage gate did not pass.")

    diagnostics_records = diagnostics.get("pair_type_by_split", {}).get("records", [])
    if not diagnostics_records:
        blockers.append("Pair distribution diagnostics are missing pair_type_by_split records.")

    return {
        "step": "7",
        "name": "Pair generation and split",
        "status": status_from(blockers),
        "blockers": blockers,
        "artifact": pair_report.get("artifact"),
        "pair_type_counts": type_counts,
        "score_band_counts": score_counts,
        "leakage": {
            "passed": leakage_report.get("passed"),
            "leaking_profile_count": leakage_report.get("leaking_profile_count"),
            "split_pair_counts": leakage_report.get("split_pair_counts"),
        },
        "diagnostics": file_record(ROOT / "reports/phase_15_pair_distribution_diagnostics.json"),
    }


def audit_step_8() -> dict[str, Any]:
    baseline = load_json(ROOT / "reports/phase_17_model_improvement_floor.json")
    ranking = load_json(ROOT / "reports/phase_17_ranking_metrics.json")
    slices = load_json(ROOT / "reports/phase_17_slice_metrics.json")
    selection = load_json(ROOT / "reports/phase_25_baseline_selection_gate.json")
    blockers: list[str] = []
    warnings: list[str] = []

    best_baseline = baseline.get("best_baseline")
    if not best_baseline:
        blockers.append("Best baseline is not recorded.")
    if not baseline.get("training_gate", {}).get("later_models_must_beat_best_baseline"):
        blockers.append("Model-improvement floor does not require later models to beat the best baseline.")
    if not ranking.get("splits"):
        blockers.append("Ranking metrics per split are missing.")

    slice_columns = {record.get("slice_column") for record in slices.get("records", [])}
    required_slice_columns = {"role_family", "language", "experience_band", "pair_type", "target_band"}
    missing_slice_columns = sorted(required_slice_columns - slice_columns)
    if missing_slice_columns:
        blockers.append(f"Baseline slice metrics missing dimensions: {missing_slice_columns}.")

    if selection.get("production_selection_passed") is not True:
        warnings.append("Phase 25 TensorFlow candidate does not yet pass strict production selection against Phase 18 preservation checks.")
    if selection.get("prototype_tradeoffs"):
        warnings.append("Prototype tradeoffs remain recorded in phase_25_baseline_selection_gate.json.")

    return {
        "step": "8",
        "name": "Baseline before model selection",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "best_baseline": best_baseline,
        "phase25_production_selection_passed": selection.get("production_selection_passed"),
        "phase25_final_readiness_cap": selection.get("final_readiness_cap"),
        "slice_columns": sorted(slice_columns),
        "ranking_splits": ranking.get("splits"),
        "prototype_tradeoffs": selection.get("prototype_tradeoffs", []),
    }


def build_report() -> dict[str, Any]:
    steps = {
        "3": audit_step_3(),
        "4": audit_step_4(),
        "5": audit_step_5(),
        "6": audit_step_6(),
        "7": audit_step_7(),
        "8": audit_step_8(),
    }
    blocking_steps = [step_id for step_id, step in steps.items() if step["status"] == "BLOCKED"]
    warning_steps = [step_id for step_id, step in steps.items() if step["status"] == "WARN"]
    if blocking_steps:
        final_decision = "blocked"
    elif warning_steps:
        final_decision = "implemented-with-warnings"
    else:
        final_decision = "pass"
    return {
        "schema_version": "training-steps-3-8-audit-v1",
        "phase_id": "training_steps_3_8_stabilization",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-3-through-step-8",
        "steps": steps,
        "blocking_steps": blocking_steps,
        "warning_steps": warning_steps,
        "final_decision": final_decision,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Steps 3-8 Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
    ]
    for step_id in ["3", "4", "5", "6", "7", "8"]:
        step = report["steps"][step_id]
        lines.extend([f"## Step {step_id}. {step['name']}", "", f"Status: **{step['status']}**", ""])
        blockers = step.get("blockers", [])
        warnings = step.get("warnings", [])
        if blockers:
            lines.append("Blockers:")
            lines.extend(f"- {item}" for item in blockers)
            lines.append("")
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in warnings)
            lines.append("")
        if not blockers and not warnings:
            lines.append("No blockers or warnings.")
            lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Steps 3-8.")
    parser.add_argument("--write", action="store_true", help="Write JSON and Markdown audit reports.")
    args = parser.parse_args()

    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] in {"pass", "implemented-with-warnings"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
