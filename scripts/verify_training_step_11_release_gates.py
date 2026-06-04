#!/usr/bin/env python3
"""Verify TODO stabilization Step 11 training release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/training_step_11_release_gates.json"
REPORT_MD_PATH = ROOT / "reports/training_step_11_release_gates.md"

PHASE_REPORTS = {
    "27.1_27.2": ROOT / "reports/phase_27_1_27_2_release_gate.json",
    "27.3_27.4": ROOT / "reports/phase_27_3_27_4_notebook_label_gate.json",
    "27.5_27.6": ROOT / "reports/phase_27_5_27_6_validation_expansion.json",
    "27.7": ROOT / "reports/phase_27_7_clean_kernel_production_export.json",
    "27.8": ROOT / "reports/phase_27_8_model_card_manifest_refresh.json",
    "27.10": ROOT / "reports/phase_27_10_requirement_matrix.json",
}

TRAINING_OWNED_REQUIREMENT_IDS = ("1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "4.1")
OUT_OF_SCOPE_REQUIREMENT_IDS = ("2.2", "3.1", "3.2", "deliverables")
REQUIRED_SCORE_BANDS = ("low", "medium", "high")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
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


def report_evidence(report_key: str) -> dict[str, Any]:
    path = PHASE_REPORTS[report_key]
    data = load_json(path)
    return {
        "report": file_record(path),
        "schema_version": data.get("schema_version"),
        "phase_id": data.get("phase_id"),
        "generated_at": data.get("generated_at"),
        "final_decision": data.get("final_decision"),
    }


def audit_phase_27_1_27_2() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_REPORTS["27.1_27.2"])
    steps = report.get("steps", {})
    step_27_1 = steps.get("27.1", {})
    step_27_2 = steps.get("27.2", {})

    if step_27_1.get("status") != "PASS":
        blockers.append("Phase 27.1 clean baseline is not PASS in the frozen release-gate report.")
    if step_27_2.get("status") != "PASS":
        blockers.append("Phase 27.2 TensorBoard release evidence is not PASS.")
    if report.get("artifact_manifest", {}).get("recorded_in_manifest") is not True:
        blockers.append("TensorBoard release evidence is not recorded in the Phase 25 artifact manifest.")

    return {
        **report_evidence("27.1_27.2"),
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "phase_27_1_status": step_27_1.get("status"),
        "phase_27_2_status": step_27_2.get("status"),
        "release_event_file": step_27_2.get("release_event_file"),
        "git_state": report.get("git_state", {}),
    }


def audit_phase_27_3_27_4() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    report = load_json(PHASE_REPORTS["27.3_27.4"])
    steps = report.get("steps", {})
    notebook = steps.get("27.3", {}).get("notebook_gate", {})
    label = steps.get("27.4", {}).get("label_gate", {})

    if steps.get("27.3", {}).get("status") != "PASS":
        blockers.append("Phase 27.3 notebook hygiene is not PASS.")
    if notebook.get("error_notebook_count") not in (0, None):
        blockers.append("Notebook hygiene found saved error outputs.")
    if notebook.get("active_unexecuted_notebook_count") not in (0, None):
        blockers.append("Notebook hygiene found active unexecuted code cells.")

    label_checks = {
        "frozen_label_items_present": label.get("unique_review_items", 0) > 0,
        "two_reviewers_recorded": label.get("reviewer_count", 0) >= 2,
        "minimum_two_reviewers_per_item": label.get("minimum_reviewers_per_item", 0) >= 2,
        "low_medium_high_covered": all(label.get("score_band_counts_by_item", {}).get(band, 0) > 0 for band in REQUIRED_SCORE_BANDS),
        "human_labels_not_model_inputs": label.get("human_labels_are_model_inputs") is False,
        "weak_labels_bootstrap_only": label.get("weak_labels_allowed_only_as_bootstrap_training_support") is True,
    }
    failed_label_checks = sorted(name for name, passed in label_checks.items() if not passed)
    if failed_label_checks:
        blockers.append(f"Phase 27.4 training label evidence checks failed: {failed_label_checks}.")

    release_blockers = label.get("blockers", [])
    if release_blockers:
        warnings.append(
            "Phase 27.4 release-scale production score claims remain limited by label-policy blockers; "
            "Step 11 treats this as a training release limitation, not a training-owned blocker."
        )
    if label.get("production_score_claims_allowed") is not True:
        warnings.append("Production score claims remain disabled until release-scale human validation coverage is available.")

    return {
        **report_evidence("27.3_27.4"),
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "phase_27_3_status": steps.get("27.3", {}).get("status"),
        "phase_27_4_training_label_evidence_status": "PASS" if not failed_label_checks else "FAIL",
        "label_checks": label_checks,
        "label_release_blocker_count": len(release_blockers),
        "production_score_claims_allowed": label.get("production_score_claims_allowed"),
        "unique_review_items": label.get("unique_review_items"),
        "reviewer_count": label.get("reviewer_count"),
        "score_band_counts_by_item": label.get("score_band_counts_by_item", {}),
    }


def audit_phase_27_5_27_6() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_REPORTS["27.5_27.6"])
    steps = report.get("steps", {})
    for step_id in ("27.5", "27.6"):
        if steps.get(step_id, {}).get("status") != "PASS":
            blockers.append(f"Phase {step_id} validation expansion is not PASS.")
    if report.get("final_decision") != "production-ready":
        blockers.append("Phase 27.5/27.6 validation expansion final decision is not production-ready.")

    return {
        **report_evidence("27.5_27.6"),
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "phase_27_5_status": steps.get("27.5", {}).get("status"),
        "phase_27_6_status": steps.get("27.6", {}).get("status"),
    }


def audit_phase_27_7() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_REPORTS["27.7"])
    failed_gates = sorted(gate.get("check") for gate in report.get("gates", []) if gate.get("status") != "PASS")
    if report.get("final_decision") == "blocked":
        blockers.append("Phase 27.7 clean-kernel production export is blocked.")
    if failed_gates:
        blockers.append(f"Phase 27.7 has failed gates: {failed_gates}.")

    return {
        **report_evidence("27.7"),
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "failed_gates": failed_gates,
        "gate_count": len(report.get("gates", [])),
    }


def audit_phase_27_8() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_REPORTS["27.8"])
    failed_gates = sorted(gate.get("check") for gate in report.get("gates", []) if gate.get("status") != "PASS")
    if report.get("final_decision") != "refresh-complete":
        blockers.append("Phase 27.8 model-card/manifest refresh is not complete.")
    if failed_gates:
        blockers.append(f"Phase 27.8 has failed gates: {failed_gates}.")

    return {
        **report_evidence("27.8"),
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "failed_gates": failed_gates,
        "gate_count": len(report.get("gates", [])),
    }


def audit_phase_27_10() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    report = load_json(PHASE_REPORTS["27.10"])
    rows = {item.get("requirement_id"): item for item in report.get("matrix", []) if isinstance(item, dict)}
    missing_training_rows = [req_id for req_id in TRAINING_OWNED_REQUIREMENT_IDS if req_id not in rows]
    blocked_training_rows = [
        {
            "requirement_id": req_id,
            "status": rows[req_id].get("status"),
            "failed_checks": rows[req_id].get("failed_checks", []),
            "missing_evidence": rows[req_id].get("missing_evidence", []),
            "untracked_evidence": rows[req_id].get("untracked_evidence", []),
        }
        for req_id in TRAINING_OWNED_REQUIREMENT_IDS
        if req_id in rows and rows[req_id].get("status") != "satisfied"
    ]
    blocked_out_of_scope_rows = [
        {
            "requirement_id": req_id,
            "status": rows.get(req_id, {}).get("status"),
            "failed_checks": rows.get(req_id, {}).get("failed_checks", []),
        }
        for req_id in OUT_OF_SCOPE_REQUIREMENT_IDS
        if rows.get(req_id, {}).get("status") not in (None, "satisfied")
    ]

    if missing_training_rows:
        blockers.append(f"Phase 27.10 is missing training-owned requirement rows: {missing_training_rows}.")
    if blocked_training_rows:
        blockers.append("Phase 27.10 has blocked training-owned requirement rows.")
    if blocked_out_of_scope_rows:
        warnings.append("Phase 27.10 still has non-training blockers owned by Model API or final integrated deliverables.")

    return {
        **report_evidence("27.10"),
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "training_owned_requirement_ids": list(TRAINING_OWNED_REQUIREMENT_IDS),
        "training_owned_satisfied_count": sum(
            1 for req_id in TRAINING_OWNED_REQUIREMENT_IDS if rows.get(req_id, {}).get("status") == "satisfied"
        ),
        "blocked_training_rows": blocked_training_rows,
        "out_of_scope_requirement_ids": list(OUT_OF_SCOPE_REQUIREMENT_IDS),
        "blocked_out_of_scope_rows": blocked_out_of_scope_rows,
    }


def build_report() -> dict[str, Any]:
    steps = {
        "phase_27_1_27_2": audit_phase_27_1_27_2(),
        "phase_27_3_27_4": audit_phase_27_3_27_4(),
        "phase_27_5_27_6": audit_phase_27_5_27_6(),
        "phase_27_7": audit_phase_27_7(),
        "phase_27_8": audit_phase_27_8(),
        "phase_27_10": audit_phase_27_10(),
    }
    blocking_steps = [step_id for step_id, step in steps.items() if step["status"] == "BLOCKED"]
    warning_steps = [step_id for step_id, step in steps.items() if step["status"] == "WARN"]
    acceptance = {
        "phase_27_1_and_27_2_passed": steps["phase_27_1_27_2"]["status"] != "BLOCKED",
        "phase_27_3_and_27_4_training_evidence_passed": steps["phase_27_3_27_4"]["status"] != "BLOCKED",
        "phase_27_5_and_27_6_passed": steps["phase_27_5_27_6"]["status"] != "BLOCKED",
        "phase_27_7_not_blocked": steps["phase_27_7"]["status"] != "BLOCKED",
        "phase_27_8_complete": steps["phase_27_8"]["status"] != "BLOCKED",
        "phase_27_10_training_owned_requirements_not_blocked": steps["phase_27_10"]["status"] != "BLOCKED",
    }
    return {
        "schema_version": "training-step-11-release-gates-v1",
        "phase_id": "training_step_11_release_gates",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-11-refresh-training-release-gates",
        "policy": {
            "training_owned_requirement_ids": list(TRAINING_OWNED_REQUIREMENT_IDS),
            "out_of_scope_requirement_ids": list(OUT_OF_SCOPE_REQUIREMENT_IDS),
            "production_score_claims_require_release_scale_human_validation": True,
        },
        "steps": steps,
        "blocking_steps": blocking_steps,
        "warning_steps": warning_steps,
        "acceptance": acceptance,
        "final_decision": "blocked" if blocking_steps else "pass-with-documented-limitations" if warning_steps else "pass",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Step 11 Release Gates",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Acceptance",
        "",
    ]
    for name, passed in report["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(["", "## Step Checks", ""])
    for step_id, step in report["steps"].items():
        lines.extend([f"### {step_id}", "", f"Status: **{step['status']}**", f"Report: `{step['report']['path']}`", ""])
        if step.get("blockers"):
            lines.append("Blockers:")
            lines.extend(f"- {item}" for item in step["blockers"])
            lines.append("")
        if step.get("warnings"):
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in step["warnings"])
            lines.append("")
        if not step.get("blockers") and not step.get("warnings"):
            lines.append("No blockers or warnings.")
            lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 11 release gates.")
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
