#!/usr/bin/env python3
"""Verify Phase 27.7 clean-kernel TensorFlow production export evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE25_REPORT_PATH = ROOT / "reports/phase_25_tensorflow_training_delivery.json"
PHASE25_TRAINING_LOOP_REPORT_PATH = ROOT / "reports/phase_25_training_evaluation_loop.json"
PHASE25_ARCHITECTURE_REPORT_PATH = ROOT / "reports/phase_25_tensorflow_architecture.json"
PHASE25_EXPORT_REPORT_PATH = ROOT / "reports/phase_25_tensorflow_artifact_export.json"
PHASE25_TENSORBOARD_REPORT_PATH = ROOT / "reports/phase_25_tensorboard_monitoring.json"
PHASE25_NOTEBOOK_PATH = ROOT / "training/notebooks/phase_25_tensorflow_training_delivery.ipynb"
ARTIFACT_MANIFEST_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
FINAL_MODEL_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras"
REPORT_JSON_PATH = ROOT / "reports/phase_27_7_clean_kernel_production_export.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_7_clean_kernel_production_export.md"

REQUIRED_CHECKS = {
    "clean_kernel_top_to_bottom_reached_final_gate",
    "saved_notebook_has_no_error_outputs",
    "artifact_manifest_hashes_match_current_files",
    "tensorboard_logs_exist",
    "clean_keras_reload_subprocess",
    "api_handoff_fixtures_validated",
    "training_only_genai_boundary_enforced",
}

REQUIRED_STEP_CHECKS = {
    "step_25_5_tensorflow_architecture",
    "step_25_6_custom_component",
    "step_25_7_training_evaluation_loop",
    "step_25_8_tensorboard_monitoring",
    "step_25_11_tensorflow_artifact_export",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, check=False, text=True, capture_output=True)


def git_state() -> dict[str, Any]:
    status = run_git(["status", "--porcelain"])
    commit = run_git(["rev-parse", "HEAD"])
    dirty_paths = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty_paths) or status.returncode != 0,
        "dirty_file_count": len(dirty_paths),
        "sample": dirty_paths[:25],
        "status": "PASS" if status.returncode == 0 and not dirty_paths else "FAIL",
    }


def notebook_error_count() -> tuple[int, list[str]]:
    notebook = load_json(PHASE25_NOTEBOOK_PATH)
    errors: list[str] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"cell[{cell_index}]: {output.get('ename', 'error')}")
    return len(errors), errors[:25]


def report_checks(report: dict[str, Any]) -> dict[str, str]:
    return {item.get("check", ""): item.get("status", "UNKNOWN") for item in report.get("strict_checks", [])}


def gate_status(name: str, passed: bool, evidence: dict[str, Any] | None = None, blocker: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"check": name, "status": "PASS" if passed else "FAIL"}
    if evidence:
        payload["evidence"] = evidence
    if blocker:
        payload["blocker"] = blocker
    return payload


def build_report() -> dict[str, Any]:
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []

    required_paths = [
        PHASE25_REPORT_PATH,
        PHASE25_TRAINING_LOOP_REPORT_PATH,
        PHASE25_ARCHITECTURE_REPORT_PATH,
        PHASE25_EXPORT_REPORT_PATH,
        PHASE25_TENSORBOARD_REPORT_PATH,
        PHASE25_NOTEBOOK_PATH,
        ARTIFACT_MANIFEST_PATH,
        FINAL_MODEL_PATH,
    ]
    missing_paths = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]
    if missing_paths:
        blockers.extend(f"Missing required Phase 25 evidence: {path}" for path in missing_paths)
        phase25_report: dict[str, Any] = {}
        training_report: dict[str, Any] = {}
        architecture_report: dict[str, Any] = {}
        export_report: dict[str, Any] = {}
        tensorboard_report: dict[str, Any] = {}
    else:
        phase25_report = load_json(PHASE25_REPORT_PATH)
        training_report = load_json(PHASE25_TRAINING_LOOP_REPORT_PATH)
        architecture_report = load_json(PHASE25_ARCHITECTURE_REPORT_PATH)
        export_report = load_json(PHASE25_EXPORT_REPORT_PATH)
        tensorboard_report = load_json(PHASE25_TENSORBOARD_REPORT_PATH)

    gates.append(gate_status("required_phase25_evidence_exists", not missing_paths, {"missing_paths": missing_paths}))

    git = git_state()
    if git["dirty"]:
        blockers.append("Worktree is dirty; clean-kernel production export must be frozen from clean git state.")
    gates.append(gate_status("clean_git_state_now", not git["dirty"], git, "Dirty worktree" if git["dirty"] else None))

    phase25_status = phase25_report.get("status")
    phase25_final_reasons = phase25_report.get("final_status_reasons", {})
    phase25_production_ready = phase25_status == "production-ready"
    if not phase25_production_ready:
        blockers.append(f"Phase 25 final report status is {phase25_status!r}, not 'production-ready'.")
    gates.append(
        gate_status(
            "phase25_final_status_production_ready",
            phase25_production_ready,
            {"status": phase25_status, "final_status_reasons": phase25_final_reasons},
            "Phase 25 final status not production-ready" if not phase25_production_ready else None,
        )
    )

    strict = report_checks(phase25_report)
    missing_or_failed_strict = sorted(check for check in REQUIRED_CHECKS if strict.get(check) != "PASS")
    if missing_or_failed_strict:
        blockers.append("Phase 25 strict clean-kernel/export checks missing or failed: " + ", ".join(missing_or_failed_strict))
    gates.append(gate_status("phase25_strict_gate_checks_pass", not missing_or_failed_strict, {"checks": {key: strict.get(key) for key in sorted(REQUIRED_CHECKS)}}))

    missing_or_failed_steps = sorted(check for check in REQUIRED_STEP_CHECKS if strict.get(check) != "PASS")
    if missing_or_failed_steps:
        blockers.append("Phase 25 required step reports missing or failed: " + ", ".join(missing_or_failed_steps))
    gates.append(gate_status("phase25_required_step_reports_pass", not missing_or_failed_steps, {"checks": {key: strict.get(key) for key in sorted(REQUIRED_STEP_CHECKS)}}))

    runtime = phase25_report.get("requirement_summary", {}).get("runtime", {})
    architecture_runtime = architecture_report.get("runtime", {})
    python_bin = str(runtime.get("python", ""))
    python_version = str(architecture_runtime.get("python", ""))
    python_313 = "3.13" in python_bin or python_version.startswith("3.13") or "3.13" in json.dumps(phase25_report.get("strict_checks", []))
    if not python_313:
        blockers.append("Phase 25 clean-kernel evidence does not show Python 3.13 runtime.")
    gates.append(gate_status("python_3_13_runtime_recorded", python_313, {"python": python_bin, "architecture_python": python_version}))

    model = training_report.get("model", {})
    functional_api = model.get("api") == "Keras Functional API" or phase25_report.get("requirement_summary", {}).get("tensorflow_functional_or_subclassing") == "Keras Functional API"
    gates.append(gate_status("tensorflow_functional_api", functional_api, {"api": model.get("api")}))
    if not functional_api:
        blockers.append("TensorFlow model architecture is not recorded as Functional API.")

    custom_components = set(model.get("custom_components", []))
    custom_component_ok = {"CosineInteractionLayer", "WeightedHuberLoss", "ProductionGateCallback"}.issubset(custom_components)
    gates.append(gate_status("custom_components_present", custom_component_ok, {"custom_components": sorted(custom_components)}))
    if not custom_component_ok:
        blockers.append("Required custom layer/loss/callback evidence missing.")

    loop_ok = model.get("uses_gradient_tape") is True and model.get("uses_model_fit") is False
    gates.append(gate_status("gradient_tape_loop_no_model_fit", loop_ok, {"uses_gradient_tape": model.get("uses_gradient_tape"), "uses_model_fit": model.get("uses_model_fit")}))
    if not loop_ok:
        blockers.append("Training loop evidence must use tf.GradientTape and no model.fit().")

    metrics = training_report.get("metrics", {})
    validation_mae = metrics.get("validation", {}).get("mae")
    test_mae = metrics.get("test", {}).get("mae")
    mae_ok = isinstance(validation_mae, (int, float)) and isinstance(test_mae, (int, float)) and validation_mae <= 0.02 and test_mae <= 0.02
    gates.append(gate_status("mae_target_le_0_02", mae_ok, {"validation_mae": validation_mae, "test_mae": test_mae, "required_max": 0.02}))
    if not mae_ok:
        blockers.append("MAE target gate failed; validation and test MAE must be <= 0.02.")

    tensorboard_events = tensorboard_report.get("tensorboard", {}).get("event_file_count") or phase25_report.get("requirement_summary", {}).get("tensorboard", {}).get("event_file_count")
    tensorboard_ok = isinstance(tensorboard_events, int) and tensorboard_events > 0
    gates.append(gate_status("tensorboard_events_recorded", tensorboard_ok, {"event_file_count": tensorboard_events}))
    if not tensorboard_ok:
        blockers.append("TensorBoard release evidence missing.")

    export_checks = {item.get("check"): item.get("status") for item in export_report.get("gate_checks", [])}
    export_ok = export_report.get("export_format") == ".keras" and export_checks.get("clean_subprocess_reload") == "PASS" and FINAL_MODEL_PATH.exists()
    export_evidence: dict[str, Any] = {"format": export_report.get("export_format"), "clean_subprocess_reload": export_checks.get("clean_subprocess_reload")}
    if FINAL_MODEL_PATH.exists():
        export_evidence["final_model"] = file_record(FINAL_MODEL_PATH)
    gates.append(gate_status("keras_export_and_inference_smoke", export_ok, export_evidence))
    if not export_ok:
        blockers.append("Final .keras export or clean subprocess inference smoke failed.")

    error_count, errors = notebook_error_count() if PHASE25_NOTEBOOK_PATH.exists() else (1, ["notebook missing"])
    notebook_ok = error_count == 0
    gates.append(gate_status("phase25_notebook_has_no_saved_errors", notebook_ok, {"error_count": error_count, "errors": errors}))
    if not notebook_ok:
        blockers.append("Phase 25 notebook has saved error outputs.")

    final_decision = "production-ready" if not blockers else "blocked"
    return {
        "schema_version": "phase-27-7-clean-kernel-production-export-v1",
        "phase_id": "phase_27_7_clean_kernel_production_export",
        "generated_at": now_iso(),
        "references": ["GAP_MODEL_TRAINING.md#phase-27.7", "GAP_MODEL_TRAINING.md", "REQUIREMENT.md", "reports/phase_25_tensorflow_training_delivery.json"],
        "policy": {
            "production_ready_requires_clean_git_state": True,
            "production_ready_requires_phase25_status": "production-ready",
            "python_runtime": "3.13.x",
            "mae_max": 0.02,
            "model_fit_allowed": False,
            "tensorboard_event_min": 1,
        },
        "git_state": git,
        "phase25": {
            "final_report": PHASE25_REPORT_PATH.relative_to(ROOT).as_posix(),
            "generated_at": phase25_report.get("generated_at"),
            "status": phase25_status,
            "final_status_reasons": phase25_final_reasons,
            "runtime": {"phase25": runtime, "architecture_report": architecture_runtime},
        },
        "gates": gates,
        "blockers": blockers,
        "final_decision": final_decision,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 27.7 Clean-Kernel Production Export Gate",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        f"Git commit: `{report['git_state'].get('commit')}`",
        f"Dirty files: `{report['git_state'].get('dirty_file_count')}`",
        "",
        "## Gate checks",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- {gate['status']} — `{gate['check']}`")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Reproduction command",
        "",
        "```bash",
        "source training/.tf-venv-3.13/bin/activate",
        "jupyter nbconvert --to notebook --execute --inplace training/notebooks/phase_25_tensorflow_training_delivery.ipynb",
        "python scripts/verify_phase_27_7_clean_kernel_export.py --write",
        "```",
    ])
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 27.7 clean-kernel TensorFlow production export gate.")
    parser.add_argument("--write", action="store_true", help="Write JSON and Markdown reports under reports/.")
    args = parser.parse_args()

    report = write_all() if args.write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "production-ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
