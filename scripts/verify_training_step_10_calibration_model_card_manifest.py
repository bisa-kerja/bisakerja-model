#!/usr/bin/env python3
"""Verify TODO stabilization Step 10 calibration, model card, and manifest evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_phase_27_8_model_card_manifest_refresh import write_all as refresh_phase_27_8

ARTIFACT_ROOT = ROOT / "artifacts/phase_25_tensorflow_training_delivery"
MODEL_CARD_PATH = ARTIFACT_ROOT / "model_card.json"
ARTIFACT_MANIFEST_PATH = ARTIFACT_ROOT / "artifact_manifest.json"
SCORE_CALIBRATION_PATH = ARTIFACT_ROOT / "score_calibration.json"
TENSORBOARD_RELEASE_MANIFEST_PATH = ARTIFACT_ROOT / "tensorboard_release/manifest.json"
PHASE_27_8_REPORT_PATH = ROOT / "reports/phase_27_8_model_card_manifest_refresh.json"
REPORT_JSON_PATH = ROOT / "reports/training_step_10_calibration_model_card_manifest.json"
REPORT_MD_PATH = ROOT / "reports/training_step_10_calibration_model_card_manifest.md"

REQUIRED_SCORE_OUTPUTS = (
    "jobFitAlignment.score",
    "atsFriendliness.score",
    "recommendations[].matchScore",
)
EXPECTED_BUCKETS = ("0-20", "21-40", "41-60", "61-80", "81-100")
SELECTED_MODEL_PATH = ARTIFACT_ROOT / "export/selected_jobfit_tf_phase25.keras"


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


def path_from_repo(value: str | None) -> Path | None:
    if not value:
        return None
    return ROOT / value.replace("/", "\\")


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


def all_metric_leaves_passed(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, dict) and "passed" in value:
            if value.get("passed") is not True:
                return False
        elif isinstance(value, dict) and not all_metric_leaves_passed(value):
            return False
    return True


def audit_calibration(model_card: dict[str, Any], score_calibration: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    metrics = score_calibration.get("metrics", {})
    model_card_calibration = model_card.get("evaluation", {}).get("calibration", {})
    buckets = tuple(item.get("bucket") for item in score_calibration.get("buckets", []))

    missing_outputs = [output for output in REQUIRED_SCORE_OUTPUTS if output not in metrics]
    if missing_outputs:
        blockers.append(f"Score calibration misses output metrics: {missing_outputs}.")
    missing_model_card_outputs = [output for output in REQUIRED_SCORE_OUTPUTS if output not in model_card_calibration]
    if missing_model_card_outputs:
        blockers.append(f"Model card misses calibration evidence: {missing_model_card_outputs}.")
    if buckets != EXPECTED_BUCKETS:
        blockers.append(f"Calibration buckets are not the expected score bands: {buckets}.")
    if not all_metric_leaves_passed(metrics):
        blockers.append("One or more calibration metric leaves did not pass.")

    refresh = model_card.get("phase_27_8_release_refresh", {})
    calibration_hash = sha256_file(SCORE_CALIBRATION_PATH) if SCORE_CALIBRATION_PATH.exists() else None
    if refresh.get("calibration_hash") != calibration_hash:
        blockers.append("Model card phase_27_8 calibration hash does not match the current score_calibration.json.")
    if score_calibration.get("schema_version") != "phase-25-score-calibration-v1":
        warnings.append("Score calibration schema version is unexpected.")

    return {
        "step": "10-calibration",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "score_calibration": file_record(SCORE_CALIBRATION_PATH),
        "required_outputs": list(REQUIRED_SCORE_OUTPUTS),
        "bucket_sequence": list(buckets),
        "all_metric_leaves_passed": all_metric_leaves_passed(metrics),
        "model_card_calibration_hash": refresh.get("calibration_hash"),
    }


def audit_model_card(model_card: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    final_artifact = model_card.get("deployment_contract", {}).get("final_model_artifact", {})
    final_path = path_from_repo(final_artifact.get("path"))
    refresh = model_card.get("phase_27_8_release_refresh", {})

    if final_path != SELECTED_MODEL_PATH:
        blockers.append(f"Model card final model path does not point at the selected Phase 25 model: {final_artifact.get('path')}.")
    if not SELECTED_MODEL_PATH.exists():
        blockers.append("Selected Phase 25 Keras model is missing.")
    else:
        selected_hash = sha256_file(SELECTED_MODEL_PATH)
        if final_artifact.get("sha256") != selected_hash:
            blockers.append("Model card final model hash does not match current selected model file.")
        if final_artifact.get("size_bytes") != SELECTED_MODEL_PATH.stat().st_size:
            blockers.append("Model card final model byte size does not match current selected model file.")
        if refresh.get("model_hash") != selected_hash:
            blockers.append("Model card phase_27_8 model hash does not match current selected model file.")

    semantics = model_card.get("score_semantics", {})
    missing_semantics = [output for output in REQUIRED_SCORE_OUTPUTS if output not in semantics]
    if missing_semantics:
        blockers.append(f"Model card misses score semantics: {missing_semantics}.")

    readiness = model_card.get("readiness", {})
    if not readiness:
        warnings.append("Model card readiness section is empty.")
    if not model_card.get("intended_use") or not model_card.get("blocked_use"):
        blockers.append("Model card must include intended_use and blocked_use.")

    return {
        "step": "10-model-card",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "model_card": file_record(MODEL_CARD_PATH),
        "selected_model": file_record(SELECTED_MODEL_PATH),
        "score_semantics_outputs": sorted(semantics.keys()),
        "phase_27_8_release_refresh": refresh,
    }


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in manifest.get("artifacts", []) if isinstance(entry, dict)]


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    entries = manifest_entries(manifest)
    mismatches: list[dict[str, Any]] = []
    missing_hash_or_size: list[str] = []

    for entry in entries:
        artifact_id = entry.get("artifact_id", "<unknown>")
        path = path_from_repo(entry.get("path"))
        if path is None:
            warnings.append(f"Manifest entry {artifact_id} has no path.")
            continue
        if not path.exists():
            blockers.append(f"Manifest entry {artifact_id} points to a missing file: {entry.get('path')}.")
            continue
        current_hash = sha256_file(path)
        current_size = path.stat().st_size
        has_hash = bool(entry.get("sha256"))
        has_size = isinstance(entry.get("size_bytes"), int)
        release_relevant = entry.get("role") == "phase25_export" or entry.get("required_for_inference") is True or rel(path).startswith(
            "artifacts/phase_25_tensorflow_training_delivery/export/"
        )
        if release_relevant and (not has_hash or not has_size):
            missing_hash_or_size.append(str(artifact_id))
        if has_hash and entry.get("sha256") != current_hash:
            mismatches.append({"artifact_id": artifact_id, "field": "sha256", "manifest": entry.get("sha256"), "current": current_hash})
        if has_size and entry.get("size_bytes") != current_size:
            mismatches.append({"artifact_id": artifact_id, "field": "size_bytes", "manifest": entry.get("size_bytes"), "current": current_size})

    if missing_hash_or_size:
        blockers.append(f"Release-relevant manifest entries miss hash or byte size: {sorted(missing_hash_or_size)}.")
    if mismatches:
        blockers.append("One or more manifest entries no longer match the current artifact files.")

    refresh = manifest.get("phase_27_8_release_refresh", {})
    if not refresh:
        blockers.append("Artifact manifest does not contain phase_27_8_release_refresh.")

    return {
        "step": "10-artifact-manifest",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "artifact_manifest": file_record(ARTIFACT_MANIFEST_PATH),
        "entry_count": len(entries),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "missing_hash_or_size": sorted(missing_hash_or_size),
        "phase_27_8_release_refresh": refresh,
    }


def audit_tensorboard(model_card: dict[str, Any], tensorboard_manifest: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    event_records = tensorboard_manifest.get("event_files", [])
    ignored_check = tensorboard_manifest.get("release_git_ignore_check", {})

    if tensorboard_manifest.get("status") != "complete":
        blockers.append("TensorBoard release manifest is not complete.")
    if tensorboard_manifest.get("event_file_count", 0) < 1:
        blockers.append("TensorBoard release manifest has no event files.")
    if ignored_check.get("release_event_file_ignored") is not False or ignored_check.get("release_manifest_ignored") is not False:
        blockers.append("TensorBoard release evidence is still in an ignored path.")

    event_mismatches: list[dict[str, Any]] = []
    for record in event_records:
        path = path_from_repo(record.get("path"))
        if path is None or not path.exists():
            event_mismatches.append({"path": record.get("path"), "issue": "missing"})
            continue
        current = file_record(path)
        if current.get("sha256") != record.get("sha256") or current.get("size_bytes") != record.get("size_bytes"):
            event_mismatches.append({"path": record.get("path"), "issue": "hash_or_size_mismatch"})
    if event_mismatches:
        blockers.append("TensorBoard release event records do not match files on disk.")

    card_tensorboard = model_card.get("training", {}).get("tensorboard", {})
    if not card_tensorboard.get("release_manifest") or not card_tensorboard.get("release_event_files"):
        blockers.append("Model card does not reference TensorBoard release evidence.")

    refresh = model_card.get("phase_27_8_release_refresh", {})
    current_manifest_hash = sha256_file(TENSORBOARD_RELEASE_MANIFEST_PATH) if TENSORBOARD_RELEASE_MANIFEST_PATH.exists() else None
    if refresh.get("tensorboard_release_manifest_hash") != current_manifest_hash:
        blockers.append("Model card TensorBoard release manifest hash does not match current file.")

    return {
        "step": "10-tensorboard-release-evidence",
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "tensorboard_release_manifest": file_record(TENSORBOARD_RELEASE_MANIFEST_PATH),
        "event_file_count": tensorboard_manifest.get("event_file_count"),
        "total_event_bytes": tensorboard_manifest.get("total_event_bytes"),
        "event_mismatches": event_mismatches,
        "release_git_ignore_check": ignored_check,
    }


def audit_phase_27_8_report() -> dict[str, Any]:
    blockers: list[str] = []
    report = load_json(PHASE_27_8_REPORT_PATH) if PHASE_27_8_REPORT_PATH.exists() else {}
    gates = {item.get("check"): item.get("status") for item in report.get("gates", []) if isinstance(item, dict)}
    if report.get("final_decision") != "refresh-complete":
        blockers.append("Phase 27.8 model-card/manifest refresh report is not complete.")
    failed_gates = sorted(name for name, status in gates.items() if status != "PASS")
    if failed_gates:
        blockers.append(f"Phase 27.8 has failed gates: {failed_gates}.")
    return {
        "step": "10-phase-27-8-refresh",
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "report": file_record(PHASE_27_8_REPORT_PATH),
        "final_decision": report.get("final_decision"),
        "gates": gates,
    }


def build_report() -> dict[str, Any]:
    missing_inputs = [
        rel(path)
        for path in [MODEL_CARD_PATH, ARTIFACT_MANIFEST_PATH, SCORE_CALIBRATION_PATH, TENSORBOARD_RELEASE_MANIFEST_PATH, PHASE_27_8_REPORT_PATH]
        if not path.exists()
    ]
    if missing_inputs:
        steps = {
            "calibration": {"status": "BLOCKED", "blockers": [f"Missing required input files: {missing_inputs}"], "warnings": []},
            "model_card": {"status": "BLOCKED", "blockers": [f"Missing required input files: {missing_inputs}"], "warnings": []},
            "artifact_manifest": {"status": "BLOCKED", "blockers": [f"Missing required input files: {missing_inputs}"], "warnings": []},
            "tensorboard": {"status": "BLOCKED", "blockers": [f"Missing required input files: {missing_inputs}"], "warnings": []},
            "phase_27_8": {"status": "BLOCKED", "blockers": [f"Missing required input files: {missing_inputs}"], "warnings": []},
        }
    else:
        model_card = load_json(MODEL_CARD_PATH)
        manifest = load_json(ARTIFACT_MANIFEST_PATH)
        score_calibration = load_json(SCORE_CALIBRATION_PATH)
        tensorboard_manifest = load_json(TENSORBOARD_RELEASE_MANIFEST_PATH)
        steps = {
            "calibration": audit_calibration(model_card, score_calibration),
            "model_card": audit_model_card(model_card),
            "artifact_manifest": audit_manifest(manifest),
            "tensorboard": audit_tensorboard(model_card, tensorboard_manifest),
            "phase_27_8": audit_phase_27_8_report(),
        }

    blocking_steps = [step_id for step_id, step in steps.items() if step["status"] == "BLOCKED"]
    warning_steps = [step_id for step_id, step in steps.items() if step["status"] == "WARN"]
    acceptance = {
        "model_card_and_manifest_consistent": not any(step in blocking_steps for step in ["model_card", "artifact_manifest"]),
        "tensorboard_release_evidence_recorded": "tensorboard" not in blocking_steps,
        "no_export_artifact_mutated_without_manifest_refresh": "artifact_manifest" not in blocking_steps,
        "phase_27_8_refresh_complete": "phase_27_8" not in blocking_steps,
    }
    return {
        "schema_version": "training-step-10-calibration-model-card-manifest-v1",
        "phase_id": "training_step_10_calibration_model_card_manifest",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-10-verifikasi-calibration-model-card-dan-artifact-manifest",
        "steps": steps,
        "blocking_steps": blocking_steps,
        "warning_steps": warning_steps,
        "acceptance": acceptance,
        "final_decision": "blocked" if blocking_steps else "implemented-with-warnings" if warning_steps else "pass",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Training Step 10 Calibration, Model Card, and Manifest",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Acceptance",
        "",
    ]
    for name, passed in report["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.append("")
    lines.append("## Step Checks")
    lines.append("")
    for step_id, step in report["steps"].items():
        lines.extend([f"### {step_id}", "", f"Status: **{step['status']}**", ""])
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
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 10.")
    parser.add_argument("--write", action="store_true", help="Refresh Phase 27.8 and write Step 10 JSON/Markdown reports.")
    args = parser.parse_args()

    if args.write:
        refresh_phase_27_8()
    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
