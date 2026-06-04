#!/usr/bin/env python3
"""Verify TODO stabilization Step 13 final stability evidence."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/training_step_13_final_stability.json"
REPORT_MD_PATH = ROOT / "reports/training_step_13_final_stability.md"

PHASE_25_REPORT_PATH = ROOT / "reports/phase_25_tensorflow_training_delivery.json"
PHASE_27_7_REPORT_PATH = ROOT / "reports/phase_27_7_clean_kernel_production_export.json"
PHASE_27_9_REPORT_PATH = ROOT / "reports/phase_27_9_model_api_production_smoke.json"
PHASE_31_REPORT_PATH = ROOT / "reports/phase_31_release_gate_report.json"
STEP_11_REPORT_PATH = ROOT / "reports/training_step_11_release_gates.json"
MODEL_CARD_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/model_card.json"
ARTIFACT_MANIFEST_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
EXPORT_DIR = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export"
TRAINING_README_PATH = ROOT / "training/README.md"
RUNNING_STEPS_PATH = ROOT / "RUNNING_STEPS.md"

INTENDED_DIRTY_PATTERNS = (
    "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json",
    "artifacts/phase_25_tensorflow_training_delivery/model_card.json",
    "reports/phase_25_model_api_handoff_fixtures.json",
    "reports/phase_27_*.json",
    "reports/phase_27_*.md",
    "reports/training_step_*.json",
    "reports/training_step_*.md",
    "scripts/verify_phase_27_*.py",
    "scripts/verify_training_step_*.py",
    "tests/test_training_step_*.py",
    "training/TODOS.md",
)
GENERATED_EVIDENCE_PATTERNS = (
    "artifacts/phase_25_tensorflow_training_delivery/*.json",
    "artifacts/phase_25_tensorflow_training_delivery/export/*.json",
    "reports/phase_25_*.json",
    "reports/phase_27_*.json",
    "reports/phase_27_*.md",
    "reports/training_step_*.json",
    "reports/training_step_*.md",
)
FORBIDDEN_DIR_PARTS = {
    ".env",
    ".venv",
    "venv",
    ".tf-venv-3.13",
    ".uv-cache",
    ".uv-python",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
FORBIDDEN_NAME_PATTERNS = ("*.pem", "*.key", "*.p12", "*secret*", "*token*", "*raw_private*")
MAX_PENDING_FILE_BYTES = 25 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


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


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)


def normalize_status_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def git_status_entries() -> list[dict[str, str]]:
    result = run_git(["status", "--porcelain=v1"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    entries: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path_text = normalize_status_path(line[3:])
        if " -> " in path_text:
            path_text = normalize_status_path(path_text.split(" -> ", 1)[1])
        entries.append({"status": status, "path": path_text})
    return entries


def git_ls_files(args: list[str]) -> list[str]:
    result = run_git(["ls-files", *args])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [normalize_status_path(line) for line in result.stdout.splitlines() if line.strip()]


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def has_forbidden_path_part(path: str) -> bool:
    parts = {part.lower() for part in path.replace("\\", "/").split("/")}
    return any(part.lower() in parts for part in FORBIDDEN_DIR_PARTS)


def has_forbidden_name(path: str) -> bool:
    name = Path(path).name.lower()
    return any(fnmatch.fnmatch(name, pattern.lower()) for pattern in FORBIDDEN_NAME_PATTERNS)


def audit_working_tree(entries: list[dict[str, str]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    unexpected = [entry for entry in entries if not matches_any(entry["path"], INTENDED_DIRTY_PATTERNS)]
    staged = [entry for entry in entries if entry["status"][0] != " " and entry["status"][0] != "?"]

    if unexpected:
        blockers.append("Working tree contains paths outside the Step 10-13 intended evidence set.")
    if entries:
        warnings.append("Working tree is intentionally dirty with pending Step 10-13 evidence; review/stage before release commit.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "dirty_file_count": len(entries),
        "unexpected_dirty_paths": unexpected,
        "staged_file_count": len(staged),
        "staged_paths": staged,
        "intended_patterns": list(INTENDED_DIRTY_PATTERNS),
    }


def audit_generated_evidence(entries: list[dict[str, str]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    tracked = set(git_ls_files([]))
    generated_entries = [entry for entry in entries if matches_any(entry["path"], GENERATED_EVIDENCE_PATTERNS)]
    pending_track = [entry["path"] for entry in generated_entries if entry["path"] not in tracked]
    tracked_generated = [entry["path"] for entry in generated_entries if entry["path"] in tracked]
    ignored_generated: list[str] = []

    for entry in generated_entries:
        path = ROOT / entry["path"]
        if path.exists():
            ignored = run_git(["check-ignore", "-q", "--", entry["path"]])
            if ignored.returncode == 0:
                ignored_generated.append(entry["path"])

    if ignored_generated:
        blockers.append(f"Generated release evidence is ignored unexpectedly: {sorted(ignored_generated)}.")
    if pending_track:
        warnings.append("New Step verifier/report/test evidence is not tracked yet; include it in the release commit if accepted.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "tracked_generated_dirty_paths": sorted(tracked_generated),
        "pending_track_generated_paths": sorted(pending_track),
        "ignored_generated_paths": sorted(ignored_generated),
    }


def audit_sensitive_and_large_pending(entries: list[dict[str, str]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    tracked_files = git_ls_files([])
    tracked_forbidden = [path for path in tracked_files if has_forbidden_path_part(path) or has_forbidden_name(path)]
    pending_forbidden = [entry["path"] for entry in entries if has_forbidden_path_part(entry["path"]) or has_forbidden_name(entry["path"])]
    large_pending: list[dict[str, Any]] = []

    for entry in entries:
        path = ROOT / entry["path"]
        if path.is_file():
            size_bytes = path.stat().st_size
            if size_bytes > MAX_PENDING_FILE_BYTES:
                large_pending.append({"path": entry["path"], "size_bytes": size_bytes})

    if tracked_forbidden:
        warnings.append("Repository already contains historical cache/secret-like tracked paths; no new pending path may add to this.")
    if pending_forbidden:
        blockers.append("Pending changes include cache, venv, secret-like, or private-data paths.")
    if large_pending:
        warnings.append("Pending changes include large files; confirm each is intentional release evidence.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "tracked_forbidden_paths": sorted(tracked_forbidden),
        "pending_forbidden_paths": sorted(pending_forbidden),
        "large_pending_paths": large_pending,
        "max_pending_file_bytes": MAX_PENDING_FILE_BYTES,
    }


def audit_docs() -> dict[str, Any]:
    blockers: list[str] = []
    training_readme = TRAINING_README_PATH.read_text(encoding="utf-8")
    running_steps = RUNNING_STEPS_PATH.read_text(encoding="utf-8")
    checks = {
        "training_readme_python_3_13": "Python `3.13.x`" in training_readme and "3.13.13" in training_readme,
        "training_readme_phase25_workflow": "phase_25_tensorflow_training_delivery.ipynb" in training_readme,
        "training_readme_release_gates": "Before any production-ready claim" in training_readme,
        "training_readme_artifact_outputs": "phase_25_tensorflow_training_delivery" in training_readme,
        "running_steps_python_3_13": "Python: `3.13.x`" in running_steps and "3.13.13" in running_steps,
        "running_steps_phase25": "Run Phase 25 Notebook" in running_steps,
        "running_steps_model_api_runtime": "Model API Runtime" in running_steps,
        "running_steps_backend_boundary": "Backend repo outside this model repository" in running_steps,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        blockers.append(f"Runtime/workflow docs are missing required Step 13 anchors: {failed}.")
    return {
        "status": status_from(blockers),
        "blockers": blockers,
        "warnings": [],
        "checks": checks,
        "documents": [rel(TRAINING_README_PATH), rel(RUNNING_STEPS_PATH)],
    }


def audit_release_evidence() -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    phase25 = load_json(PHASE_25_REPORT_PATH)
    phase27_7 = load_json(PHASE_27_7_REPORT_PATH)
    phase27_9 = load_json(PHASE_27_9_REPORT_PATH)
    phase31 = load_json(PHASE_31_REPORT_PATH)
    step11 = load_json(STEP_11_REPORT_PATH)
    model_card = load_json(MODEL_CARD_PATH)

    if phase25.get("status") != "production-ready":
        blockers.append("Phase 25 final report is not production-ready.")
    if phase27_7.get("final_decision") != "production-ready":
        blockers.append("Phase 27.7 clean-kernel export gate is not production-ready.")
    if step11.get("final_decision") == "blocked":
        blockers.append("Training Step 11 release gate audit is blocked.")
    if phase31.get("final_decision") != "passed":
        blockers.append("Phase 31 release gate is not passed.")
    if phase27_9.get("final_decision") == "blocked":
        warnings.append("Model API production smoke remains blocked and must stay documented as a serving/integration limitation.")
    if model_card.get("phase_27_8_release_refresh", {}).get("production_claim_allowed") is False:
        warnings.append("Model card still disables broad production score claims until all policy/integration gates are satisfied.")

    return {
        "status": status_from(blockers, warnings),
        "blockers": blockers,
        "warnings": warnings,
        "phase25_status": phase25.get("status"),
        "phase25_generated_at": phase25.get("generated_at"),
        "phase27_7_final_decision": phase27_7.get("final_decision"),
        "phase27_7_generated_at": phase27_7.get("generated_at"),
        "step11_final_decision": step11.get("final_decision"),
        "phase27_9_final_decision": phase27_9.get("final_decision"),
        "phase27_9_blockers": phase27_9.get("blockers", []),
        "phase31_final_decision": phase31.get("final_decision"),
        "model_card_production_claim_allowed": model_card.get("phase_27_8_release_refresh", {}).get("production_claim_allowed"),
        "evidence_files": {
            "phase25": file_record(PHASE_25_REPORT_PATH),
            "phase27_7": file_record(PHASE_27_7_REPORT_PATH),
            "step11": file_record(STEP_11_REPORT_PATH),
            "phase27_9": file_record(PHASE_27_9_REPORT_PATH),
            "phase31": file_record(PHASE_31_REPORT_PATH),
            "model_card": file_record(MODEL_CARD_PATH),
            "artifact_manifest": file_record(ARTIFACT_MANIFEST_PATH),
        },
    }


def build_final_note(release: dict[str, Any]) -> dict[str, Any]:
    runtime = {
        "current_verifier_python": sys.version.split()[0],
        "target_python": "3.13.x",
        "tensorflow": "2.21.0",
        "keras": "3.14.1",
        "serving_smoke_python": "3.13.13",
    }
    limitations = [
        "Current git working tree contains intentional Step 10-13 evidence and still needs human review/staging before commit.",
        "Production score claims remain limited by release-scale human/recruiter validation policy.",
        "Model API production smoke is blocked until serving dependencies, Prisma snapshot, real Keras loader smoke, and live FastAPI smoke pass.",
        "Backend auth, persistence, public response formatting, and job hydration remain outside this training repository.",
    ]
    return {
        "runtime": runtime,
        "notebook_rerun_at": release.get("phase25_generated_at"),
        "clean_kernel_gate_at": release.get("phase27_7_generated_at"),
        "artifact_export_folder": rel(EXPORT_DIR),
        "phase25_status": release.get("phase25_status"),
        "release_gate_status": release.get("step11_final_decision"),
        "model_api_smoke_status": release.get("phase27_9_final_decision"),
        "limitations": limitations,
    }


def build_report() -> dict[str, Any]:
    entries = git_status_entries()
    checks = {
        "working_tree_intent": audit_working_tree(entries),
        "generated_evidence_tracking": audit_generated_evidence(entries),
        "sensitive_cache_private_data": audit_sensitive_and_large_pending(entries),
        "runtime_workflow_docs": audit_docs(),
        "release_evidence": audit_release_evidence(),
    }
    blocking_checks = [check_id for check_id, check in checks.items() if check["status"] == "BLOCKED"]
    warning_checks = [check_id for check_id, check in checks.items() if check["status"] == "WARN"]
    acceptance = {
        "working_tree_only_intended_changes": not checks["working_tree_intent"]["unexpected_dirty_paths"],
        "generated_artifacts_and_reports_accounted_for": checks["generated_evidence_tracking"]["status"] != "BLOCKED",
        "no_cache_venv_secret_or_private_data_pending": checks["sensitive_cache_private_data"]["status"] != "BLOCKED",
        "training_docs_match_runtime_and_workflow": checks["runtime_workflow_docs"]["status"] != "BLOCKED",
        "production_readiness_claim_supported_by_reports": checks["release_evidence"]["status"] != "BLOCKED",
        "remaining_risks_are_explicit": True,
    }
    final_note = build_final_note(checks["release_evidence"])
    return {
        "schema_version": "training-step-13-final-stability-v1",
        "phase_id": "training_step_13_final_stability",
        "generated_at": now_iso(),
        "todo_source": "training/TODOS.md#step-13-review-stabilitas-final",
        "checks": checks,
        "blocking_checks": blocking_checks,
        "warning_checks": warning_checks,
        "acceptance": acceptance,
        "final_stability_note": final_note,
        "final_decision": "blocked" if blocking_checks else "pass-with-documented-limitations" if warning_checks else "pass",
    }


def write_markdown(report: dict[str, Any]) -> None:
    note = report["final_stability_note"]
    lines = [
        "# Training Step 13 Final Stability",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Final decision: **{report['final_decision']}**",
        "",
        "## Final Stability Note",
        "",
        f"- Runtime: Python `{note['runtime']['target_python']}`; verifier ran on `{note['runtime']['current_verifier_python']}`; TensorFlow `{note['runtime']['tensorflow']}`; Keras `{note['runtime']['keras']}`.",
        f"- Phase 25 rerun evidence: `{note['notebook_rerun_at']}`.",
        f"- Artifact export folder: `{note['artifact_export_folder']}`.",
        f"- Phase 25 status: **{note['phase25_status']}**.",
        f"- Training release gate status: **{note['release_gate_status']}**.",
        f"- Model API smoke status: **{note['model_api_smoke_status']}**.",
        "",
        "## Acceptance",
        "",
    ]
    for name, passed in report["acceptance"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(["", "## Remaining Limitations", ""])
    lines.extend(f"- {item}" for item in note["limitations"])
    lines.extend(["", "## Checks", ""])
    for check_id, check in report["checks"].items():
        lines.extend([f"### {check_id}", "", f"Status: **{check['status']}**", ""])
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
    parser = argparse.ArgumentParser(description="Verify training TODO stabilization Step 13 final stability.")
    parser.add_argument("--write", action="store_true", help="Write JSON and Markdown final stability reports.")
    args = parser.parse_args()

    os.chdir(ROOT)
    report = build_report()
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
