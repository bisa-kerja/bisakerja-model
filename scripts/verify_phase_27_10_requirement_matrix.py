#!/usr/bin/env python3
"""Build Phase 27.10 end-to-end REQUIREMENT.md evidence matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_27_10_requirement_matrix.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_10_requirement_matrix.md"

REQUIREMENT_IDS = ("1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "3.1", "3.2", "4.1", "deliverables")

PHASE_27_1_REPORT = ROOT / "reports/phase_27_1_27_2_release_gate.json"
PHASE_27_7_REPORT = ROOT / "reports/phase_27_7_clean_kernel_production_export.json"
PHASE_27_8_REPORT = ROOT / "reports/phase_27_8_model_card_manifest_refresh.json"
PHASE_27_9_REPORT = ROOT / "reports/phase_27_9_model_api_production_smoke.json"
TENSORBOARD_RELEASE_MANIFEST = ROOT / "artifacts/phase_25_tensorflow_training_delivery/tensorboard_release/manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_tracked_paths() -> set[str]:
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=False, capture_output=True)
    if proc.returncode != 0:
        return set()
    return {item.decode("utf-8") for item in proc.stdout.split(b"\0") if item}


def evidence_record(path: str | Path, tracked_paths: set[str]) -> dict[str, Any]:
    candidate = ROOT / path if not isinstance(path, Path) else path
    record: dict[str, Any] = {"path": rel(candidate), "exists": candidate.exists(), "tracked": rel(candidate) in tracked_paths}
    if candidate.is_file():
        record.update({"sha256": sha256_file(candidate), "size_bytes": candidate.stat().st_size})
    return record


def gate_status(report: dict[str, Any], check: str) -> str | None:
    for gate in report.get("gates", []):
        if gate.get("check") == check:
            return gate.get("status")
    return None


def gate_evidence(report: dict[str, Any], check: str) -> dict[str, Any]:
    for gate in report.get("gates", []):
        if gate.get("check") == check:
            return gate.get("evidence", {})
    return {}


def tensorboard_evidence_paths() -> list[str]:
    paths = [rel(TENSORBOARD_RELEASE_MANIFEST)]
    manifest = load_json(TENSORBOARD_RELEASE_MANIFEST)
    for item in manifest.get("event_files", []):
        path = item.get("path")
        if path:
            paths.append(path)
    return paths


def row(
    requirement_id: str,
    requirement: str,
    evidence_paths: list[str],
    checks: dict[str, bool],
    tracked_paths: set[str],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    evidence = [evidence_record(path, tracked_paths) for path in evidence_paths]
    missing = [item["path"] for item in evidence if not item["exists"]]
    untracked = [item["path"] for item in evidence if item["exists"] and not item["tracked"]]
    failed_checks = [name for name, passed in checks.items() if not passed]
    if missing:
        status = "missing-evidence"
    elif failed_checks:
        status = "failed-gate"
    elif untracked:
        status = "tracking-blocked"
    else:
        status = "satisfied"
    return {
        "requirement_id": requirement_id,
        "requirement": requirement,
        "status": status,
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_evidence": missing,
        "untracked_evidence": untracked,
        "evidence": evidence,
        "notes": notes or [],
    }


def build_report() -> dict[str, Any]:
    tracked = git_tracked_paths()
    phase27_1 = load_json(PHASE_27_1_REPORT)
    phase27_7 = load_json(PHASE_27_7_REPORT)
    phase27_8 = load_json(PHASE_27_8_REPORT)
    phase27_9 = load_json(PHASE_27_9_REPORT)

    rows = [
        row(
            "1.1",
            "TensorFlow deep learning architecture using Functional API or Model Subclassing.",
            ["REQUIREMENT.md", "reports/phase_25_tensorflow_architecture.json", "training/notebooks/phase_25_tensorflow_training_delivery.ipynb", rel(PHASE_27_7_REPORT)],
            {"tensorflow_functional_api_gate_passed": gate_status(phase27_7, "tensorflow_functional_api") == "PASS"},
            tracked,
        ),
        row(
            "1.2",
            "At least one advanced custom component: custom layer, loss, or callback.",
            ["reports/phase_25_custom_component.json", "model_api/custom_objects.py", "artifacts/phase_25_tensorflow_training_delivery/export/registered_custom_objects_smoke.py", rel(PHASE_27_7_REPORT)],
            {"custom_components_present_gate_passed": gate_status(phase27_7, "custom_components_present") == "PASS"},
            tracked,
            ["Custom components include " + ", ".join(gate_evidence(phase27_7, "custom_components_present").get("custom_components", []))],
        ),
        row(
            "1.3",
            "Full training/evaluation loop using tf.GradientTape; model.fit() not used as main training path.",
            ["reports/phase_25_training_evaluation_loop.json", "training/notebooks/phase_25_tensorflow_training_delivery.ipynb", rel(PHASE_27_7_REPORT)],
            {"gradient_tape_loop_no_model_fit_gate_passed": gate_status(phase27_7, "gradient_tape_loop_no_model_fit") == "PASS"},
            tracked,
        ),
        row(
            "1.4",
            "TensorBoard monitoring logs are included in repository release evidence.",
            ["reports/phase_25_tensorboard_monitoring.json", rel(PHASE_27_1_REPORT), *tensorboard_evidence_paths()],
            {
                "phase_27_2_tensorboard_release_gate_passed": (phase27_1.get("steps", {}).get("27.2", {}).get("status") == "PASS"),
                "tensorboard_events_recorded_gate_passed": gate_status(phase27_7, "tensorboard_events_recorded") == "PASS",
            },
            tracked,
        ),
        row(
            "1.5",
            "Regression MAE target <= 0.02.",
            ["reports/phase_25_baseline_selection_gate.json", "artifacts/phase_25_tensorflow_training_delivery/model_card.json", rel(PHASE_27_7_REPORT)],
            {"mae_target_le_0_02_gate_passed": gate_status(phase27_7, "mae_target_le_0_02") == "PASS"},
            tracked,
            ["MAE evidence: " + json.dumps(gate_evidence(phase27_7, "mae_target_le_0_02"), sort_keys=True)],
        ),
        row(
            "2.1",
            "Export trained model in production TensorFlow format: .keras or SavedModel.",
            ["reports/phase_25_tensorflow_artifact_export.json", "artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras", "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json", rel(PHASE_27_8_REPORT)],
            {"keras_export_and_inference_smoke_gate_passed": gate_status(phase27_7, "keras_export_and_inference_smoke") == "PASS"},
            tracked,
        ),
        row(
            "2.2",
            "Inference code loads exported model and produces JSON-compatible prediction output.",
            ["model_api/inference.py", "artifacts/phase_25_tensorflow_training_delivery/export/inference_smoke_fixture.json", "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json", rel(PHASE_27_9_REPORT)],
            {
                "source_inference_fixture_present": (ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/inference_smoke_fixture.json").exists(),
                "real_keras_custom_object_loader_smoke_passed": gate_status(phase27_9, "real_keras_custom_object_loader_smoke") == "PASS",
            },
            tracked,
            ["Production runtime loader evidence remains tied to Step 27.9."],
        ),
        row(
            "3.1",
            "REST API implemented with FastAPI or Flask.",
            ["model_api/app.py", "model_api/README.md", "tests/model_api/test_phase_26_layout.py", rel(PHASE_27_9_REPORT)],
            {
                "fastapi_source_present": (ROOT / "model_api/app.py").exists(),
                "live_fastapi_smoke_passed": gate_status(phase27_9, "live_fastapi_health_model_info_inference_smoke") == "PASS",
            },
            tracked,
        ),
        row(
            "3.2",
            "REST API loads model, accepts input, runs inference, returns JSON.",
            ["model_api/app.py", "model_api/schemas.py", "model_api/inference.py", "tests/model_api/test_phase_26_layout.py", rel(PHASE_27_9_REPORT)],
            {
                "phase26_tests_unskipped_passed": gate_status(phase27_9, "phase26_tests_unskipped") == "PASS",
                "live_fastapi_health_model_info_inference_smoke_passed": gate_status(phase27_9, "live_fastapi_health_model_info_inference_smoke") == "PASS",
            },
            tracked,
            ["Fake model or fake embedding smoke is not accepted for production evidence."],
        ),
        row(
            "4.1",
            "Generative AI is secondary feature or wrapper boundary, not core training mutation.",
            ["reports/phase_25_training_only_genai_boundary.json", "artifacts/phase_25_tensorflow_training_delivery/training_only_genai_boundary.json", "artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json", "GAP_MODEL_TRAINING.md"],
            {"genai_boundary_artifacts_present": all((ROOT / path).exists() for path in ["reports/phase_25_training_only_genai_boundary.json", "artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json"])},
            tracked,
        ),
        row(
            "deliverables",
            "Repository contains training source, custom component, GradientTape loop, model export, inference code, REST API, GenAI boundary, TensorBoard logs, docs, requirements, README.",
            [
                "training/notebooks/phase_25_tensorflow_training_delivery.ipynb",
                "model_api/custom_objects.py",
                "reports/phase_25_training_evaluation_loop.json",
                "artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras",
                "model_api/inference.py",
                "model_api/app.py",
                "artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json",
                *tensorboard_evidence_paths(),
                "requirements.txt",
                "training/README.md",
                "model_api/README.md",
                "README.md",
            ],
            {
                "all_requirement_rows_present": True,
                "model_card_manifest_refresh_complete": phase27_8.get("final_decision") == "refresh-complete",
                "runtime_smoke_production_passed": phase27_9.get("final_decision") == "production-smoke-passed",
            },
            tracked,
            ["Runtime smoke and clean tracked release evidence are required before final production-ready claim."],
        ),
    ]

    blockers = []
    for item in rows:
        if item["status"] != "satisfied":
            blockers.append({"requirement_id": item["requirement_id"], "status": item["status"], "failed_checks": item["failed_checks"], "missing_evidence": item["missing_evidence"], "untracked_evidence": item["untracked_evidence"]})

    return {
        "schema_version": "phase-27-10-requirement-matrix-v1",
        "phase_id": "phase_27_10_requirement_matrix",
        "generated_at": now_iso(),
        "references": ["GAP_MODEL_TRAINING.md#step-27.10", "REQUIREMENT.md", "GAP_MODEL_TRAINING.md"],
        "policy": {
            "all_requirement_sections_required": list(REQUIREMENT_IDS),
            "evidence_must_exist": True,
            "evidence_must_be_git_tracked_for_production_claim": True,
            "real_runtime_smoke_required_for_api_sections": True,
        },
        "matrix": rows,
        "coverage": {
            "required_ids": list(REQUIREMENT_IDS),
            "covered_ids": [item["requirement_id"] for item in rows],
            "missing_ids": sorted(set(REQUIREMENT_IDS) - {item["requirement_id"] for item in rows}),
            "satisfied_count": sum(1 for item in rows if item["status"] == "satisfied"),
            "row_count": len(rows),
        },
        "blockers": blockers,
        "final_decision": "requirements-satisfied" if not blockers else "blocked",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 27.10 Requirement Matrix",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Decision: `{report['final_decision']}`",
        "",
        "## Matrix",
    ]
    for item in report["matrix"]:
        lines.append(f"- `{item['requirement_id']}` {item['status']} — {item['requirement']}")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        for item in report["blockers"]:
            lines.append(f"- `{item['requirement_id']}` {item['status']}")
            for check in item["failed_checks"]:
                lines.append(f"  - failed check: `{check}`")
            for path in item["missing_evidence"]:
                lines.append(f"  - missing: `{path}`")
            for path in item["untracked_evidence"][:8]:
                lines.append(f"  - untracked: `{path}`")
            if len(item["untracked_evidence"]) > 8:
                lines.append(f"  - untracked: ... {len(item['untracked_evidence']) - 8} more")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON/Markdown report")
    args = parser.parse_args(argv)
    report = write_all() if args.write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "requirements-satisfied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
