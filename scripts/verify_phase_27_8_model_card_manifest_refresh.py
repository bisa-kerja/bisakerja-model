#!/usr/bin/env python3
"""Refresh Phase 27.8 model-card and artifact-manifest release hashes."""

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
ARTIFACT_ROOT = ROOT / "artifacts/phase_25_tensorflow_training_delivery"
MODEL_CARD_PATH = ARTIFACT_ROOT / "model_card.json"
ARTIFACT_MANIFEST_PATH = ARTIFACT_ROOT / "artifact_manifest.json"
REPORT_JSON_PATH = ROOT / "reports/phase_27_8_model_card_manifest_refresh.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_8_model_card_manifest_refresh.md"

REQUIRED_ARTIFACTS = {
    "dataset_manifest": ARTIFACT_ROOT / "dataset_manifest.json",
    "label_manifest": ARTIFACT_ROOT / "label_manifest.json",
    "feature_config": ARTIFACT_ROOT / "feature_config.json",
    "source_tensorflow_feature_config": ARTIFACT_ROOT / "tensorflow_feature_config.json",
    "score_calibration": ARTIFACT_ROOT / "score_calibration.json",
    "final_keras_model": ARTIFACT_ROOT / "export/selected_jobfit_tf_phase25.keras",
    "phase25_tensorboard_release_manifest": ARTIFACT_ROOT / "tensorboard_release/manifest.json",
}

SCORE_SEMANTICS = {
    "jobFitAlignment.score": {
        "scale": "0-100 integer",
        "bands": {
            "0-20": "very weak evidence of fit",
            "21-40": "weak fit; major gaps likely",
            "41-60": "partial fit; review gaps before action",
            "61-80": "good fit; candidate evidence aligns with role",
            "81-100": "strong fit; high overlap with supplied role/candidate evidence",
        },
        "production_claim_policy": "Requires frozen human/reviewer validation coverage; weak labels are bootstrap/training support only.",
    },
    "atsFriendliness.score": {
        "scale": "0-100 integer",
        "bands": {
            "0-20": "parse failure or severe ATS risk",
            "21-40": "many structural risks",
            "41-60": "mixed parse/section evidence",
            "61-80": "usable CV with some issues",
            "81-100": "clear parse and low formatting risk",
        },
    },
    "recommendations[].matchScore": {
        "scale": "0-100 integer",
        "bands": {
            "0-60": "stretch",
            "61-80": "good",
            "81-100": "strong",
        },
        "candidate_policy": "Only rank backend-provided candidate IDs; never invent or hydrate jobs.",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def dirty_paths() -> list[str]:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return ["<git status failed>"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def tensorboard_event_records() -> list[dict[str, Any]]:
    release_dir = ARTIFACT_ROOT / "tensorboard_release"
    if not release_dir.exists():
        return []
    return [file_record(path) for path in sorted(release_dir.rglob("events.out.tfevents*")) if path.is_file()]


def split_seed_from_manifest(manifest: dict[str, Any]) -> int | None:
    for entry in manifest.get("artifacts", []):
        refs = entry.get("reproducibility_references") if isinstance(entry, dict) else None
        if isinstance(refs, dict) and isinstance(refs.get("split_seed"), int):
            return refs["split_seed"]
    return None


def build_release_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    missing: list[str] = []
    for artifact_id, path in REQUIRED_ARTIFACTS.items():
        if path.exists():
            artifacts[artifact_id] = file_record(path)
        else:
            missing.append(rel(path))
    events = tensorboard_event_records()
    if events:
        artifacts["tensorboard_event_files"] = events
    else:
        missing.append("artifacts/phase_25_tensorflow_training_delivery/tensorboard_release/events.out.tfevents*")
    return {
        "git_commit": git_commit(),
        "git_dirty_file_count": len(dirty_paths()),
        "split_seed": split_seed_from_manifest(manifest),
        "artifacts": artifacts,
        "missing": sorted(set(missing)),
    }


def update_model_card(model_card: dict[str, Any], snapshot: dict[str, Any], generated_at: str) -> dict[str, Any]:
    updated = dict(model_card)
    artifacts = snapshot["artifacts"]

    updated["generated_at"] = generated_at
    updated["score_semantics"] = SCORE_SEMANTICS
    updated["intended_use"] = [
        "Model-core job-fit scoring for backend-provided CV/profile and job context.",
        "Training-owned calibrated scores and artifact handoff for separate Model API runtime.",
        "Candidate reranking for backend-provided job IDs only; backend owns hydration and public prose.",
    ]
    updated["blocked_use"] = [
        "Automated hiring decision, rejection, eligibility, salary, or protected-class inference.",
        "Production score claims without frozen human/reviewer validation coverage.",
        "Backend-owned persistence, auth, DB hydration, or GenAI wrapper output generation.",
        "Scoring candidate jobs not supplied by the backend request.",
    ]
    updated["limitations"] = [
        "Weak labels remain bootstrap/training support only; production score claims require human/reviewer validation coverage.",
        "Current release remains blocked until clean-git export, real Python 3.13 Model API smoke, requirement matrix, and final Phase 27 release gate pass.",
        "ATS and recommendation readiness depend on release-scale validation fixtures or approved anonymized real data.",
        "External GenAI prose is wrapper-owned and is not part of model-core inference.",
    ]
    updated.setdefault("readiness", {})
    updated["readiness"].update(
        {
            "phase_27_8_refresh_status": "hashes-refreshed-production-claim-blocked-until-all-phase27-gates-pass",
            "production_ready_requires_clean_git_state": True,
            "production_ready_requires_real_model_api_smoke": True,
            "production_ready_requires_requirement_matrix": True,
        }
    )
    updated["phase_27_8_release_refresh"] = {
        "generated_at": generated_at,
        "git_commit": snapshot["git_commit"],
        "git_dirty_file_count": snapshot["git_dirty_file_count"],
        "split_seed": snapshot["split_seed"],
        "dataset_hash": artifacts.get("dataset_manifest", {}).get("sha256"),
        "label_hash": artifacts.get("label_manifest", {}).get("sha256"),
        "feature_config_hash": artifacts.get("feature_config", {}).get("sha256"),
        "tensorflow_feature_config_hash": artifacts.get("source_tensorflow_feature_config", {}).get("sha256"),
        "tensorboard_release_manifest_hash": artifacts.get("phase25_tensorboard_release_manifest", {}).get("sha256"),
        "tensorboard_event_hashes": [item["sha256"] for item in artifacts.get("tensorboard_event_files", [])],
        "model_hash": artifacts.get("final_keras_model", {}).get("sha256"),
        "calibration_hash": artifacts.get("score_calibration", {}).get("sha256"),
        "production_claim_allowed": False,
        "production_claim_blockers": [
            "clean-git Phase 25 production-ready export not proven",
            "real Python 3.13 Model API smoke not recorded",
            "end-to-end REQUIREMENT.md matrix not recorded",
            "final Phase 27 release gate not recorded",
        ],
    }

    data = dict(updated.get("data", {}))
    if "dataset_manifest" in artifacts:
        data["dataset_manifest"] = artifacts["dataset_manifest"]
    if "label_manifest" in artifacts:
        data["label_manifest"] = artifacts["label_manifest"]
    updated["data"] = data

    deployment = dict(updated.get("deployment_contract", {}))
    if "final_keras_model" in artifacts:
        deployment["final_model_artifact"] = {**deployment.get("final_model_artifact", {}), **artifacts["final_keras_model"], "format": ".keras"}
    updated["deployment_contract"] = deployment

    training = dict(updated.get("training", {}))
    tensorboard = dict(training.get("tensorboard", {}))
    tensorboard["release_manifest"] = artifacts.get("phase25_tensorboard_release_manifest")
    tensorboard["release_event_files"] = artifacts.get("tensorboard_event_files", [])
    training["tensorboard"] = tensorboard
    updated["training"] = training
    return updated


def update_manifest(manifest: dict[str, Any], snapshot: dict[str, Any], generated_at: str) -> dict[str, Any]:
    updated = dict(manifest)
    by_id = {entry.get("artifact_id"): entry for entry in updated.get("artifacts", []) if isinstance(entry, dict)}
    for artifact_id, record in snapshot["artifacts"].items():
        if artifact_id == "tensorboard_event_files":
            continue
        entry = by_id.get(artifact_id)
        if entry is not None and isinstance(record, dict):
            entry["sha256"] = record["sha256"]
            entry["size_bytes"] = record["size_bytes"]
    if MODEL_CARD_PATH.exists() and "model_card" in by_id:
        model_record = file_record(MODEL_CARD_PATH)
        by_id["model_card"].update({"sha256": model_record["sha256"], "size_bytes": model_record["size_bytes"]})
    updated["generated_at"] = generated_at
    updated["phase_27_8_release_refresh"] = {
        "generated_at": generated_at,
        "git_commit": snapshot["git_commit"],
        "git_dirty_file_count": snapshot["git_dirty_file_count"],
        "dataset_hash": snapshot["artifacts"].get("dataset_manifest", {}).get("sha256"),
        "label_hash": snapshot["artifacts"].get("label_manifest", {}).get("sha256"),
        "feature_config_hash": snapshot["artifacts"].get("feature_config", {}).get("sha256"),
        "tensorflow_feature_config_hash": snapshot["artifacts"].get("source_tensorflow_feature_config", {}).get("sha256"),
        "tensorboard_release_manifest_hash": snapshot["artifacts"].get("phase25_tensorboard_release_manifest", {}).get("sha256"),
        "model_hash": snapshot["artifacts"].get("final_keras_model", {}).get("sha256"),
        "calibration_hash": snapshot["artifacts"].get("score_calibration", {}).get("sha256"),
        "score_semantics_refreshed_in_model_card": True,
        "production_claim_allowed": False,
    }
    return updated


def gate(name: str, passed: bool, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"check": name, "status": "PASS" if passed else "FAIL"}
    if evidence is not None:
        payload["evidence"] = evidence
    return payload


def build_report(write: bool = False) -> dict[str, Any]:
    generated_at = now_iso()
    missing_inputs = [rel(path) for path in (MODEL_CARD_PATH, ARTIFACT_MANIFEST_PATH) if not path.exists()]
    model_card = load_json(MODEL_CARD_PATH) if MODEL_CARD_PATH.exists() else {}
    manifest = load_json(ARTIFACT_MANIFEST_PATH) if ARTIFACT_MANIFEST_PATH.exists() else {"artifacts": []}
    snapshot = build_release_snapshot(manifest)

    if write and not missing_inputs:
        write_json(MODEL_CARD_PATH, update_model_card(model_card, snapshot, generated_at))
        snapshot = build_release_snapshot(update_manifest(manifest, snapshot, generated_at))
        manifest = update_manifest(manifest, snapshot, generated_at)
        write_json(ARTIFACT_MANIFEST_PATH, manifest)
        model_card = load_json(MODEL_CARD_PATH)
        manifest = load_json(ARTIFACT_MANIFEST_PATH)

    refreshed = bool(model_card.get("phase_27_8_release_refresh"))
    manifest_refreshed = bool(manifest.get("phase_27_8_release_refresh"))
    gates = [
        gate("required_model_card_and_manifest_exist", not missing_inputs, {"missing": missing_inputs}),
        gate("required_release_hash_inputs_exist", not snapshot["missing"], {"missing": snapshot["missing"]}),
        gate("model_card_contains_phase_27_8_release_refresh", refreshed, model_card.get("phase_27_8_release_refresh", {})),
        gate("artifact_manifest_contains_phase_27_8_release_refresh", manifest_refreshed, manifest.get("phase_27_8_release_refresh", {})),
        gate("score_semantics_present", bool(model_card.get("score_semantics")), {"outputs": sorted(model_card.get("score_semantics", {}).keys())}),
        gate("production_claim_remains_blocked_until_all_phase27_gates", model_card.get("phase_27_8_release_refresh", {}).get("production_claim_allowed") is False),
    ]
    model_entry = next((entry for entry in manifest.get("artifacts", []) if entry.get("artifact_id") == "model_card"), None)
    if model_entry and MODEL_CARD_PATH.exists():
        gates.append(gate("manifest_model_card_hash_matches_current_file", model_entry.get("sha256") == sha256_file(MODEL_CARD_PATH), {"manifest_sha256": model_entry.get("sha256"), "current_sha256": sha256_file(MODEL_CARD_PATH)}))
    if ARTIFACT_MANIFEST_PATH.exists():
        snapshot["artifact_manifest"] = file_record(ARTIFACT_MANIFEST_PATH)
    blockers = [g["check"] for g in gates if g["status"] != "PASS"]
    return {
        "schema_version": "phase-27-8-model-card-manifest-refresh-v1",
        "phase_id": "phase_27_8_model_card_manifest_refresh",
        "generated_at": generated_at,
        "references": ["GAP_MODEL_TRAINING.md#step-27.8", "GAP_MODEL_TRAINING.md", "REQUIREMENT.md"],
        "release_snapshot": snapshot,
        "gates": gates,
        "blockers": blockers,
        "final_decision": "refresh-complete" if not blockers else "blocked",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 27.8 Model Card and Manifest Refresh",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Decision: `{report['final_decision']}`",
        "",
        "## Gates",
    ]
    for item in report["gates"]:
        lines.append(f"- {item['status']} `{item['check']}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(["", "## Policy", "Production score claims remain blocked until every Phase 27 release gate passes."])
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    report = build_report(write=True)
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh model_card/artifact_manifest and write report files")
    args = parser.parse_args(argv)
    report = write_all() if args.write else build_report(write=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "refresh-complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
