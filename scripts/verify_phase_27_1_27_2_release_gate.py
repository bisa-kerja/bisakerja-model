#!/usr/bin/env python3
"""Verify release baseline and TensorBoard evidence gates for production readiness.

This script covers release gates for:
- Step 27.1: clean git baseline is mandatory for production status.
- Step 27.2: TensorBoard event evidence must live under a release-visible, non-ignored path
  and be recorded in the Phase 25 artifact manifest with SHA-256 and byte size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE25_ROOT = ROOT / "artifacts/phase_25_tensorflow_training_delivery"
PHASE25_MANIFEST_PATH = PHASE25_ROOT / "artifact_manifest.json"
TENSORBOARD_MONITORING_MANIFEST_PATH = PHASE25_ROOT / "tensorboard_monitoring_manifest.json"
TENSORBOARD_RELEASE_DIR = PHASE25_ROOT / "tensorboard_release"
TENSORBOARD_RELEASE_MANIFEST_PATH = TENSORBOARD_RELEASE_DIR / "manifest.json"
REPORT_JSON_PATH = ROOT / "reports/phase_27_1_27_2_release_gate.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_1_27_2_release_gate.md"

ARTIFACT_ID_TENSORBOARD_RELEASE_MANIFEST = "phase25_tensorboard_release_manifest"
ARTIFACT_ID_TENSORBOARD_EVENT_FILE = "phase25_tensorboard_event_file"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def run_git(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return raw


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_dirty_paths() -> list[str]:
    result = run_git(["status", "--porcelain"])
    return [line for line in result.stdout.splitlines() if line.strip()]


def git_commit() -> str | None:
    result = run_git(["rev-parse", "--short", "HEAD"])
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_git_ignored(path: Path) -> bool:
    result = run_git(["check-ignore", "-q", rel(path)])
    return result.returncode == 0


def source_tensorboard_event() -> dict[str, Any]:
    manifest = load_json(TENSORBOARD_MONITORING_MANIFEST_PATH)
    events = manifest.get("tensorboard", {}).get("event_files", [])
    if not events:
        raise RuntimeError("tensorboard_monitoring_manifest.json has no event_files")
    event = events[0]
    expected_name = Path(str(event["path"])).name
    expected_sha = str(event["sha256"])
    expected_size = int(event["bytes"])

    candidates = [ROOT / str(event["path"])]
    candidates.extend((ROOT / "artifacts/tensorboard").glob(f"**/{expected_name}"))
    candidates.extend(TENSORBOARD_RELEASE_DIR.glob(f"**/{expected_name}"))

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.stat().st_size != expected_size:
            continue
        if sha256_file(candidate) != expected_sha:
            continue
        return {
            "source_path": candidate,
            "expected_name": expected_name,
            "expected_sha256": expected_sha,
            "expected_size_bytes": expected_size,
            "monitoring_manifest": manifest,
        }

    raise FileNotFoundError(
        "Matching TensorBoard event file not found for "
        f"name={expected_name!r}, sha256={expected_sha}, size={expected_size}"
    )


def ensure_tensorboard_release_evidence() -> dict[str, Any]:
    source = source_tensorboard_event()
    source_path = Path(source["source_path"])
    run_id = str(source["monitoring_manifest"].get("tensorboard", {}).get("run_id", source_path.parent.name))
    release_event_path = TENSORBOARD_RELEASE_DIR / run_id / source["expected_name"]
    release_event_path.parent.mkdir(parents=True, exist_ok=True)
    if not release_event_path.exists() or sha256_file(release_event_path) != source["expected_sha256"]:
        shutil.copy2(source_path, release_event_path)

    event_record = file_record(release_event_path)
    if event_record["sha256"] != source["expected_sha256"] or event_record["size_bytes"] != source["expected_size_bytes"]:
        raise RuntimeError("Release TensorBoard event hash/size does not match monitoring manifest")

    release_manifest = {
        "schema_version": "phase-27-tensorboard-release-manifest-v1",
        "phase_id": "phase_27_production_readiness_release_gate",
        "generated_at": now_utc(),
        "requirement_reference": "REQUIREMENT.md#1.4",
        "source_monitoring_manifest": file_record(TENSORBOARD_MONITORING_MANIFEST_PATH),
        "source_ignored_log_path": rel(source_path),
        "release_log_root": rel(TENSORBOARD_RELEASE_DIR),
        "release_git_ignore_check": {
            "release_event_file_ignored": is_git_ignored(release_event_path),
            "release_manifest_ignored": is_git_ignored(TENSORBOARD_RELEASE_MANIFEST_PATH),
            "production_rule": "Release TensorBoard evidence must be committed from non-ignored paths before production-ready status."
        },
        "event_files": [event_record],
        "event_file_count": 1,
        "total_event_bytes": event_record["size_bytes"],
        "status": "complete" if not is_git_ignored(release_event_path) else "blocked",
    }
    write_json(TENSORBOARD_RELEASE_MANIFEST_PATH, release_manifest)
    return {
        "release_manifest": load_json(TENSORBOARD_RELEASE_MANIFEST_PATH),
        "release_manifest_record": file_record(TENSORBOARD_RELEASE_MANIFEST_PATH),
        "release_event_record": event_record,
    }


def artifact_manifest_entry(artifact_id: str, path: str, sha256: str, size_bytes: int, *, role: str, schema_version: str | None) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "consumer": "Phase 27 production release gate, reproducibility review, REQUIREMENT.md section 1.4 verification",
        "format": Path(path).suffix.lstrip(".") or "binary",
        "path": path,
        "producer": "scripts/verify_phase_27_1_27_2_release_gate.py --write",
        "reproducibility_references": {
            "requirement": "REQUIREMENT.md#1.4",
            "source_tensorboard_manifest": "artifacts/phase_25_tensorflow_training_delivery/tensorboard_monitoring_manifest.json",
        },
        "required_for_inference": False,
        "role": role,
        "schema_version": schema_version,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def update_artifact_manifest(release: dict[str, Any]) -> None:
    manifest = load_json(PHASE25_MANIFEST_PATH)
    artifacts = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("artifact_id") not in {ARTIFACT_ID_TENSORBOARD_RELEASE_MANIFEST, ARTIFACT_ID_TENSORBOARD_EVENT_FILE}
    ]
    release_manifest_record = release["release_manifest_record"]
    release_event_record = release["release_event_record"]
    artifacts.extend(
        [
            artifact_manifest_entry(
                ARTIFACT_ID_TENSORBOARD_RELEASE_MANIFEST,
                release_manifest_record["path"],
                release_manifest_record["sha256"],
                release_manifest_record["size_bytes"],
                role="tensorboard_release_manifest",
                schema_version="phase-27-tensorboard-release-manifest-v1",
            ),
            artifact_manifest_entry(
                ARTIFACT_ID_TENSORBOARD_EVENT_FILE,
                release_event_record["path"],
                release_event_record["sha256"],
                release_event_record["size_bytes"],
                role="tensorboard_event_release_evidence",
                schema_version=None,
            ),
        ]
    )
    manifest["artifacts"] = artifacts
    manifest["generated_at"] = now_utc()
    manifest["phase_27_release_gate_update"] = {
        "steps": ["27.1", "27.2"],
        "tensorboard_release_manifest": release_manifest_record["path"],
        "tensorboard_event_file": release_event_record["path"],
        "git_dirty_must_be_false_for_production": True,
        "tensorboard_release_evidence_required_for_production": True,
    }
    write_json(PHASE25_MANIFEST_PATH, manifest)


def build_report(release: dict[str, Any] | None = None) -> dict[str, Any]:
    dirty = git_dirty_paths()
    commit = git_commit()
    release_manifest_exists = TENSORBOARD_RELEASE_MANIFEST_PATH.exists()
    if release is None and release_manifest_exists:
        release_manifest = load_json(TENSORBOARD_RELEASE_MANIFEST_PATH)
        event_files = release_manifest.get("event_files", [])
        release = {
            "release_manifest": release_manifest,
            "release_manifest_record": file_record(TENSORBOARD_RELEASE_MANIFEST_PATH),
            "release_event_record": event_files[0] if event_files else None,
        }

    tensorboard_errors: list[str] = []
    tensorboard_status = "PASS"
    if release is None:
        tensorboard_status = "FAIL"
        tensorboard_errors.append("TensorBoard release manifest is missing")
    else:
        event_record = release.get("release_event_record")
        if not event_record:
            tensorboard_status = "FAIL"
            tensorboard_errors.append("TensorBoard release event record is missing")
        else:
            event_path = ROOT / str(event_record["path"])
            if not event_path.exists():
                tensorboard_status = "FAIL"
                tensorboard_errors.append(f"TensorBoard release event file missing: {event_record['path']}")
            else:
                actual = file_record(event_path)
                if actual["sha256"] != event_record["sha256"]:
                    tensorboard_status = "FAIL"
                    tensorboard_errors.append(f"TensorBoard release event sha256 mismatch: {event_record['path']}")
                if actual["size_bytes"] != event_record["size_bytes"]:
                    tensorboard_status = "FAIL"
                    tensorboard_errors.append(f"TensorBoard release event size mismatch: {event_record['path']}")
                if is_git_ignored(event_path):
                    tensorboard_status = "FAIL"
                    tensorboard_errors.append(f"TensorBoard release event path is git-ignored: {event_record['path']}")
        if release_manifest_exists and is_git_ignored(TENSORBOARD_RELEASE_MANIFEST_PATH):
            tensorboard_status = "FAIL"
            tensorboard_errors.append(f"TensorBoard release manifest path is git-ignored: {rel(TENSORBOARD_RELEASE_MANIFEST_PATH)}")

    baseline_status = "PASS" if not dirty else "FAIL"
    final_decision = "production-ready" if baseline_status == "PASS" and tensorboard_status == "PASS" else "blocked"
    return {
        "schema_version": "phase-27-1-27-2-release-gate-v1",
        "phase_id": "phase_27_production_readiness_release_gate",
        "generated_at": now_utc(),
        "git_state": {
            "commit": commit,
            "dirty": bool(dirty),
            "dirty_path_count": len(dirty),
            "dirty_paths": dirty,
            "production_rule": "git_dirty_at_setup=false and git_dirty_at_export=false are mandatory before production-ready status.",
        },
        "steps": {
            "27.1": {
                "name": "Freeze a clean release baseline",
                "status": baseline_status,
                "implemented_gate": True,
                "required_next_action": "Commit or intentionally remove every dirty path, rerun final export from clean worktree, and record clean git commit." if dirty else None,
            },
            "27.2": {
                "name": "Make TensorBoard evidence release-visible",
                "status": tensorboard_status,
                "release_manifest": rel(TENSORBOARD_RELEASE_MANIFEST_PATH) if release_manifest_exists else None,
                "release_event_file": None if not release or not release.get("release_event_record") else release["release_event_record"]["path"],
                "errors": tensorboard_errors,
                "requirement_reference": "REQUIREMENT.md#1.4",
            },
        },
        "artifact_manifest": {
            "path": rel(PHASE25_MANIFEST_PATH),
            "tensorboard_release_manifest_artifact_id": ARTIFACT_ID_TENSORBOARD_RELEASE_MANIFEST,
            "tensorboard_event_file_artifact_id": ARTIFACT_ID_TENSORBOARD_EVENT_FILE,
            "recorded_in_manifest": _artifact_ids_present(),
        },
        "final_decision": final_decision,
    }


def _artifact_ids_present() -> bool:
    if not PHASE25_MANIFEST_PATH.exists():
        return False
    manifest = load_json(PHASE25_MANIFEST_PATH)
    ids = {item.get("artifact_id") for item in manifest.get("artifacts", [])}
    return {ARTIFACT_ID_TENSORBOARD_RELEASE_MANIFEST, ARTIFACT_ID_TENSORBOARD_EVENT_FILE}.issubset(ids)


def write_markdown_report(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 27.1-27.2 Release Gate",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        "## Gates",
        "",
        f"- Step 27.1 clean baseline: `{report['steps']['27.1']['status']}`",
        f"- Step 27.2 TensorBoard release evidence: `{report['steps']['27.2']['status']}`",
        "",
        "## Evidence",
        "",
        f"- Git commit: `{report['git_state']['commit']}`",
        f"- Dirty path count: `{report['git_state']['dirty_path_count']}`",
        f"- TensorBoard release manifest: `{report['steps']['27.2']['release_manifest']}`",
        f"- TensorBoard event file: `{report['steps']['27.2']['release_event_file']}`",
        f"- Artifact manifest: `{report['artifact_manifest']['path']}`",
        "",
        "## Production rule",
        "",
        "Production-ready status requires clean setup/export git state and TensorBoard event evidence committed from non-ignored release paths.",
        "",
    ]
    if report["git_state"]["dirty_paths"]:
        lines.extend(["## Dirty paths", ""])
        lines.extend(f"- `{path}`" for path in report["git_state"]["dirty_paths"])
        lines.append("")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="copy TensorBoard evidence and write reports/manifests")
    args = parser.parse_args(argv)

    release = None
    if args.write:
        release = ensure_tensorboard_release_evidence()
        update_artifact_manifest(release)

    report = build_report(release)
    if args.write:
        write_json(REPORT_JSON_PATH, report)
        write_markdown_report(report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if report["final_decision"] == "production-ready" else 1


if __name__ == "__main__":
    sys.exit(main())
