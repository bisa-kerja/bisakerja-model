#!/usr/bin/env python3
"""Verify Phase 49 staging promotion decision and production guardrails."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_49_staging_promotion_guardrails.json"
REPORT_MD_PATH = ROOT / "reports/phase_49_staging_promotion_guardrails.md"
GUARDRAILS_PATH = ROOT / "docs/runbooks/staging-promotion-production-guardrails.md"
PHASE43_REPORT_PATH = ROOT / "reports/phase_43_multilingual_e5_small_migration_decision.json"
PHASE44_REPORT_PATH = ROOT / "reports/phase_44_embedding_compatibility_audit.json"
PHASE45_REPORT_PATH = ROOT / "reports/phase_45_multilingual_e5_small_training_delivery.json"
PHASE46_REPORT_PATH = ROOT / "reports/phase_46_calibration_model_card_manifest_handoff_refresh.json"
PHASE47_REPORT_PATH = ROOT / "reports/phase_47_model_api_runtime_support.json"
PHASE48_REPORT_PATH = ROOT / "reports/phase_48_backend_staging_integration.json"
PHASE46_MODEL_CARD_PATH = ROOT / "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/model_card.json"
TODOS_PATH = ROOT / "TODOS.md"
PHASE46_ROOT = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
PHASE25_ROOT = "artifacts/phase_25_tensorflow_training_delivery"
PHASE46_EMBEDDING = "intfloat/multilingual-e5-small"
PHASE25_EMBEDDING = "intfloat/e5-base-v2"
DECISION = "staging-experiment-only"
SCHEMA_VERSION = "phase-49-staging-promotion-guardrails-v1"

SOURCE_PATHS = (
    GUARDRAILS_PATH,
    PHASE43_REPORT_PATH,
    PHASE44_REPORT_PATH,
    PHASE45_REPORT_PATH,
    PHASE46_REPORT_PATH,
    PHASE47_REPORT_PATH,
    PHASE48_REPORT_PATH,
    PHASE46_MODEL_CARD_PATH,
    TODOS_PATH,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(read(path) for path in paths)


def run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    return result.stdout.strip() or None


def git_state() -> dict[str, Any]:
    porcelain = run_text(["git", "status", "--porcelain"]) or ""
    dirty_lines = [line for line in porcelain.splitlines() if line]
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty_file_count": len(dirty_lines),
        "dirty_sample": dirty_lines[:40],
    }


def nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def all_tokens(text: str, tokens: list[str]) -> bool:
    return all(token in text for token in tokens)


def build_report() -> dict[str, Any]:
    text = file_text(SOURCE_PATHS)
    phase43 = read_json(PHASE43_REPORT_PATH)
    phase44 = read_json(PHASE44_REPORT_PATH)
    phase45 = read_json(PHASE45_REPORT_PATH)
    phase46 = read_json(PHASE46_REPORT_PATH)
    phase47 = read_json(PHASE47_REPORT_PATH)
    phase48 = read_json(PHASE48_REPORT_PATH)
    model_card = read_json(PHASE46_MODEL_CARD_PATH)
    promotion = model_card.get("promotion_decision", {}) if isinstance(model_card, dict) else {}
    phase48_live_items = phase48.get("live_staging_evidence_required", [])

    checks = {
        "phase43_to_48_evidence_compiled": (
            phase43.get("final_decision") == "approved_for_staging_experiment_only"
            and phase44.get("status") == "complete"
            and phase44.get("final_decision") == "retrain_required_recalibration_required_direct_swap_blocked"
            and phase45.get("status") == "complete"
            and phase45.get("selection_decision") == "select_for_staging_shadow_validation"
            and phase46.get("status") == "complete"
            and phase47.get("status") == "complete"
            and phase48.get("status") == "ready_for_staging_execution"
        ),
        "readiness_decision_is_experiment_only": (
            promotion.get("decision") == DECISION
            and promotion.get("production_claim_allowed") is False
            and promotion.get("default_artifact_root") == PHASE46_ROOT
            and promotion.get("rollback_artifact_root") == PHASE25_ROOT
            and all_tokens(text, [DECISION, "live staging smoke", "rollback drill", "operator credentials"])
        ),
        "deployment_docs_cover_default_and_rollback_env": all_tokens(
            text,
            [
                f"MODEL_API_ARTIFACT_ROOT={PHASE46_ROOT}",
                f"MODEL_API_EXPECTED_EMBEDDING_MODEL={PHASE46_EMBEDDING}",
                f"MODEL_API_ARTIFACT_ROOT={PHASE25_ROOT}",
                f"MODEL_API_EXPECTED_EMBEDDING_MODEL={PHASE25_EMBEDDING}",
                "docker compose",
                "Nginx checks",
                "Cloud Run checks",
                "Hugging Face staging caveats",
            ],
        ),
        "monitoring_checklist_complete": all_tokens(
            text,
            [
                "timeout rate",
                "MODEL_NOT_READY",
                "inference latency",
                "memory RSS",
                "CPU saturation",
                "score distribution drift",
                "recommendation count",
                "backend downstream errors",
            ],
        ),
        "production_blockers_are_explicit": all_tokens(
            text,
            [
                "human/reviewer validation",
                "slice coverage",
                "calibration confidence",
                "privacy review",
                "cost/resource monitoring",
                "rollback drill",
                "public response contract safety",
            ],
        ),
        "suggested_execution_order_blocks_direct_constant_changes": all_tokens(
            read(TODOS_PATH),
            [
                "Complete Phase 49 before broader staging/demo or production claims",
                "Do not change runtime constants or default env examples directly",
                "MODEL_API_ARTIFACT_ROOT",
                "MODEL_API_EXPECTED_EMBEDDING_MODEL",
            ],
        ),
        "rejected_artifact_policy_documented": all_tokens(
            text,
            [
                "If multilingual-E5-small is later rejected",
                "Keep reports and artifacts for audit history",
                "Do not reference Phase 46 artifact paths in default env examples",
                "Mark the model card and promotion report as `rejected`",
            ],
        ),
        "production_remains_blocked_by_live_evidence": (
            phase48.get("operator_execution_required") is True
            and len(phase48_live_items) >= 6
            and promotion.get("production_claim_allowed") is False
            and "live staging Backend public AI CV Analyzer smoke" in text
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_49_staging_promotion_decision_production_guardrails",
        "generated_at": now_utc(),
        "status": "complete" if not blockers else "blocked",
        "decision": DECISION,
        "decision_reasons": [
            "Phase 46 multilingual-E5-small artifact is complete and runtime-verifiable.",
            "Phase 48 live staging smoke, shadow evidence, and rollback drill are still operator-executed gates.",
            "Production claims remain blocked until human/reviewer validation, calibration, contract, runtime, monitoring, privacy, and rollback gates pass.",
        ],
        "checks": checks,
        "blockers": blockers,
        "deployment": {
            "default_staging": {"root": PHASE46_ROOT, "embedding_model": PHASE46_EMBEDDING},
            "rollback": {"root": PHASE25_ROOT, "embedding_model": PHASE25_EMBEDDING},
        },
        "monitoring_required": [
            "timeout rate",
            "MODEL_NOT_READY",
            "inference latency",
            "memory RSS",
            "CPU saturation",
            "score distribution drift",
            "recommendation count",
            "backend downstream errors",
        ],
        "production_blockers": [
            "live staging Backend public AI CV Analyzer smoke",
            "old-vs-new shadow comparison",
            "rollback drill",
            "human/reviewer validation scale",
            "slice coverage",
            "calibration confidence",
            "privacy review",
            "cost/resource monitoring",
        ],
        "operator_execution_required": True,
        "sources": [rel(path) for path in SOURCE_PATHS if path.exists()],
        "git": git_state(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 49 Staging Promotion Decision and Production Guardrails",
        "",
        f"Status: `{report['status']}`",
        f"Decision: `{report['decision']}`",
        f"Generated at: {report['generated_at']}",
        "",
        "multilingual-E5-small remains staging experiment only. Production rollout stays blocked until live staging, validation, monitoring, privacy, contract, runtime, and rollback gates pass.",
        "",
        "## Decision Reasons",
    ]
    lines.extend(f"- {item}" for item in report["decision_reasons"])
    lines.extend(["", "## Checks"])
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{item}`" for item in report["blockers"])
    lines.extend(["", "## Staging Env"])
    lines.append(f"- Default: `{PHASE46_ROOT}` / `{PHASE46_EMBEDDING}`")
    lines.append(f"- Rollback: `{PHASE25_ROOT}` / `{PHASE25_EMBEDDING}`")
    lines.extend(["", "## Production Blockers"])
    lines.extend(f"- {item}" for item in report["production_blockers"])
    lines.extend(["", "## Monitoring"])
    lines.extend(f"- {item}" for item in report["monitoring_required"])
    return "\n".join(lines) + "\n"


def write_all() -> dict[str, Any]:
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    REPORT_MD_PATH.write_text(markdown_report(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    write = bool(argv and "--write" in argv)
    report = write_all() if write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
