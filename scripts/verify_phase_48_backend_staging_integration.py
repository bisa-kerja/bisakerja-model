#!/usr/bin/env python3
"""Verify Phase 48 staging integration, shadow comparison, and rollback evidence."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_48_backend_staging_integration.json"
REPORT_MD_PATH = ROOT / "reports/phase_48_backend_staging_integration.md"
RUNBOOK_PATH = ROOT / "docs/runbooks/backend-staging-shadow-rollback.md"
SHADOW_SCRIPT_PATH = ROOT / "scripts/run_phase_48_backend_staging_shadow.py"
PHASE47_REPORT_PATH = ROOT / "reports/phase_47_model_api_runtime_support.json"
PHASE46_REPORT_PATH = ROOT / "reports/phase_46_calibration_model_card_manifest_handoff_refresh.json"
PHASE43_DECISION_PATH = ROOT / "artifacts/phase_43_multilingual_e5_small_migration/migration_decision_risk_register.json"
PHASE25_MODEL_CARD_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/model_card.json"
PHASE46_MODEL_CARD_PATH = ROOT / "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/model_card.json"
PHASE25_MANIFEST_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
PHASE46_MANIFEST_PATH = ROOT / "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/artifact_manifest.json"
PHASE25_ROOT = "artifacts/phase_25_tensorflow_training_delivery"
PHASE46_ROOT = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
PHASE25_EMBEDDING = "intfloat/e5-base-v2"
PHASE46_EMBEDDING = "intfloat/multilingual-e5-small"
SCHEMA_VERSION = "phase-48-backend-staging-integration-v1"

SOURCE_PATHS = (
    RUNBOOK_PATH,
    SHADOW_SCRIPT_PATH,
    PHASE47_REPORT_PATH,
    PHASE46_REPORT_PATH,
    PHASE43_DECISION_PATH,
    PHASE25_MODEL_CARD_PATH,
    PHASE46_MODEL_CARD_PATH,
    PHASE25_MANIFEST_PATH,
    PHASE46_MANIFEST_PATH,
    ROOT / "GAP_MODEL_TRAINING.md",
    ROOT / "REQUIREMENT.md",
    ROOT / "TODOS.md",
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


def embedding_from_model_card(payload: dict[str, Any]) -> str | None:
    for path in (
        ("data", "embedding_contract", "embedding_model"),
        ("embedding_contract", "embedding_model"),
        ("model", "embedding_model"),
    ):
        current: Any = payload
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, str):
            return current
    return None


def build_report() -> dict[str, Any]:
    text = file_text(SOURCE_PATHS)
    phase47 = read_json(PHASE47_REPORT_PATH)
    phase46 = read_json(PHASE46_REPORT_PATH)
    phase43 = read_json(PHASE43_DECISION_PATH)
    phase25_card = read_json(PHASE25_MODEL_CARD_PATH)
    phase46_card = read_json(PHASE46_MODEL_CARD_PATH)
    checks = {
        "phase47_runtime_support_complete": phase47.get("status") == "complete"
        and phase47.get("checks", {}).get("phase46ArtifactVerifies") is True
        and phase47.get("checks", {}).get("phase25RollbackArtifactVerifies") is True,
        "phase46_artifact_selected_for_staging": phase46.get("status") == "complete"
        and embedding_from_model_card(phase46_card) == PHASE46_EMBEDDING,
        "phase25_rollback_artifact_preserved": embedding_from_model_card(phase25_card) == PHASE25_EMBEDDING
        and PHASE25_ROOT in text
        and PHASE25_EMBEDDING in text,
        "staging_deploy_env_documented": all(
            token in text
            for token in [
                "MODEL_API_ARTIFACT_ROOT=artifacts/phase_46_calibration_model_card_manifest_handoff_refresh",
                "MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/multilingual-e5-small",
                "SENTENCE_TRANSFORMERS_HOME",
                "docker compose",
                "persistent cache",
            ]
        ),
        "readiness_metadata_gate_documented": all(
            token in text
            for token in [
                "/live",
                "/health",
                "/ready",
                "/model-info",
                "embeddingModel",
                "artifactPhase",
                "artifactHash",
                "intfloat/multilingual-e5-small",
            ]
        ),
        "warmup_and_direct_model_smoke_supported": all(
            token in text
            for token in [
                "run_phase_48_backend_staging_shadow.py",
                "/internal/model/cv-analysis",
                "x-model-api-include-observability",
                "parseLatencyMs",
                "embeddingLatencyMs",
                "tensorflowLatencyMs",
                "totalLatencyMs",
            ]
        ),
        "backend_public_smoke_supported": all(
            token in text
            for token in [
                "/api/v1/ai/cv-analyzer",
                "USER_ACCESS_TOKEN",
                "persistResult=false",
                "JOB_SEARCH",
                "DIRECT_JOB_DETAIL",
                "cv-analysis-v2",
                "jobRecommendations",
                "no private field leakage",
            ]
        ),
        "shadow_compare_and_allowed_deltas_defined": all(
            token in text
            for token in [
                "compare_shadow_outputs",
                "score_delta_threshold",
                "score delta > 5 points",
                "top recommendation rank swap",
                "matchLevel",
                "matchedSkills",
                "missingSkills",
                "ATS stability",
                "high-fit recall",
            ]
        )
        and "ranking" in phase43.get("go_no_go_thresholds", {}).get("quality", {}),
        "failure_behavior_matrix_defined": all(
            token in text
            for token in [
                "MODEL_NOT_READY",
                "timeout",
                "invalid PDF",
                "empty candidates",
                "invalid token",
                "rollback artifact path",
            ]
        ),
        "rollback_commands_frozen": all(
            token in text
            for token in [
                "MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery",
                "MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2",
                "curl -fsS http://127.0.0.1:3004/ready",
                "Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}",
            ]
        ),
        "secrets_redacted_in_evidence": all(
            token in text
            for token in [
                "<redacted>",
                "Do not write service tokens",
                "DATABASE_URL",
                "OPENROUTER_API_KEY",
            ]
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "phase_48_backend_staging_integration_shadow_rollback",
        "generated_at": now_utc(),
        "status": "ready_for_staging_execution" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "operator_execution_required": True,
        "live_staging_evidence_required": [
            "Deploy staging Model API revision with Phase 46 artifact env.",
            "Run readiness probes and record /model-info metadata.",
            "Run direct Model API multipart smoke.",
            "Run Backend public AI CV Analyzer smoke with non-production token.",
            "Run old-vs-new shadow comparison over frozen fixtures.",
            "Run rollback drill to Phase 25 E5-base artifact path.",
        ],
        "commands": {
            "phase48_gate": "python scripts/verify_phase_48_backend_staging_integration.py --write",
            "phase48_new_model_api_smoke": "python scripts/run_phase_48_backend_staging_shadow.py --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --output reports/phase_48_backend_staging_shadow_report.json",
            "phase48_backend_public_smoke": "python scripts/run_phase_48_backend_staging_shadow.py --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --backend-api-url ${BACKEND_API_URL} --user-access-token ${USER_ACCESS_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --output reports/phase_48_backend_staging_shadow_report.json",
            "phase48_shadow_compare": "python scripts/run_phase_48_backend_staging_shadow.py --old-model-api-url ${OLD_MODEL_API_URL} --new-model-api-url ${MODEL_API_URL} --model-api-token ${MODEL_API_SERVICE_TOKEN} --score-delta-threshold 5 --output reports/phase_48_backend_staging_shadow_report.json",
            "rollback_to_phase25": "MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2 docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api",
        },
        "artifact_switch": {
            "new": {"root": PHASE46_ROOT, "embedding_model": PHASE46_EMBEDDING},
            "rollback": {"root": PHASE25_ROOT, "embedding_model": PHASE25_EMBEDDING},
        },
        "sources": [rel(path) for path in SOURCE_PATHS if path.exists()],
        "git": git_state(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 48 Backend/Staging Integration, Shadow Comparison, and Rollback Plan",
        "",
        f"Status: `{report['status']}`",
        f"Generated at: {report['generated_at']}",
        "",
        "Repo-side staging harness is ready. Live deployment, Backend token smoke, old-vs-new shadow run, and rollback drill still require operator credentials and approval.",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- `{item}`" for item in report["blockers"])
    lines.extend(["", "## Required Live Evidence"])
    lines.extend(f"- {item}" for item in report["live_staging_evidence_required"])
    lines.extend(["", "## Commands"])
    for name, command in report["commands"].items():
        lines.append(f"- `{name}`: `{command}`")
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
    return 0 if report["status"] == "ready_for_staging_execution" else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
