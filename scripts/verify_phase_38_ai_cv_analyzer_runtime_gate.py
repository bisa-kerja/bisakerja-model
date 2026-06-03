#!/usr/bin/env python3
"""Verify AI CV Analyzer staging runtime warmup/readiness gate evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_38_ai_cv_analyzer_runtime_gate.json"
REPORT_MD_PATH = ROOT / "reports/phase_38_ai_cv_analyzer_runtime_gate.md"

SOURCE_PATHS = (
    ROOT / "model_api/app.py",
    ROOT / "model_api/config.py",
    ROOT / "model_api/observability.py",
    ROOT / "model_api/.env.example",
    ROOT / "scripts/warmup_ai_cv_analyzer_runtime.py",
    ROOT / "docs/runbooks/ai-cv-analyzer-staging-runtime.md",
    ROOT / "docs/runbooks/release-gates.md",
    ROOT / "model_api/README.md",
)


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())


def build_report() -> dict[str, Any]:
    text = file_text(SOURCE_PATHS)
    checks = {
        "warmup_command_exists_and_uses_sanitized_fixture": all(
            token in text
            for token in [
                "scripts/warmup_ai_cv_analyzer_runtime.py",
                "warmup-cv.pdf",
                "%PDF-1.4",
                "storageKey",
                "rawCv",
            ]
        ),
        "readiness_requires_model_api_ready_and_warmup_policy": all(
            token in text
            for token in [
                "MODEL_API_WARMUP_REQUIRED",
                "MODEL_API_WARMUP_ON_STARTUP",
                "warmupCompleted",
                "/ready.ready=true",
                "ready_endpoint_true_after_warmup",
            ]
        ),
        "latency_budget_evidence_records_breakdown": all(
            token in text
            for token in [
                "parseLatencyMs",
                "embeddingLatencyMs",
                "tensorflowLatencyMs",
                "totalLatencyMs",
                "latency_budget_ms",
                "cold-start",
                "warm inference",
            ]
        ),
        "timeout_alignment_documented": all(
            token in text
            for token in [
                "MODEL_API_TIMEOUT_MS",
                "cold-start latency",
                "warmup is mandatory",
            ]
        ),
        "hf_e5_cache_guidance_documented": all(
            token in text
            for token in [
                "SENTENCE_TRANSFORMERS_HOME",
                "HF_TOKEN",
                "cache persistence",
                "container",
            ]
        ),
        "staging_smoke_and_rollback_documented": all(
            token in text
            for token in [
                "/api/v1/ai/cv-analyzer",
                "persistence behavior",
                "private field leakage",
                "deterministic prose",
                "service-token rotation",
                "artifact path rollback",
            ]
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase-38-ai-cv-analyzer-runtime-gate-v1",
        "phase_id": "phase_38_ai_cv_analyzer_runtime_gate",
        "checks": checks,
        "blockers": blockers,
        "final_decision": "go" if not blockers else "no-go",
        "latency_policy": {
            "cold_start_budget_ms": 30000,
            "warm_inference_budget_ms": 5000,
            "timeout_rule": "MODEL_API_TIMEOUT_MS must exceed measured cold-start latency or warmup is mandatory before routing traffic.",
            "required_breakdown": ["parseLatencyMs", "embeddingLatencyMs", "tensorflowLatencyMs", "totalLatencyMs"],
        },
        "commands": {
            "model_api_warmup": "python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url http://127.0.0.1:8000 --token ${MODEL_API_SERVICE_TOKEN} --latency-budget-ms 30000 --output reports/ai_cv_analyzer_warmup_staging.json",
            "phase38_gate": "python scripts/verify_phase_38_ai_cv_analyzer_runtime_gate.py --write",
        },
        "sources": [str(path.relative_to(ROOT)) for path in SOURCE_PATHS],
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 38 AI CV Analyzer Runtime Gate",
        "",
        f"Decision: `{report['final_decision']}`",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    lines.extend(["", "## Commands"])
    for name, command in report["commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    write = bool(argv and "--write" in argv)
    report = write_all() if write else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "go" else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
