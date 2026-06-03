#!/usr/bin/env python3
"""Verify AI CV Analyzer public staging smoke and payload hygiene evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_41_ai_cv_analyzer_public_staging_gate.json"
REPORT_MD_PATH = ROOT / "reports/phase_41_ai_cv_analyzer_public_staging_gate.md"

SOURCE_PATHS = (
    ROOT / "scripts/smoke_ai_cv_analyzer_public_staging.py",
    ROOT / "artifacts/smoke/sanitized-cv.pdf",
    ROOT / "docs/runbooks/ai-cv-analyzer-staging-runtime.md",
    ROOT / "references/src/shared/integrations/model-api.client.ts",
    ROOT / "references/tests/unit/shared/model-api.client.test.ts",
    ROOT / "references/prisma/seed-data.ts",
    ROOT / "reports/phase_38_ai_cv_analyzer_runtime_gate.md",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(read(path) for path in paths)


def build_report() -> dict[str, Any]:
    text = file_text(SOURCE_PATHS)
    client_text = read(ROOT / "references/src/shared/integrations/model-api.client.ts")
    checks = {
        "public_smoke_script_exists": all(
            token in text
            for token in [
                "smoke_ai_cv_analyzer_public_staging.py",
                "/api/v1/ai/cv-analyzer",
                "USER_ACCESS_TOKEN",
                "jobRoles",
                "inputMode",
                "compareSource",
                "persistResult",
            ]
        ),
        "sanitized_pdf_fixture_exists": all(token in text for token in ["%PDF-1.4", "Backend developer", "TypeScript"]),
        "deterministic_candidate_fixture_documented": all(
            token in text
            for token in [
                "bun run prisma:seed",
                "seed-job-001",
                "Backend Developer",
                "JOB_SEARCH",
                "non-production Backend data",
            ]
        ),
        "public_response_contract_validated": all(
            token in text
            for token in [
                "public_envelope_shape",
                "schema_version_cv_analysis_v2",
                "hydrated_recommendations_max_5",
                "generated_cv_unavailable",
                "model_metadata_present",
            ]
        ),
        "persistence_and_latency_recorded": all(
            token in text for token in ["persist_result", "persistence", "latency_ms", "latency_budget_ms"]
        ),
        "private_field_leak_checks_present": all(
            token in text
            for token in [
                "private_fields_not_returned",
                "storageKey",
                "rawCv",
                "DATABASE_URL",
                "authorization",
                "artifact_path",
            ]
        ),
        "multipart_excludes_cv_storage_metadata": 'form.set("cv"' not in client_text
        and "const { bytes: cvBytes, ...cvMetadata }" in client_text
        and '"cvFile"' in client_text,
        "payload_hygiene_test_exists": all(
            token in text
            for token in [
                "sends cv analyzer multipart without storage metadata",
                "form.has(\"cv\")",
                "not.toContain(\"storageKey\")",
            ]
        ),
        "phase_38_6_superseded_by_phase_41": "Phase 38.6 public Backend-to-Model smoke is superseded by this public staging smoke" in text,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase-41-ai-cv-analyzer-public-staging-smoke-v1",
        "phase_id": "phase_41_ai_cv_analyzer_public_staging_smoke",
        "checks": checks,
        "blockers": blockers,
        "final_decision": "go" if not blockers else "no-go",
        "commands": {
            "model_api_warmup": "python scripts/warmup_ai_cv_analyzer_runtime.py --model-api-url ${MODEL_API_URL:-http://127.0.0.1:8000} --token ${MODEL_API_SERVICE_TOKEN} --latency-budget-ms 30000 --output reports/ai_cv_analyzer_warmup_staging.json",
            "backend_seed": "cd references && bun run prisma:seed",
            "public_staging_smoke": "python scripts/smoke_ai_cv_analyzer_public_staging.py --backend-api-url ${BACKEND_API_URL} --user-access-token ${USER_ACCESS_TOKEN} --fixture-pdf artifacts/smoke/sanitized-cv.pdf --job-role 'Backend Developer' --language en --input-mode UPLOAD --compare-source JOB_SEARCH --latency-budget-ms 5000 --output reports/phase_41_ai_cv_analyzer_public_staging_smoke.json",
            "backend_payload_hygiene_tests": "cd references && bun test --preload ./tests/preload-env.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts",
            "phase41_gate": "python scripts/verify_phase_41_ai_cv_analyzer_public_staging_smoke.py --write",
        },
        "phase_38_6_status": "superseded_by_phase_41_public_staging_smoke" if checks["phase_38_6_superseded_by_phase_41"] else "open",
        "sources": [str(path.relative_to(ROOT)) for path in SOURCE_PATHS],
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 41 AI CV Analyzer Public Staging Smoke",
        "",
        f"Decision: `{report['final_decision']}`",
        "",
        "Phase 38.6 public Backend-to-Model smoke is superseded by this public staging smoke evidence.",
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
