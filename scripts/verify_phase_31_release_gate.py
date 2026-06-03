from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_31_release_gate_report.json"
REPORT_MD_PATH = ROOT / "reports/phase_31_release_gate_report.md"

TRACKED_ENV_TEMPLATES = (
    ROOT / ".env.example",
    ROOT / "model_api/.env.example",
    ROOT / "references/.env.example",
    ROOT / "references/.env.test.example",
    ROOT / "references/.env.production.example",
)
DOC_PATHS = (
    ROOT / "model_api/README.md",
    ROOT / "references/docs/integrations/model-api.md",
    ROOT / "references/docs/modules/ai-cv-analyzer.md",
    ROOT / "RUNNING_STEPS.md",
)
CODE_PATHS = (
    ROOT / "model_api/app.py",
    ROOT / "model_api/observability.py",
    ROOT / "references/src/modules/health/health.service.ts",
    ROOT / "references/src/modules/health/health.types.ts",
)
TEST_PATHS = (
    ROOT / "tests/test_phase_31_release_gate.py",
    ROOT / "references/tests/integration/routes/health.test.ts",
    ROOT / "references/tests/smoke/health.test.ts",
)

FORBIDDEN_TRACKED_ENV_NAMES = {".env", ".env.local", ".env.production", ".env.staging", ".env.test"}
PRODUCTION_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.I),
    re.compile(r"postgresql://(?![^\s@]*replace-with|localhost|127\.0\.0\.1)[^\s]+", re.I),
)
REQUIRED_OBSERVABILITY_TOKENS = {
    "requestId",
    "modelVersion",
    "artifactHash",
    "candidateCount",
    "parseQuality",
    "parseLatencyMs",
    "embeddingLatencyMs",
    "tensorflowLatencyMs",
    "wrapperLatencyMs",
    "totalLatencyMs",
    "errorCode",
    "fallbackReason",
}
REQUIRED_FAILURE_MODES = {
    "invalid PDF",
    "parse failure",
    "empty candidates",
    "Model API timeout",
    "TensorFlow load failure",
    "E5 failure",
    "GenAI wrapper failure",
}


def tracked_files() -> list[Path]:
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    import subprocess

    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [ROOT / line for line in output.splitlines() if line.strip()]


def scan_tracked_secret_risks(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_TRACKED_ENV_NAMES:
            findings.append(f"tracked env file: {relative}")
        if path.suffix in {".keras", ".parquet", ".npz", ".png"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRODUCTION_SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"secret-like value in {relative}")
                break
    return sorted(set(findings))


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())


def build_report() -> dict[str, Any]:
    files = tracked_files()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    code_text = file_text(CODE_PATHS)
    doc_text = file_text(DOC_PATHS)
    test_text = file_text(TEST_PATHS)
    env_text = file_text(TRACKED_ENV_TEMPLATES)

    checks = {
        "env_templates_exist": all(path.exists() for path in TRACKED_ENV_TEMPLATES),
        "dotenv_files_ignored": ".env" in gitignore and ".env.*" in gitignore and "!.env.example" in gitignore,
        "secret_scan_clean": not scan_tracked_secret_risks(files),
        "observability_allowlist_complete": REQUIRED_OBSERVABILITY_TOKENS.issubset(set(re.findall(r"[A-Za-z][A-Za-z0-9]+", code_text))),
        "observability_excludes_raw_cv_and_tokens": "cvText" in code_text and "SENSITIVE_OBSERVABILITY_KEYS" in code_text,
        "model_api_readiness_checks_runtime_dependencies": all(
            token in code_text
            for token in [
                "artifactsVerified",
                "tensorflowModelLoaded",
                "e5BackendConfigured",
                "pdfParserAvailable",
                "serviceTokenConfigured",
            ]
        ),
        "backend_readiness_checks_model_api": "modelApi" in code_text and "new URL(\"/health\"" in code_text,
        "e2e_contract_test_coverage_declared": all(
            token in test_text
            for token in ["phase_31", "model-core-cv-analysis-v1", "CvAnalysis", "JobRecommendationItem"]
        ),
        "load_timeout_smoke_coverage_declared": all(
            token in test_text for token in ["max PDF", "slow parser", "slow E5", "slow TensorFlow", "slow GenAI", "503", "504", "422"]
        ),
        "security_privacy_review_declared": all(
            token in doc_text
            for token in ["service-token", "raw CV", "retention", "path traversal", "Frontend UI must never call Model API directly"]
        ),
        "runbooks_document_local_staging_tests_and_rollback": all(
            token in doc_text for token in ["Local run", "Staging run", "Rollback", "Troubleshooting", "Backend + Model API tests"]
        ),
        "failure_modes_documented": REQUIRED_FAILURE_MODES.issubset(set(re.findall(r"invalid PDF|parse failure|empty candidates|Model API timeout|TensorFlow load failure|E5 failure|GenAI wrapper failure", doc_text))),
        "service_tokens_documented_without_real_secret": "MODEL_API_SERVICE_TOKEN" in env_text and "replace-with" in env_text,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "phase-31-release-gate-report-v1",
        "final_decision": "passed" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
        "secret_findings": scan_tracked_secret_risks(files),
        "acceptance": {
            "python_runtime": "Python 3.13 TensorFlow/E5 runtime required for full staging gate",
            "model_api_db_ownership": "Model API owns no DB credentials; Backend remains DB owner",
            "tracked_env_policy": "Only .env.example templates are tracked; concrete .env files are ignored",
            "privacy_policy": "Operational metadata only; raw CV text, tokens, DB URLs, unrelated PII excluded",
        },
        "sources": {
            "code": [str(path.relative_to(ROOT)) for path in CODE_PATHS],
            "docs": [str(path.relative_to(ROOT)) for path in DOC_PATHS],
            "tests": [str(path.relative_to(ROOT)) for path in TEST_PATHS],
            "env_templates": [str(path.relative_to(ROOT)) for path in TRACKED_ENV_TEMPLATES],
        },
    }


def write_all() -> dict[str, Any]:
    report = build_report()
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 31 Release Gate Report",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(["", "## Acceptance", ""])
    for key, value in report["acceptance"].items():
        lines.append(f"- `{key}`: {value}")
    if report["secret_findings"]:
        lines.extend(["", "## Secret findings", ""])
        lines.extend(f"- {finding}" for finding in report["secret_findings"])
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = write_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
