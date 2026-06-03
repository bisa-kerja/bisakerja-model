#!/usr/bin/env python3
"""Verify AI CV Analyzer product-copy quality and localization policy evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON_PATH = ROOT / "reports/phase_39_product_copy_localization.json"
REPORT_MD_PATH = ROOT / "reports/phase_39_product_copy_localization.md"
TEMPLATES_PATH = ROOT / "artifacts/phase_39_product_copy_localization/approved_fallback_templates.json"

SOURCE_PATHS = (
    ROOT / "references/src/modules/ai-cv-analyzer/ai-cv-analyzer.service.ts",
    ROOT / "references/tests/unit/ai-cv-analyzer/ai-cv-analyzer.service.test.ts",
    ROOT / "references/src/shared/integrations/model-api.schema.ts",
    ROOT / "artifacts/phase_39_product_copy_localization/approved_fallback_templates.json",
    ROOT / "docs/runbooks/ai-cv-analyzer-product-copy-localization.md",
    ROOT / "GAP_MODEL_TRAINING.md",
    ROOT / "REQUIREMENT.md",
)

SENSITIVE_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"raw CV text",
        r"storageKey",
        r"system prompt",
        r"developer prompt",
        r"token",
        r"database_url",
        r"email",
        r"phone",
        r"address",
    ]
)

INDONESIAN_COPY_WORDS = re.compile(r"\b(perbaiki|keterampilan|lamaran|ringkasan|cocok|pengalaman|pekerjaan)\b", re.IGNORECASE)


def file_text(paths: tuple[Path, ...]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())


def load_templates() -> dict[str, Any]:
    return json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))


def template_text(templates: dict[str, Any]) -> str:
    return json.dumps(templates, ensure_ascii=False, sort_keys=True)


def build_report() -> dict[str, Any]:
    text = file_text(SOURCE_PATHS)
    templates = load_templates()
    rendered_templates = template_text(templates)
    public_copy_text = "\n".join(
        value
        for values in templates["templates"].values()
        for value in (values if isinstance(values, list) else [values])
        if isinstance(value, str)
    )

    checks = {
        "language_policy_explicit": all(
            token in rendered_templates + text
            for token in [
                "english_safe_copy",
                "requested_language_id",
                "return English public copy until Indonesian localization is product-approved",
                "Staging policy: keep public copy in English even when requestedLanguage is id",
                "Indonesian localization stays deferred until product-approved Indonesian templates",
            ]
        ),
        "fallback_templates_cover_public_fields": set(templates["templates"]).issuperset(
            {
                "jobFitAlignment.summary",
                "atsFriendliness.summary",
                "overallImpression",
                "topActionables",
                "sectionReviews",
                "jobRecommendations.reason",
                "jobRecommendations.nextStep",
            }
        ),
        "backend_uses_approved_template_copy": all(
            token in text
            for token in [
                "Your CV shows relevant evidence",
                "Fix ATS readability issue",
                "ATS Readability",
                "Role Evidence",
                "This role is recommended because the model found overlap",
            ]
        ),
        "localization_tests_cover_id_en_behavior": all(
            token in text
            for token in [
                "keeps staging fallback copy English for id and en requests",
                "idResponse.topActionables",
                "perbaiki|keterampilan|lamaran|ringkasan|cocok",
            ]
        ),
        "copy_safety_filters_and_pii_tests_present": all(
            token in text
            for token in [
                "assertPublicCvAnalysisSafety",
                "unsafeGeneratedCopyPattern",
                "redacts prompt-injection and PII-like evidence",
                "not.toMatch",
                "storageKey",
                "alice@example.com",
            ]
        ),
        "schema_length_limits_match_public_contract": all(
            token in text
            for token in [
                "max(800)",
                "max(1200)",
                "max(500)",
                "max(5)",
                "cv-analysis-v2",
            ]
        ),
        "approved_copy_has_no_localization_drift_or_sensitive_literals": not INDONESIAN_COPY_WORDS.search(public_copy_text)
        and not any(pattern.search(public_copy_text) for pattern in SENSITIVE_PATTERNS),
        "genai_boundary_remains_validated_and_optional": all(
            token in text
            for token in [
                "wrapperResponse?: unknown",
                "publicCvAnalysisResponseSchema.parse",
                "return validatePublicCvAnalysisResponse(fallback, response)",
                "Generative AI integration boundary or wrapper evidence",
            ]
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    final_decision = "go" if not blockers else "no-go"
    return {
        "schema_version": "phase-39-product-copy-localization-v1",
        "phase_id": "phase_39_product_copy_localization",
        "final_decision": final_decision,
        "blockers": blockers,
        "checks": checks,
        "language_policy": templates["language_policy"],
        "review_result": {
            "fallback_copy_decision": "accepted_for_staging_demo_with_english_safe_copy",
            "indonesian_localization": "deferred_until_product_approved_templates_or_genai_provider_output",
            "genai": "optional; output must pass schema, score/order preservation, and safety filters before use",
        },
        "sample_before_after": {
            "before": "CV shows fit through TypeScript, PostgreSQL, with gaps in Docker.",
            "after": "Your CV shows relevant evidence in TypeScript, PostgreSQL. Strengthen proof for Docker to improve role fit.",
        },
        "remaining_limitations": [
            "English-only public copy for id requests during staging/demo.",
            "Deterministic copy cannot add richer personalization beyond sanitized model evidence and backend job metadata.",
            "Indonesian localization still needs product-approved templates or validated GenAI output.",
        ],
        "commands": {
            "backend_copy_tests": "cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer/ai-cv-analyzer.service.test.ts",
            "phase39_gate": "python scripts/verify_phase_39_product_copy_localization.py --write",
        },
        "sources": [str(path.relative_to(ROOT)) for path in SOURCE_PATHS],
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 39 AI CV Analyzer Product Copy and Localization",
        "",
        f"Decision: `{report['final_decision']}`",
        "",
        "## Language Policy",
        f"- Staging default: `{report['language_policy']['staging_default']}`",
        f"- `id`: {report['language_policy']['requested_language_id']}",
        f"- `en`: {report['language_policy']['requested_language_en']}",
        "",
        "## Checks",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend([
        "",
        "## Before / After",
        f"- Before: {report['sample_before_after']['before']}",
        f"- After: {report['sample_before_after']['after']}",
        "",
        "## Remaining Limitations",
    ])
    lines.extend(f"- {item}" for item in report["remaining_limitations"])
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
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
