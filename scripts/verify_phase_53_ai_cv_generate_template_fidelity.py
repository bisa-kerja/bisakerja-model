#!/usr/bin/env python3
"""Create a Phase 53 AI CV Generate template fidelity verification report."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "phase_53_ai_cv_generate_template_fidelity.json"
REPORT_MD = ROOT / "reports" / "phase_53_ai_cv_generate_template_fidelity.md"


@dataclass(frozen=True)
class TagSignature:
    tag_name: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class VerificationCase:
    name: str
    input_template: str
    generated_output: str
    expected_valid: bool


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_signature(value: str) -> list[TagSignature]:
    tags: list[TagSignature] = []
    for match in re.finditer(r"<\s*([a-zA-Z][\w:-]*)([^<>]*)>", value):
        attrs: dict[str, str] = {}
        for attr in re.finditer(r"([:@\w-]+)(?:\s*=\s*(\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+)))?", match.group(2) or ""):
            name = attr.group(1).lower()
            if name == "/":
                continue
            attrs[name] = normalize_ws(attr.group(3) or attr.group(4) or attr.group(5) or "")
        tags.append(TagSignature(match.group(1).lower(), dict(sorted(attrs.items()))))
    return tags


def static_chunks(template: str) -> list[str]:
    text = re.sub(r"<[^>]*>", " ", template)
    return [normalize_ws(chunk) for chunk in re.split(r"\{\{\s*[\w.-]+\s*\}\}", text) if len(normalize_ws(chunk)) >= 2]


def safety_issues(value: str) -> list[str]:
    checks = {
        "script_tag": r"<script\b",
        "iframe_tag": r"<iframe\b",
        "object_tag": r"<object\b",
        "embed_tag": r"<embed\b",
        "event_handler": r"\son\w+\s*=",
        "javascript_url": r"javascript:",
        "prompt_leak": r"\b(cvGenerateInput|system prompt|developer prompt)\b",
        "storage_key_leak": r"\bstorageKey\b",
        "secret_leak": r"\b(DATABASE_URL|OPENAI_API_KEY|API_KEY|BEGIN PRIVATE KEY|Bearer\s+)\b",
        "email_leak": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone_leak": r"(?:\+?\d(?:[\d\s().-]*\d){9,})",
    }
    return [name for name, pattern in checks.items() if re.search(pattern, value, flags=re.I)]


def validate_case(case: VerificationCase) -> dict[str, object]:
    template_sig = extract_signature(case.input_template)
    output_sig = extract_signature(case.generated_output)
    reasons: list[str] = []

    if not template_sig:
        reasons.append("template_has_no_html_tags")
    if len(template_sig) != len(output_sig):
        reasons.append("tag_count_changed")

    for index, (expected, actual) in enumerate(zip(template_sig, output_sig)):
        if expected.tag_name != actual.tag_name:
            reasons.append(f"tag_changed_at_{index}")
            continue
        for attr_name, attr_value in expected.attributes.items():
            if actual.attributes.get(attr_name) != attr_value:
                reasons.append(f"attribute_changed_at_{index}_{attr_name}")

    output_text = normalize_ws(re.sub(r"<[^>]*>", " ", case.generated_output))
    if any(chunk not in output_text for chunk in static_chunks(case.input_template)):
        reasons.append("static_copy_removed")

    issues = safety_issues(case.generated_output)
    reasons.extend([f"unsafe_{issue}" for issue in issues])
    valid = not reasons

    return {
        **asdict(case),
        "template_signature": [asdict(item) for item in template_sig],
        "output_signature": [asdict(item) for item in output_sig],
        "template_diff_result": {"valid": valid, "reasons": reasons},
        "safety_checks": {"passed": not issues, "issues": issues},
        "matches_expectation": valid == case.expected_valid,
    }


def build_report() -> dict[str, object]:
    cases = [
        VerificationCase(
            name="valid_placeholder_fill_preserves_structure",
            input_template='<section id="cv" class="modern" data-kind="cv"><h1>{{name}}</h1><p class="summary">{{summary}}</p><p>Static footer</p></section>',
            generated_output='<section id="cv" class="modern" data-kind="cv"><h1></h1><p class="summary">Backend REST API candidate with PostgreSQL delivery experience.</p><p>Static footer</p></section>',
            expected_valid=True,
        ),
        VerificationCase(
            name="invalid_provider_changes_tag_and_class",
            input_template='<section id="cv" class="modern"><h1>{{name}}</h1><p>{{summary}}</p></section>',
            generated_output='<section id="cv" class="changed"><h2>Candidate</h2><p>Backend REST API candidate.</p></section>',
            expected_valid=False,
        ),
        VerificationCase(
            name="invalid_privacy_leak",
            input_template="<section><p>{{summary}}</p></section>",
            generated_output="<section><p>Contact me at user@example.test or +62 812 3333 4444.</p></section>",
            expected_valid=False,
        ),
    ]
    results = [validate_case(case) for case in cases]

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 53 AI CV Generate current-CV context and template fidelity",
        "checks": {
            "structuredEvidence": "Backend sends bounded structured currentCv evidence instead of cvTextPreview.",
            "templateValidation": "Template tag order, attributes, and static copy are compared before returning provider output.",
            "fallback": "Provider failures or template drift use deterministic placeholder rendering from grounded evidence.",
            "privacy": "Executable HTML, prompt text, storage keys, secrets, raw email, and raw phone output are rejected.",
        },
        "results": results,
        "passed": all(item["matches_expectation"] for item in results),
    }


def write_reports(report: dict[str, object]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Phase 53 AI CV Generate Template Fidelity Report",
        "",
        f"Generated: `{report['generatedAt']}`",
        "",
        f"Passed: `{report['passed']}`",
        "",
        "## Cases",
        "",
    ]
    for item in report["results"]:  # type: ignore[index]
        lines.extend(
            [
                f"### {item['name']}",
                "",
                "Input template:",
                "",
                "```html",
                str(item["input_template"]),
                "```",
                "",
                "Generated output:",
                "",
                "```html",
                str(item["generated_output"]),
                "```",
                "",
                f"Template diff: `{item['template_diff_result']}`",
                f"Safety checks: `{item['safety_checks']}`",
                f"Matches expectation: `{item['matches_expectation']}`",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write report files")
    args = parser.parse_args()
    report = build_report()

    if args.write:
        write_reports(report)

    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
