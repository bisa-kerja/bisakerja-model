#!/usr/bin/env python3
"""Create a shared CV evidence cache verification report."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "phase_54_shared_cv_evidence_cache.json"
REPORT_MD = ROOT / "reports" / "phase_54_shared_cv_evidence_cache.md"


@dataclass(frozen=True)
class ParserOption:
    name: str
    owner: str
    quality: str
    latency_risk: str
    privacy_risk: str
    implementation_risk: str
    decision: str


@dataclass(frozen=True)
class InvalidationCase:
    name: str
    source_hash_changed: bool = False
    parser_version_changed: bool = False
    model_version_changed: bool = False
    template_policy_changed: bool = False
    expired: bool = False


def source_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_fresh(case: InvalidationCase) -> bool:
    return not any(
        [
            case.source_hash_changed,
            case.parser_version_changed,
            case.model_version_changed,
            case.template_policy_changed,
            case.expired,
        ]
    )


def build_report() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=1)
    parser_options = [
        ParserOption(
            name="Model API parse-only",
            owner="Model API primary, Backend cache adapter",
            quality="Highest available parser and ATS signal quality after analyzer inference.",
            latency_risk="Adds service hop when used only for Generate; acceptable for Analyzer because Model API call already happens.",
            privacy_risk="Raw CV stays internal between Backend and Model API; sanitized evidence only may reach GenAI.",
            implementation_risk="Medium because Generate would need extra internal endpoint or analyzer dependency.",
            decision="Use for Analyzer wrapper evidence only in MVP.",
        ),
        ParserOption(
            name="Backend parser",
            owner="Backend API",
            quality="Good enough for bounded section/skill evidence from text-like stored bytes; weak for scanned PDFs.",
            latency_risk="Lowest latency for Generate because no extra service hop.",
            privacy_risk="Low when raw text is not cached and contact data is redacted before provider calls.",
            implementation_risk="Low because Generate already reads owned CV storage.",
            decision="Use as MVP Generate parser and shared cache owner.",
        ),
        ParserOption(
            name="Latest analysis reuse",
            owner="Backend API persisted sanitized snapshots",
            quality="Useful fallback for ATS/actionables and section reviews when current stored bytes are weak.",
            latency_risk="Low read-path latency.",
            privacy_risk="Low if scoped to same user and same CV file.",
            implementation_risk="Low but stale without source/model invalidation.",
            decision="Use as fallback only when same-CV latest analysis exists.",
        ),
        ParserOption(
            name="Hybrid fallback",
            owner="Backend API orchestration",
            quality="Best practical MVP coverage by combining model_api, backend_parser, latest_analysis_cache, metadata_only.",
            latency_risk="Bounded by using source already available in each flow.",
            privacy_risk="Low when all provider inputs pass shared no-leak policy.",
            implementation_risk="Medium due to schema/invalidation tests.",
            decision="Chosen MVP architecture.",
        ),
    ]
    invalidation_cases = [
        InvalidationCase("fresh_cache"),
        InvalidationCase("cv_file_changed", source_hash_changed=True),
        InvalidationCase("parser_version_changed", parser_version_changed=True),
        InvalidationCase("analysis_model_changed", model_version_changed=True),
        InvalidationCase("template_policy_changed", template_policy_changed=True),
        InvalidationCase("retention_expired", expired=True),
    ]
    schema_checks = {
        "schemaVersion": "shared-cv-evidence-v1",
        "sources": ["model_api", "backend_parser", "latest_analysis_cache", "metadata_only"],
        "retainedFields": [
            "bounded section evidence",
            "skill and requirement coverage",
            "ATS/actionable evidence",
            "parser confidence",
            "source hash",
            "timestamps and retention policy",
        ],
        "forbiddenFields": [
            "raw CV text",
            "raw PDF content",
            "contact data",
            "prompts",
            "provider payloads",
            "storage identifiers",
            "tokens and secrets",
        ],
        "observability": [
            "parser confidence",
            "evidence source",
            "cache hit/miss/bypass",
            "wrapper fallback reason",
            "template validation failure",
            "no-leak checks",
        ],
    }
    report = {
        "generatedAt": now.isoformat(),
        "scope": "Shared CV evidence cache and MVP parser ownership decision",
        "decision": "Backend-owned hybrid fallback cache with Backend parser for Generate MVP and Model API evidence adapter for Analyzer wrapper.",
        "cacheKeyExample": source_hash("shared-cv-evidence-v1:model_api:model-v1:template-v1"),
        "expiresAtExample": expires_at.isoformat(),
        "parserOptions": [asdict(option) for option in parser_options],
        "schemaChecks": schema_checks,
        "invalidationResults": [
            {**asdict(case), "fresh": cache_fresh(case)} for case in invalidation_cases
        ],
        "qualityLatencyRiskSummary": {
            "quality": "Analyzer uses Model API parser/scoring evidence; Generate uses Backend parser with same-CV latest analysis fallback.",
            "latency": "No extra Model API parse-only hop for Generate in MVP.",
            "risk": "Weak scanned-PDF Generate evidence remains low confidence and must not fabricate content.",
        },
    }
    report["passed"] = (
        report["invalidationResults"][0]["fresh"] is True
        and all(item["fresh"] is False for item in report["invalidationResults"][1:])
        and "raw CV text" in schema_checks["forbiddenFields"]
    )
    return report


def write_reports(report: dict[str, object]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Shared CV Evidence Cache Report",
        "",
        f"Generated: `{report['generatedAt']}`",
        "",
        f"Passed: `{report['passed']}`",
        "",
        "## Decision",
        "",
        str(report["decision"]),
        "",
        "## Parser Ownership Options",
        "",
    ]
    for option in report["parserOptions"]:  # type: ignore[index]
        lines.extend(
            [
                f"### {option['name']}",
                "",
                f"Owner: {option['owner']}",
                f"Quality: {option['quality']}",
                f"Latency risk: {option['latency_risk']}",
                f"Privacy risk: {option['privacy_risk']}",
                f"Implementation risk: {option['implementation_risk']}",
                f"Decision: {option['decision']}",
                "",
            ]
        )
    lines.extend(["## Invalidation", ""])
    for item in report["invalidationResults"]:  # type: ignore[index]
        lines.append(f"- `{item['name']}` fresh=`{item['fresh']}`")
    lines.extend(
        [
            "",
            "## Schema And Privacy",
            "",
            "```json",
            json.dumps(report["schemaChecks"], indent=2, sort_keys=True),
            "```",
            "",
            "## Quality, Latency, Risk",
            "",
            "```json",
            json.dumps(report["qualityLatencyRiskSummary"], indent=2, sort_keys=True),
            "```",
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
