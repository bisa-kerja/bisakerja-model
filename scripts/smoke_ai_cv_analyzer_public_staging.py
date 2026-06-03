#!/usr/bin/env python3
"""Run public Backend-to-Model AI CV Analyzer staging smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "artifacts/smoke/sanitized-cv.pdf"
PRIVATE_FIELD_TOKENS = (
    "storageKey",
    "rawCv",
    "cvText",
    "cvFileBytes",
    "authorization",
    "Bearer ",
    "MODEL_API_SERVICE_TOKEN",
    "DATABASE_URL",
    "DIRECT_DATABASE_URL",
    "system prompt",
    "developer prompt",
    "artifact_path",
    "MODEL_API_ARTIFACT_ROOT",
)


def now_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def multipart_body(fields: list[tuple[str, str]], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----bisakerja-ai-cv-analyzer-public-smoke"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


def post_public_analyzer(args: argparse.Namespace) -> tuple[int, dict[str, Any], float]:
    pdf_path = Path(args.fixture_pdf)
    pdf_bytes = pdf_path.read_bytes()
    fields: list[tuple[str, str]] = [
        ("language", args.language),
        ("inputMode", args.input_mode),
        ("compareSource", args.compare_source),
        ("persistResult", "true" if args.persist_result else "false"),
    ]
    for role in args.job_role:
        fields.append(("jobRoles", role))
    if args.direct_job_id:
        fields.append(("directJobId", args.direct_job_id))
    if args.cv_file_id:
        fields.append(("cvFileId", args.cv_file_id))

    body, boundary = multipart_body(fields, "cvFile", pdf_path.name, pdf_bytes)
    headers = {
        "accept": "application/json",
        "content-type": f"multipart/form-data; boundary={boundary}",
        "authorization": f"Bearer {args.user_access_token}",
    }
    if args.request_id:
        headers[args.request_id_header] = args.request_id

    started = perf_counter()
    req = Request(f"{args.backend_api_url.rstrip('/')}/api/v1/ai/cv-analyzer", data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=args.timeout) as res:  # noqa: S310 - operator-provided staging URL
            payload = json.loads(res.read().decode("utf-8"))
            return res.status, payload, now_ms(started)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        payload = json.loads(raw) if raw else {"error": str(exc)}
        return exc.code, payload, now_ms(started)
    except URLError as exc:
        return 0, {"error": str(exc)}, now_ms(started)


def get_path(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_checks(status_code: int, body: dict[str, Any], latency_ms: float, args: argparse.Namespace) -> dict[str, bool]:
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    analysis = get_path(data, "analysisResult") if isinstance(data, dict) else None
    recommendations = analysis.get("jobRecommendations") if isinstance(analysis, dict) else None
    generated_cv = analysis.get("generatedCv") if isinstance(analysis, dict) else None
    model = analysis.get("model") if isinstance(analysis, dict) else None
    body_text = json.dumps(body, sort_keys=True)
    return {
        "http_200": status_code == 200,
        "public_envelope_shape": all(key in body for key in ["success", "message", "data", "meta"]),
        "success_true": body.get("success") is True,
        "schema_version_cv_analysis_v2": isinstance(analysis, dict) and analysis.get("schemaVersion") == "cv-analysis-v2",
        "hydrated_recommendations_max_5": isinstance(recommendations, list) and len(recommendations) <= 5,
        "generated_cv_unavailable": isinstance(generated_cv, dict) and generated_cv.get("available") is False,
        "model_metadata_present": isinstance(model, dict) and bool(model.get("name")) and bool(model.get("version")),
        "latency_budget_met": latency_ms <= args.latency_budget_ms,
        "private_fields_not_returned": not any(token in body_text for token in PRIVATE_FIELD_TOKENS),
        "persistence_policy_recorded": isinstance(args.persist_result, bool),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    status_code, body, latency_ms = post_public_analyzer(args)
    checks = build_checks(status_code, body, latency_ms, args)
    analysis = get_path(body, "data", "analysisResult") if isinstance(body, dict) else None
    model = analysis.get("model") if isinstance(analysis, dict) else None
    return {
        "schema_version": "ai-cv-analyzer-public-staging-smoke-v1",
        "backend_api_url": args.backend_api_url,
        "request_id": args.request_id,
        "compare_source": args.compare_source,
        "input_mode": args.input_mode,
        "job_roles": args.job_role,
        "persist_result": args.persist_result,
        "latency_budget_ms": args.latency_budget_ms,
        "latency_ms": latency_ms,
        "status_code": status_code,
        "checks": checks,
        "final_decision": "pass" if all(checks.values()) else "fail",
        "model": model if isinstance(model, dict) else None,
        "response_summary": {
            "success": body.get("success"),
            "schemaVersion": analysis.get("schemaVersion") if isinstance(analysis, dict) else None,
            "recommendationCount": len(analysis.get("jobRecommendations", [])) if isinstance(analysis, dict) and isinstance(analysis.get("jobRecommendations"), list) else None,
        },
        "candidate_fixture_policy": "Run against seeded non-production Backend data. `cd references && bun run prisma:seed` provides active Backend Developer jobs for JOB_SEARCH.",
        "private_field_tokens_checked": list(PRIVATE_FIELD_TOKENS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-api-url", default="http://127.0.0.1:3000")
    parser.add_argument("--user-access-token", required=True)
    parser.add_argument("--fixture-pdf", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--job-role", action="append", default=["Backend Developer"])
    parser.add_argument("--language", choices=["en", "id"], default="en")
    parser.add_argument("--input-mode", choices=["UPLOAD", "REFERENCE"], default="UPLOAD")
    parser.add_argument("--compare-source", choices=["JOB_SEARCH", "DIRECT_JOB_DETAIL", "BOOKMARK"], default="JOB_SEARCH")
    parser.add_argument("--direct-job-id", default=None)
    parser.add_argument("--cv-file-id", default=None)
    parser.add_argument("--persist-result", action="store_true")
    parser.add_argument("--request-id", default="phase41_public_staging_smoke")
    parser.add_argument("--request-id-header", default="x-request-id")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--latency-budget-ms", type=float, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_41_ai_cv_analyzer_public_staging_smoke.json")
    args = parser.parse_args(argv)

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["final_decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
