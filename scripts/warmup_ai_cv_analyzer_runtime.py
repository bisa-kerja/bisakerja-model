#!/usr/bin/env python3
"""Warm Model API AI CV Analyzer runtime before staging/demo traffic."""

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
FIXTURE_PDF = b"%PDF-1.4\n1 0 obj<</Type /Page>>stream\n(Summary Backend engineer) Tj\n(Skills Python SQL REST API) Tj\n(Experience 2020 2024) Tj\nendstream\n%%EOF"


def now_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def request_json(url: str, *, token: str | None = None, timeout: float = 30) -> tuple[int, dict[str, Any], float]:
    headers = {"accept": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    started = perf_counter()
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as res:  # noqa: S310 - operator-provided URL for staging smoke
            body = json.loads(res.read().decode("utf-8"))
            return res.status, body, now_ms(started)
    except HTTPError as exc:
        payload = exc.read().decode("utf-8")
        body = json.loads(payload) if payload else {"error": str(exc)}
        return exc.code, body, now_ms(started)


def multipart_body(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----bisakerja-model-api-warmup"
    chunks: list[bytes] = []
    for name, value in fields.items():
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


def post_model_warmup(base_url: str, *, token: str | None, timeout: float) -> tuple[int, dict[str, Any], float]:
    fields = {
        "requestId": "warmup_cv_analysis_staging_fixture",
        "language": "en",
        "inputMode": "UPLOAD",
        "compareSource": "JOB_SEARCH",
        "jobRoles": json.dumps(["Backend Engineer"]),
        "jobCandidates": json.dumps(
            [
                {
                    "jobId": "warmup-job-backend-engineer",
                    "scoringInput": {
                        "titleText": "Backend Engineer",
                        "requirementSummary": "Build REST APIs using Python and SQL.",
                        "requiredSkills": ["python", "sql", "rest api"],
                        "roleFamily": "backend",
                    },
                }
            ]
        ),
        "rankingPolicy": json.dumps(
            {
                "maxRecommendations": 1,
                "requireCandidateJobIds": True,
                "deduplicateByJobId": True,
                "backendOwnsHydration": True,
            }
        ),
    }
    body, boundary = multipart_body(fields, "cvFile", "warmup-cv.pdf", FIXTURE_PDF)
    headers = {
        "content-type": f"multipart/form-data; boundary={boundary}",
        "accept": "application/json",
        "x-model-api-include-observability": "true",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"
    started = perf_counter()
    req = Request(f"{base_url.rstrip('/')}/internal/model/cv-analysis", data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as res:  # noqa: S310 - operator-provided URL for staging smoke
            payload = json.loads(res.read().decode("utf-8"))
            return res.status, payload, now_ms(started)
    except HTTPError as exc:
        payload = exc.read().decode("utf-8")
        body_json = json.loads(payload) if payload else {"error": str(exc)}
        return exc.code, body_json, now_ms(started)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    ready_before = request_json(f"{args.model_api_url.rstrip('/')}/ready", timeout=args.timeout)
    warmup = post_model_warmup(args.model_api_url, token=args.token, timeout=args.timeout)
    ready_after = request_json(f"{args.model_api_url.rstrip('/')}/ready", timeout=args.timeout)
    data = warmup[1].get("data") if isinstance(warmup[1], dict) and isinstance(warmup[1].get("data"), dict) else warmup[1]
    observability = data.get("observability", {}) if isinstance(data, dict) else {}
    checks = {
        "ready_endpoint_reachable_before_warmup": ready_before[0] == 200,
        "warmup_inference_succeeded": warmup[0] == 200 and isinstance(data, dict) and data.get("schemaVersion") == "model-core-cv-analysis-v1",
        "ready_endpoint_true_after_warmup": ready_after[0] == 200 and bool(ready_after[1].get("ready")),
        "latency_budget_met": warmup[2] <= args.latency_budget_ms,
        "private_fields_not_returned": "storageKey" not in json.dumps(warmup[1]) and "rawCv" not in json.dumps(warmup[1]),
        "observability_has_latency_breakdown": all(
            key in observability for key in ["embeddingLatencyMs", "tensorflowLatencyMs", "totalLatencyMs", "parseLatencyMs"]
        ),
    }
    return {
        "schema_version": "ai-cv-analyzer-runtime-warmup-v1",
        "model_api_url": args.model_api_url,
        "latency_budget_ms": args.latency_budget_ms,
        "checks": checks,
        "final_decision": "pass" if all(checks.values()) else "fail",
        "ready_before": {"status_code": ready_before[0], "latency_ms": ready_before[2], "body": ready_before[1]},
        "warmup": {
            "status_code": warmup[0],
            "latency_ms": warmup[2],
            "schemaVersion": data.get("schemaVersion") if isinstance(data, dict) else None,
            "observability": observability,
        },
        "ready_after": {"status_code": ready_after[0], "latency_ms": ready_after[2], "body": ready_after[1]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=None, help="MODEL_API_SERVICE_TOKEN value")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--latency-budget-ms", type=float, default=30000)
    parser.add_argument("--output", type=Path, default=None)
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
