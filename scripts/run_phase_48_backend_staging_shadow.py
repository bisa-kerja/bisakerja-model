#!/usr/bin/env python3
"""Run Phase 48 Backend/Staging integration smoke and shadow comparison.

This operator script is safe-by-default: it writes local evidence only, redacts
secrets, and never mutates Backend state unless --persist-result is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "artifacts/smoke/sanitized-cv.pdf"
PHASE25_ARTIFACT_ROOT = "artifacts/phase_25_tensorflow_training_delivery"
PHASE46_ARTIFACT_ROOT = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
PHASE25_EMBEDDING_MODEL = "intfloat/e5-base-v2"
PHASE46_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_OUTPUT = ROOT / "reports/phase_48_backend_staging_shadow_report.json"

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
    "OPENROUTER_API_KEY",
    "system prompt",
    "developer prompt",
    "artifact_path",
    "MODEL_API_ARTIFACT_ROOT",
)
REDACTED_KEYS = {
    "authorization",
    "token",
    "access_token",
    "user_access_token",
    "model_api_service_token",
    "MODEL_API_SERVICE_TOKEN",
    "USER_ACCESS_TOKEN",
}
DEFAULT_JOB_CANDIDATES: list[dict[str, Any]] = [
    {
        "jobId": "phase48-seed-backend-001",
        "scoringInput": {
            "titleText": "Backend Developer",
            "requirementSummary": "Build REST APIs with Python, SQL, and TypeScript integration.",
            "requiredSkills": ["python", "sql", "rest api", "typescript"],
            "roleFamily": "backend",
        },
    },
    {
        "jobId": "phase48-seed-frontend-001",
        "scoringInput": {
            "titleText": "Frontend Developer",
            "requirementSummary": "Build React UI and collaborate with API teams.",
            "requiredSkills": ["react", "typescript", "css"],
            "roleFamily": "frontend",
        },
    },
]
DEFAULT_RANKING_POLICY = {
    "maxRecommendations": 2,
    "requireCandidateJobIds": True,
    "deduplicateByJobId": True,
    "backendOwnsHydration": True,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def now_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def sanitize_url(value: str | None) -> str | None:
    if not value:
        return value
    return value.split("?", 1)[0]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in REDACTED_KEYS or any(secret in key for secret in ("TOKEN", "SECRET", "PASSWORD")):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if value.startswith("Bearer "):
            return "Bearer <redacted>"
        return value
    return value


def private_leak_tokens(payload: Any) -> list[str]:
    text = json.dumps(redact(payload), sort_keys=True)
    return [token for token in PRIVATE_FIELD_TOKENS if token in text]


def multipart_body(fields: list[tuple[str, str]], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----bisakerja-phase48-staging-shadow"
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


def request_json(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, data: bytes | None = None, timeout: float = 60) -> dict[str, Any]:
    started = perf_counter()
    req = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:  # noqa: S310 - operator-provided staging URL
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
            status_code = response.status
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw) if raw else {"error": str(exc)}
        except json.JSONDecodeError:
            body = {"error": raw or str(exc)}
        status_code = exc.code
    except (URLError, TimeoutError) as exc:
        body = {"error": str(exc)}
        status_code = 0
    return {
        "url": sanitize_url(url),
        "status_code": status_code,
        "latency_ms": now_ms(started),
        "body": redact(body),
        "private_leak_tokens": private_leak_tokens(body),
    }


def bearer_headers(token: str | None, *, accept_json: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {}
    if accept_json:
        headers["accept"] = "application/json"
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


def probe_model_api(base_url: str, token: str | None, timeout: float) -> dict[str, Any]:
    root = base_url.rstrip("/")
    probes = {
        "live": request_json(f"{root}/live", timeout=timeout),
        "health": request_json(f"{root}/health", timeout=timeout),
        "ready": request_json(f"{root}/ready", timeout=timeout),
        "model_info": request_json(f"{root}/model-info", headers=bearer_headers(token), timeout=timeout),
    }
    ready_body = probes["ready"].get("body", {})
    info_body = probes["model_info"].get("body", {})
    return {
        "base_url": sanitize_url(base_url),
        "probes": probes,
        "checks": {
            "live_200": probes["live"]["status_code"] == 200,
            "health_200": probes["health"]["status_code"] == 200,
            "ready_200": probes["ready"]["status_code"] == 200,
            "ready_true": isinstance(ready_body, dict) and ready_body.get("ready") is True,
            "model_info_200": probes["model_info"]["status_code"] == 200,
            "model_info_has_embedding_policy": isinstance(info_body, dict) and isinstance(info_body.get("embeddingPolicy"), dict),
            "no_private_leakage": not any(probe["private_leak_tokens"] for probe in probes.values()),
        },
    }


def direct_model_smoke(base_url: str, token: str, fixture_pdf: Path, request_id: str, timeout: float) -> dict[str, Any]:
    pdf_bytes = fixture_pdf.read_bytes()
    fields = [
        ("requestId", request_id),
        ("language", "en"),
        ("inputMode", "UPLOAD"),
        ("compareSource", "JOB_SEARCH"),
        ("jobRoles", "Backend Developer"),
        ("jobCandidates", json.dumps(DEFAULT_JOB_CANDIDATES, sort_keys=True)),
        ("rankingPolicy", json.dumps(DEFAULT_RANKING_POLICY, sort_keys=True)),
    ]
    body, boundary = multipart_body(fields, "cvFile", fixture_pdf.name, pdf_bytes)
    headers = bearer_headers(token)
    headers["content-type"] = f"multipart/form-data; boundary={boundary}"
    headers["x-model-api-include-observability"] = "true"
    result = request_json(
        f"{base_url.rstrip('/')}/internal/model/cv-analysis",
        method="POST",
        headers=headers,
        data=body,
        timeout=timeout,
    )
    checks = direct_model_checks(result)
    return {"fixture": rel(fixture_pdf), "result": result, "checks": checks}


def backend_public_smoke(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.backend_api_url or not args.user_access_token:
        return None
    pdf_bytes = Path(args.fixture_pdf).read_bytes()
    fields = [
        ("language", "en"),
        ("inputMode", "UPLOAD"),
        ("compareSource", args.compare_source),
        ("persistResult", "true" if args.persist_result else "false"),
        ("jobRoles", "Backend Developer"),
    ]
    if args.direct_job_id:
        fields.append(("directJobId", args.direct_job_id))
    body, boundary = multipart_body(fields, "cvFile", Path(args.fixture_pdf).name, pdf_bytes)
    headers = bearer_headers(args.user_access_token)
    headers["content-type"] = f"multipart/form-data; boundary={boundary}"
    if args.request_id:
        headers["x-request-id"] = args.request_id
    result = request_json(
        f"{args.backend_api_url.rstrip('/')}/api/v1/ai/cv-analyzer",
        method="POST",
        headers=headers,
        data=body,
        timeout=args.timeout,
    )
    return {
        "backend_api_url": sanitize_url(args.backend_api_url),
        "compare_source": args.compare_source,
        "persist_result": args.persist_result,
        "result": result,
        "checks": backend_public_checks(result, args.latency_budget_ms),
    }


def get_path(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_analysis_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("body"), dict):
        return extract_analysis_payload(payload["body"])
    if isinstance(payload.get("result"), dict):
        return extract_analysis_payload(payload["result"])
    if payload.get("schemaVersion") in {"model-core-cv-analysis-v1", "cv-analysis-v2"}:
        return payload
    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("analysisResult"), dict):
            return data["analysisResult"]
        if data.get("schemaVersion") in {"model-core-cv-analysis-v1", "cv-analysis-v2"}:
            return data
    return payload


def recommendations_from_analysis(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    direct = get_path(analysis, "candidateReranking", "recommendations")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    public = analysis.get("jobRecommendations")
    if isinstance(public, list):
        return [item for item in public if isinstance(item, dict)]
    return []


def score_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def top_job_id(recommendation: dict[str, Any]) -> str | None:
    for key in ("jobId", "id"):
        value = recommendation.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def recommendation_score(recommendation: dict[str, Any]) -> float | None:
    for key in ("matchScore", "score"):
        value = score_value(recommendation.get(key))
        if value is not None:
            return value
    return None


def compare_shadow_outputs(old_payload: Any, new_payload: Any, *, score_delta_threshold: float = 5.0) -> dict[str, Any]:
    old_analysis = extract_analysis_payload(old_payload)
    new_analysis = extract_analysis_payload(new_payload)
    old_recs = recommendations_from_analysis(old_analysis)
    new_recs = recommendations_from_analysis(new_analysis)
    old_top = old_recs[0] if old_recs else {}
    new_top = new_recs[0] if new_recs else {}
    old_score = score_value(get_path(old_analysis, "jobFitAlignment", "score"))
    new_score = score_value(get_path(new_analysis, "jobFitAlignment", "score"))
    score_delta = None if old_score is None or new_score is None else round(new_score - old_score, 3)
    top_old = top_job_id(old_top)
    top_new = top_job_id(new_top)
    top_score_old = recommendation_score(old_top)
    top_score_new = recommendation_score(new_top)
    top_score_delta = None if top_score_old is None or top_score_new is None else round(top_score_new - top_score_old, 3)
    ats_old = score_value(get_path(old_analysis, "atsFriendliness", "score"))
    ats_new = score_value(get_path(new_analysis, "atsFriendliness", "score"))
    ats_delta = None if ats_old is None or ats_new is None else round(ats_new - ats_old, 3)
    old_rank = [top_job_id(item) for item in old_recs]
    new_rank = [top_job_id(item) for item in new_recs]
    flags = {
        "score_delta_above_threshold": score_delta is not None and abs(score_delta) > score_delta_threshold,
        "top_recommendation_rank_swap": bool(top_old and top_new and top_old != top_new),
        "top_score_delta_above_threshold": top_score_delta is not None and abs(top_score_delta) > score_delta_threshold,
        "ats_score_delta_above_threshold": ats_delta is not None and abs(ats_delta) > score_delta_threshold,
        "recommendation_count_changed": len(old_recs) != len(new_recs),
        "private_field_leakage": bool(private_leak_tokens(old_payload) or private_leak_tokens(new_payload)),
    }
    return {
        "thresholds": {"score_delta_points": score_delta_threshold},
        "old": {
            "schemaVersion": old_analysis.get("schemaVersion"),
            "jobFitScore": old_score,
            "atsScore": ats_old,
            "rankedJobIds": old_rank,
            "topJobId": top_old,
            "topMatchScore": top_score_old,
            "matchedSkills": get_path(old_analysis, "jobFitAlignment", "matchedSkills") or old_top.get("matchedSkills"),
            "missingSkills": get_path(old_analysis, "jobFitAlignment", "missingSkills") or old_top.get("missingSkills"),
        },
        "new": {
            "schemaVersion": new_analysis.get("schemaVersion"),
            "jobFitScore": new_score,
            "atsScore": ats_new,
            "rankedJobIds": new_rank,
            "topJobId": top_new,
            "topMatchScore": top_score_new,
            "matchedSkills": get_path(new_analysis, "jobFitAlignment", "matchedSkills") or new_top.get("matchedSkills"),
            "missingSkills": get_path(new_analysis, "jobFitAlignment", "missingSkills") or new_top.get("missingSkills"),
        },
        "deltas": {"jobFitScore": score_delta, "topMatchScore": top_score_delta, "atsScore": ats_delta},
        "flags": flags,
        "decision": "review" if any(flags.values()) else "pass",
    }


def direct_model_checks(result: dict[str, Any]) -> dict[str, bool]:
    body = result.get("body", {})
    analysis = extract_analysis_payload(body)
    recommendations = recommendations_from_analysis(analysis)
    observability = analysis.get("observability") if isinstance(analysis.get("observability"), dict) else {}
    return {
        "http_200": result.get("status_code") == 200,
        "schema_model_core": analysis.get("schemaVersion") == "model-core-cv-analysis-v1",
        "jobfit_score_present": score_value(get_path(analysis, "jobFitAlignment", "score")) is not None,
        "recommendations_present": bool(recommendations),
        "observability_present": isinstance(observability, dict) and "totalLatencyMs" in observability,
        "no_backend_owned_fields": not any(token in json.dumps(analysis, sort_keys=True) for token in ["topActionables", "sectionReviews", "companyName", "nextStep"]),
        "no_private_leakage": not result.get("private_leak_tokens"),
    }


def backend_public_checks(result: dict[str, Any], latency_budget_ms: float) -> dict[str, bool]:
    body = result.get("body", {})
    analysis = get_path(body, "data", "analysisResult")
    recommendations = analysis.get("jobRecommendations") if isinstance(analysis, dict) else None
    generated_cv = analysis.get("generatedCv") if isinstance(analysis, dict) else None
    model = analysis.get("model") if isinstance(analysis, dict) else None
    return {
        "http_200": result.get("status_code") == 200,
        "public_envelope_shape": isinstance(body, dict) and all(key in body for key in ["success", "message", "data", "meta"]),
        "schema_version_cv_analysis_v2": isinstance(analysis, dict) and analysis.get("schemaVersion") == "cv-analysis-v2",
        "hydrated_recommendations_max_5": isinstance(recommendations, list) and len(recommendations) <= 5,
        "generated_cv_unavailable_or_object": isinstance(generated_cv, dict),
        "model_metadata_present": isinstance(model, dict) and bool(model.get("name")) and bool(model.get("version")),
        "latency_budget_met": float(result.get("latency_ms", 0)) <= latency_budget_ms,
        "no_private_leakage": not result.get("private_leak_tokens"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    fixture = Path(args.fixture_pdf)
    report: dict[str, Any] = {
        "schema_version": "phase-48-backend-staging-shadow-report-v1",
        "phase_id": "phase_48_backend_staging_integration_shadow_rollback",
        "generated_at": now_utc(),
        "fixture": rel(fixture),
        "thresholds": {
            "score_delta_points": args.score_delta_threshold,
            "latency_budget_ms": args.latency_budget_ms,
            "first_inference_budget_ms": 30000,
            "warm_inference_budget_ms": 5000,
        },
        "expected_new_artifact": {
            "artifact_root": PHASE46_ARTIFACT_ROOT,
            "embedding_model": PHASE46_EMBEDDING_MODEL,
        },
        "rollback_artifact": {
            "artifact_root": PHASE25_ARTIFACT_ROOT,
            "embedding_model": PHASE25_EMBEDDING_MODEL,
        },
    }
    if args.new_model_api_url:
        report["new_model_api"] = probe_model_api(args.new_model_api_url, args.model_api_token, args.timeout)
        report["new_direct_model_smoke"] = direct_model_smoke(
            args.new_model_api_url,
            args.model_api_token,
            fixture,
            args.request_id + "_new_direct",
            args.timeout,
        )
    if args.old_model_api_url:
        report["old_model_api"] = probe_model_api(args.old_model_api_url, args.old_model_api_token or args.model_api_token, args.timeout)
        report["old_direct_model_smoke"] = direct_model_smoke(
            args.old_model_api_url,
            args.old_model_api_token or args.model_api_token,
            fixture,
            args.request_id + "_old_direct",
            args.timeout,
        )
    if "old_direct_model_smoke" in report and "new_direct_model_smoke" in report:
        report["shadow_comparison"] = compare_shadow_outputs(
            report["old_direct_model_smoke"]["result"],
            report["new_direct_model_smoke"]["result"],
            score_delta_threshold=args.score_delta_threshold,
        )
    report["backend_public_smoke"] = backend_public_smoke(args)
    check_groups = [
        report.get("new_model_api", {}).get("checks", {}),
        report.get("new_direct_model_smoke", {}).get("checks", {}),
    ]
    if report.get("backend_public_smoke") is not None:
        check_groups.append(report["backend_public_smoke"].get("checks", {}))
    if report.get("shadow_comparison") is not None:
        check_groups.append({"shadow_comparison_passes_or_reviewed": report["shadow_comparison"].get("decision") == "pass"})
    failed = [name for group in check_groups for name, passed in group.items() if not passed]
    missing_live_scope = [
        name
        for name, enabled in {
            "new_model_api_url": bool(args.new_model_api_url),
            "backend_api_url_and_user_access_token": bool(args.backend_api_url and args.user_access_token),
            "old_model_api_url_for_shadow_compare": bool(args.old_model_api_url),
        }.items()
        if not enabled
    ]
    report["failure_behavior_plan"] = {
        "model_not_ready": "Expect 503 MODEL_NOT_READY from Model API and Backend graceful failure/fallback.",
        "timeout": "Keep Backend timeout above MODEL_API_TIMEOUT_MS; block rollout when warm inference exceeds budget.",
        "invalid_pdf": "Expect 4xx validation error; no persistence of raw/private fields in public response.",
        "empty_candidates": "Expect contract validation or empty recommendation-safe response, never invented jobs.",
        "invalid_token": "Expect 401 from /model-info and protected inference routes.",
    }
    report["rollback_plan"] = {
        "artifact_env": {
            "MODEL_API_ARTIFACT_ROOT": PHASE25_ARTIFACT_ROOT,
            "MODEL_API_EXPECTED_EMBEDDING_MODEL": PHASE25_EMBEDDING_MODEL,
        },
        "commands": [
            "cd /opt/bisakerja-model-api",
            "edit .env.production so MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery",
            "edit .env.production so MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2",
            "MODEL_API_ENV_FILE=.env.production COMPOSE_PROJECT_NAME=bisakerja-model-api docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api",
            "curl -fsS http://127.0.0.1:3004/ready",
            "curl -fsS -H 'Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}' http://127.0.0.1:3004/model-info",
        ],
    }
    report["missing_live_scope"] = missing_live_scope
    report["failed_checks"] = failed
    report["final_decision"] = "pass" if not failed and not missing_live_scope else ("fail" if failed else "partial")
    return redact(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-model-api-url", default=None, help="staging Model API URL running Phase 46 multilingual-E5-small")
    parser.add_argument("--old-model-api-url", default=None, help="rollback/baseline Model API URL running Phase 25 E5-base")
    parser.add_argument("--model-api-token", default="", help="Model API internal service token")
    parser.add_argument("--old-model-api-token", default="", help="optional token for old Model API URL")
    parser.add_argument("--backend-api-url", default=None)
    parser.add_argument("--user-access-token", default="")
    parser.add_argument("--fixture-pdf", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--compare-source", choices=["JOB_SEARCH", "DIRECT_JOB_DETAIL", "BOOKMARK"], default="JOB_SEARCH")
    parser.add_argument("--direct-job-id", default=None)
    parser.add_argument("--persist-result", action="store_true")
    parser.add_argument("--request-id", default="phase48_staging_shadow")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--latency-budget-ms", type=float, default=5000)
    parser.add_argument("--score-delta-threshold", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["final_decision"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
