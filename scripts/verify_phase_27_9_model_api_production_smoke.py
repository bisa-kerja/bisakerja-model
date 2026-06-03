#!/usr/bin/env python3
"""Verify Phase 27.9 real Model API production-smoke readiness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_JSON_PATH = ROOT / "reports/phase_27_9_model_api_production_smoke.json"
REPORT_MD_PATH = ROOT / "reports/phase_27_9_model_api_production_smoke.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
HANDOFF_FIXTURES_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json"
KERAS_SMOKE_SCRIPT = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/registered_custom_objects_smoke.py"
KERAS_MODEL_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras"
FEATURE_MATRIX_PATH = ROOT / "artifacts/phase_25_tensorflow_training_delivery/tensorflow_training_features_v1.npz"
KERAS_SMOKE_OUTPUT_PATH = ROOT / "reports/phase_27_9_registered_custom_objects_smoke.json"

REQUIRED_MODULES = ("fastapi", "tensorflow", "keras", "sentence_transformers", "numpy", "uvicorn")
PHASE26_TEST_COMMAND = [sys.executable, "-m", "unittest", "tests.model_api.test_phase_26_layout"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
    started = perf_counter()
    try:
        proc = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True, timeout=timeout)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return {
            "command": " ".join(command),
            "returncode": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": proc.stdout.splitlines()[-40:],
            "stderr_tail": proc.stderr.splitlines()[-40:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-40:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-40:] if isinstance(exc.stderr, str) else [],
        }


def parse_unittest_result(result: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(result.get("stdout_tail", []) + result.get("stderr_tail", []))
    skipped = 0
    if "skipped=" in text:
        marker = text.split("skipped=", 1)[1]
        digits = []
        for char in marker:
            if char.isdigit():
                digits.append(char)
            else:
                break
        skipped = int("".join(digits) or "0")
    tests_run = None
    if "Ran " in text and " tests" in text:
        try:
            tests_run = int(text.split("Ran ", 1)[1].split(" tests", 1)[0])
        except ValueError:
            tests_run = None
    return {"passed": result.get("returncode") == 0, "skipped": skipped, "tests_run": tests_run}


def requirement_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    if not REQUIREMENTS_PATH.exists():
        return pins
    for raw in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.lower()] = version
    return pins


def fake_not_allowed_live_smoke_possible() -> bool:
    return sys.version_info[:2] == (3, 13) and all(module_available(name) for name in REQUIRED_MODULES)


def run_live_fastapi_smoke() -> dict[str, Any]:
    """Run actual app endpoints when Python 3.13 serving deps are installed.

    This intentionally uses the real app factory and default runtime loader. It
    does not mock TensorFlow or E5; if dependencies/artifacts fail, the returned
    evidence blocks production readiness.
    """

    if not fake_not_allowed_live_smoke_possible():
        return {"status": "not_run", "reason": "Python 3.13 and all serving dependencies are required"}

    started = perf_counter()
    try:
        from fastapi.testclient import TestClient  # type: ignore[import-not-found]
        from model_api.app import create_app

        app = create_app()
        fixture_root = json.loads(HANDOFF_FIXTURES_PATH.read_text(encoding="utf-8"))["positive"]
        reranking_fixture = fixture_root["candidateRerankingCoreRequest"]
        cv_fixture = {
            "requestId": "req_phase27_9_live_cv_analysis_smoke",
            "inputVersion": "cv-analyzer-v1",
            "language": reranking_fixture["language"],
            "inputMode": "UPLOAD",
            "compareSource": "JOB_SEARCH",
            "profile": reranking_fixture["profileFeatures"],
            "jobCandidates": reranking_fixture["jobCandidates"],
            "rankingPolicy": reranking_fixture["rankingPolicy"],
            "maxRecommendations": reranking_fixture["rankingPolicy"].get("maxRecommendations", 5),
        }
        service_token = os.environ.get("MODEL_API_SERVICE_TOKEN")
        auth_headers = {"authorization": f"Bearer {service_token}"} if service_token else {}
        with TestClient(app) as client:
            health_started = perf_counter()
            health = client.get("/health")
            health_ms = round((perf_counter() - health_started) * 1000, 3)
            info_started = perf_counter()
            model_info = client.get("/model-info", headers=auth_headers)
            model_info_ms = round((perf_counter() - info_started) * 1000, 3)
            inference_started = perf_counter()
            inference = client.post("/inference/cv-analysis", json=cv_fixture, headers=auth_headers)
            inference_ms = round((perf_counter() - inference_started) * 1000, 3)
        total_ms = round((perf_counter() - started) * 1000, 3)
        return {
            "status": "PASS" if health.status_code == 200 and model_info.status_code == 200 and inference.status_code == 200 else "FAIL",
            "total_elapsed_ms": total_ms,
            "endpoint_latency_ms": {"health": health_ms, "model_info": model_info_ms, "cv_analysis": inference_ms},
            "responses": {
                "health": {"status_code": health.status_code, "body": health.json()},
                "model_info": {"status_code": model_info.status_code, "ready": model_info.json().get("ready"), "model": model_info.json().get("model")},
                "cv_analysis": {"status_code": inference.status_code, "success": inference.json().get("success"), "data_keys": sorted((inference.json().get("data") or {}).keys())},
            },
        }
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        return {"status": "FAIL", "elapsed_ms": round((perf_counter() - started) * 1000, 3), "error": repr(exc)}


def gate(name: str, passed: bool, evidence: dict[str, Any] | None = None, blocker: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"check": name, "status": "PASS" if passed else "FAIL"}
    if evidence is not None:
        payload["evidence"] = evidence
    if blocker:
        payload["blocker"] = blocker
    return payload


def build_report(run_live: bool = False, run_tests: bool = True) -> dict[str, Any]:
    pins = requirement_pins()
    module_map = {name: module_available(name) for name in REQUIRED_MODULES}
    python_313 = sys.version_info[:2] == (3, 13)
    deps_available = all(module_map.values())
    tests_result = run_command(PHASE26_TEST_COMMAND, timeout=180) if run_tests else {"returncode": None, "skipped": None, "not_run": True}
    parsed_tests = parse_unittest_result(tests_result) if run_tests else {"passed": False, "skipped": None, "tests_run": None}
    keras_smoke_command = [
        sys.executable,
        str(KERAS_SMOKE_SCRIPT),
        str(KERAS_MODEL_PATH),
        str(FEATURE_MATRIX_PATH),
        str(KERAS_SMOKE_OUTPUT_PATH),
    ]
    keras_smoke = (
        run_command(keras_smoke_command, timeout=180)
        if python_313 and deps_available and KERAS_SMOKE_SCRIPT.exists()
        else {"status": "not_run", "reason": "Python 3.13 with TensorFlow/Keras required"}
    )
    live_smoke = run_live_fastapi_smoke() if run_live else {"status": "not_run", "reason": "pass --run-live to execute real FastAPI/TensorFlow/E5 endpoint smoke"}

    gates = [
        gate("python_3_13_serving_runtime", python_313, {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}, "Current interpreter is not Python 3.13" if not python_313 else None),
        gate("root_requirements_pin_serving_runtime", {"fastapi", "tensorflow", "keras", "sentence-transformers", "numpy"}.issubset(set(pins)), {"pins": pins}),
        gate("serving_dependencies_importable", deps_available, {"modules": module_map}, "Install root requirements.txt in Python 3.13 env" if not deps_available else None),
        gate("phase26_tests_unskipped", parsed_tests["passed"] and parsed_tests["skipped"] == 0, {"summary": parsed_tests, "result": tests_result}, "Phase 26 tests must run with zero skips" if parsed_tests.get("skipped") else None),
        gate("real_keras_custom_object_loader_smoke", keras_smoke.get("returncode") == 0, keras_smoke, "Real .keras custom-object loader smoke not run or failed"),
        gate("live_fastapi_health_model_info_inference_smoke", live_smoke.get("status") == "PASS", live_smoke, "Live FastAPI/TensorFlow/E5 smoke not run or failed"),
    ]
    blockers = [item["check"] for item in gates if item["status"] != "PASS"]
    return {
        "schema_version": "phase-27-9-model-api-production-smoke-v1",
        "phase_id": "phase_27_9_model_api_production_smoke",
        "generated_at": now_iso(),
        "references": ["TODOS.md#step-27.9", "model_api/README.md", "REQUIREMENT.md#2.2", "REQUIREMENT.md#3"],
        "policy": {
            "python_runtime": "3.13.x",
            "requirements_file": "requirements.txt",
            "phase26_tests_must_be_unskipped": True,
            "real_keras_artifact_required": True,
            "real_fastapi_endpoint_smoke_required": True,
            "fake_model_or_fake_embedding_allowed_for_production_smoke": False,
        },
        "gates": gates,
        "latency": live_smoke.get("endpoint_latency_ms", {}) if isinstance(live_smoke, dict) else {},
        "blockers": blockers,
        "final_decision": "production-smoke-passed" if not blockers else "blocked",
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Phase 27.9 Model API Production Smoke",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Decision: `{report['final_decision']}`",
        "",
        "## Gates",
    ]
    for item in report["gates"]:
        lines.append(f"- {item['status']} `{item['check']}`")
    if report["blockers"]:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {item}" for item in report["blockers"])
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(run_live: bool = False) -> dict[str, Any]:
    report = build_report(run_live=run_live, run_tests=True)
    write_json(REPORT_JSON_PATH, report)
    write_markdown(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON/Markdown report")
    parser.add_argument("--run-live", action="store_true", help="attempt real FastAPI/TensorFlow/E5 endpoint smoke")
    args = parser.parse_args(argv)
    report = write_all(run_live=args.run_live) if args.write else build_report(run_live=args.run_live, run_tests=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["final_decision"] == "production-smoke-passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
