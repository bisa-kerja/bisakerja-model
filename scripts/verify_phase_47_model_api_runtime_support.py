#!/usr/bin/env python3
"""Verify Model API runtime support for versioned embedding artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows portability
    resource = None

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_api.app import build_cv_analysis_response_payload, validate_runtime_embedding_contract  # noqa: E402
from model_api.artifacts import verify_runtime_artifacts  # noqa: E402
from model_api.config import ArtifactPaths  # noqa: E402
from model_api.features import TensorFlowFeatureConfig  # noqa: E402
from model_api.inference import InferenceService, RuntimeState, ScoreCalibrationPolicy  # noqa: E402
from model_api.schemas import CandidateJobInput, CandidateScoringInput, CvAnalysisModelCoreRequest, ModelArtifactIdentity, ModelIdentity, SanitizedProfileInput  # noqa: E402

REPORTS = ROOT / "reports"
REPORT_JSON_PATH = REPORTS / "phase_47_model_api_runtime_support.json"
REPORT_MD_PATH = REPORTS / "phase_47_model_api_runtime_support.md"
PHASE25_ROOT = "artifacts/phase_25_tensorflow_training_delivery"
PHASE46_ROOT = "artifacts/phase_46_calibration_model_card_manifest_handoff_refresh"
PHASE25_ID = "phase_25_tensorflow_training_delivery"
PHASE46_ID = "phase_46_calibration_model_card_manifest_handoff_refresh"
SCHEMA_VERSION = "phase-47-model-api-runtime-support-v1"


class FakeModel:
    name = "phase47_performance_smoke_fake_model"

    def predict(self, values, verbose=0):
        return [[0.72] for _ in values]


class FakeEmbeddingBackend:
    backend_name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    return result.stdout.strip() or None


def git_state() -> dict[str, Any]:
    porcelain = run_text(["git", "status", "--porcelain"]) or ""
    dirty_lines = [line for line in porcelain.splitlines() if line]
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty_file_count": len(dirty_lines),
        "dirty_sample": dirty_lines[:40],
    }


def artifact_support(root: str) -> dict[str, Any]:
    started = perf_counter()
    paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": root})
    report = verify_runtime_artifacts(paths)
    feature_config = TensorFlowFeatureConfig.from_path(paths.tensorflow_feature_config_path)
    contract = validate_runtime_embedding_contract(paths=paths, artifact_report=report, feature_config=feature_config)
    return {
        "artifactRoot": root,
        "artifactPhase": report.manifest.phase_id,
        "manifestSchemaVersion": report.manifest.schema_version,
        "modelPath": str(paths.model_path),
        "runtimeArtifactIds": list(report.runtime_artifact_ids),
        "artifactHash": report.artifact_hashes.get("final_keras_model"),
        "embeddingPolicy": contract.as_dict(),
        "verifyMs": round((perf_counter() - started) * 1000, 3),
    }


def smoke_request() -> CvAnalysisModelCoreRequest:
    return CvAnalysisModelCoreRequest(
        requestId="phase47-performance-smoke",
        inputVersion="cv-analyzer-v1",
        language="en",
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=SanitizedProfileInput(
            cvText="Backend engineer with Python SQL REST API experience.",
            profileText="Backend engineer Python SQL.",
            targetRoles=("Backend Engineer",),
            normalizedSkills=("python", "sql", "rest api"),
            detectedCvSectionNames=("summary", "skills", "experience"),
        ),
        jobCandidates=(
            CandidateJobInput(
                jobId="phase47-job-1",
                scoringInput=CandidateScoringInput(
                    titleText="Backend Engineer",
                    requirementSummary="Build REST APIs with Python and SQL.",
                    requiredSkills=("python", "sql", "rest api"),
                    roleFamily="backend",
                ),
            ),
        ),
        maxRecommendations=1,
    )


def performance_smoke(phase46: dict[str, Any]) -> dict[str, Any]:
    identity = ModelIdentity(
        name="phase47",
        version="performance-smoke",
        artifact=ModelArtifactIdentity(path=str(phase46["modelPath"]), sha256=phase46["artifactHash"]),
        artifact_phase=str(phase46["artifactPhase"]),
        embedding_model=str(phase46["embeddingPolicy"]["embeddingModel"]),
    )
    service = InferenceService(
        model=FakeModel(),
        state=RuntimeState(ready=True, model_identity=identity, artifact_manifest_phase=str(phase46["artifactPhase"]), message="ready"),
    )
    paths = ArtifactPaths.from_env({"MODEL_API_ARTIFACT_ROOT": PHASE46_ROOT})
    feature_config = TensorFlowFeatureConfig.from_path(paths.tensorflow_feature_config_path)
    backend = FakeEmbeddingBackend(str(phase46["embeddingPolicy"]["embeddingModel"]))
    request = smoke_request()

    first_started = perf_counter()
    first = build_cv_analysis_response_payload(
        request,
        service=service,
        feature_config=feature_config,
        calibration_policy=ScoreCalibrationPolicy(),
        embedding_backend=backend,
        environment="test",
        timeout_ms=30_000,
        include_observability=True,
    )
    first_ms = round((perf_counter() - first_started) * 1000, 3)

    warm_started = perf_counter()
    warm = build_cv_analysis_response_payload(
        request,
        service=service,
        feature_config=feature_config,
        calibration_policy=ScoreCalibrationPolicy(),
        embedding_backend=backend,
        environment="test",
        timeout_ms=30_000,
        include_observability=True,
    )
    warm_ms = round((perf_counter() - warm_started) * 1000, 3)

    if resource is None:
        rss_bytes = 0
    else:
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    return {
        "startupArtifactVerifyMs": phase46["verifyMs"],
        "readyCheckSimulated": True,
        "firstInferenceMs": first_ms,
        "warmInferenceMs": warm_ms,
        "embeddingBackendCalls": backend.calls,
        "maxRssMb": round(rss_bytes / (1024 * 1024), 3),
        "cacheBehavior": "fake sentence-transformers backend reused in-process",
        "responseContractStable": set(first) == set(warm),
        "backendOwnedFieldsLeaked": any(key in json.dumps(first) for key in ["topActionables", "companyName", "reason", "nextStep"]),
    }


def build_report() -> dict[str, Any]:
    phase25 = artifact_support(PHASE25_ROOT)
    phase46 = artifact_support(PHASE46_ROOT)
    smoke = performance_smoke(phase46)
    checks = {
        "phase25RollbackArtifactVerifies": phase25["artifactPhase"] == PHASE25_ID,
        "phase46ArtifactVerifies": phase46["artifactPhase"] == PHASE46_ID,
        "phase46EmbeddingDeclared": phase46["embeddingPolicy"]["embeddingModel"] == "intfloat/multilingual-e5-small",
        "phase25EmbeddingDeclared": phase25["embeddingPolicy"]["embeddingModel"] == "intfloat/e5-base-v2",
        "prefixPolicyPreserved": phase46["embeddingPolicy"]["profilePrefix"] == "query:" and phase46["embeddingPolicy"]["jobPrefix"] == "passage:",
        "artifactRootSwitchesModelPath": phase46["modelPath"].endswith("selected_jobfit_tf_phase46_multilingual_e5_small.keras"),
        "performanceSmokeRecorded": smoke["firstInferenceMs"] >= 0 and smoke["warmInferenceMs"] >= 0,
        "responseContractStable": bool(smoke["responseContractStable"]),
        "backendOwnedFieldsDoNotLeak": not bool(smoke["backendOwnedFieldsLeaked"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_utc(),
        "status": "complete" if all(checks.values()) else "blocked",
        "checks": checks,
        "artifact_support": {
            PHASE25_ID: phase25,
            PHASE46_ID: phase46,
        },
        "performance_smoke": smoke,
        "git": git_state(),
    }


def markdown_report(report: dict[str, Any]) -> str:
    checks = "\n".join(f"- {'PASS' if passed else 'FAIL'} {name}" for name, passed in report["checks"].items())
    return "\n".join(
        [
            "# Phase 47 Model API Runtime Support",
            "",
            f"Status: {report['status']}",
            f"Generated at: {report['generated_at']}",
            "",
            "## Checks",
            checks,
            "",
            "## Runtime Metadata",
            f"- Phase 25 embedding: {report['artifact_support'][PHASE25_ID]['embeddingPolicy']['embeddingModel']}",
            f"- Phase 46 embedding: {report['artifact_support'][PHASE46_ID]['embeddingPolicy']['embeddingModel']}",
            f"- Phase 46 model path: {report['artifact_support'][PHASE46_ID]['modelPath']}",
            "",
            "## Performance Smoke",
            f"- Startup artifact verify: {report['performance_smoke']['startupArtifactVerifyMs']} ms",
            f"- First inference: {report['performance_smoke']['firstInferenceMs']} ms",
            f"- Warm inference: {report['performance_smoke']['warmInferenceMs']} ms",
            f"- Max RSS: {report['performance_smoke']['maxRssMb']} MB",
            "",
        ]
    )


def write_all() -> dict[str, Any]:
    report = build_report()
    write_json(REPORT_JSON_PATH, report)
    REPORT_MD_PATH.write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> int:
    report = write_all()
    print(json.dumps({"status": report["status"], "report": rel(REPORT_JSON_PATH)}, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
