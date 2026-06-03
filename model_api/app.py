"""FastAPI application factory for Model API serving.

The factory keeps FastAPI optional until serving dependencies are pinned in the
Phase 26 packaging step. Importing this module must remain lightweight for tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from .artifacts import ArtifactVerificationReport, verify_runtime_artifacts
from .config import RuntimeConfig
from .errors import (
    ArtifactError,
    ContractValidationError,
    FeatureBuildError,
    InferenceTimeoutError,
    ModelApiError,
    ModelLoadError,
    ModelNotReadyError,
    UnsupportedArtifactVersionError,
)
from .features import (
    PHASE25_FEATURE_ORDER,
    TensorFlowFeatureConfig,
    TextEmbeddingBackend,
    build_feature_vectors_for_request,
    normalized_skill_set,
)
from .inference import InferenceService, RuntimeState, ScoreCalibrationPolicy, utc_now_iso
from .schemas import (
    MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
    MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
    AtsFriendlinessCore,
    CvAnalysisModelCoreRequest,
    JobFitAlignmentCore,
    OverallImpressionCore,
    ScoreSignal,
    parse_cv_analysis_model_core_request,
)
from .schemas import ModelIdentity
from .validators import validate_model_core_payload


def _model_identity_payload(identity: ModelIdentity | None) -> dict[str, object] | None:
    if identity is None:
        return None
    return asdict(identity)


def _runtime_state_payload(state: RuntimeState) -> dict[str, object]:
    return {
        "ready": state.ready,
        "message": state.message,
        "artifactManifestPhase": state.artifact_manifest_phase,
        "loadedAt": state.loaded_at,
        "errorCode": state.error_code,
        "model": _model_identity_payload(state.model_identity),
    }


def _artifact_verification_payload(report: ArtifactVerificationReport) -> dict[str, object]:
    return {
        "manifestPhase": report.manifest.phase_id,
        "runtimeArtifactIds": report.runtime_artifact_ids,
        "artifactHashes": report.artifact_hashes,
        "artifactSizes": report.artifact_sizes,
    }


def _score_signal_payload(signal: ScoreSignal) -> dict[str, object]:
    return {"key": signal.key, "label": signal.label, "value": signal.value}


def _recommendation_payload(recommendation) -> dict[str, object]:
    return {
        "jobId": recommendation.jobId,
        "matchScore": recommendation.matchScore,
        "matchLevel": recommendation.matchLevel,
        "rankingSignals": [_score_signal_payload(signal) for signal in recommendation.rankingSignals],
        "matchedSkills": list(recommendation.matchedSkills),
        "missingSkills": list(recommendation.missingSkills),
    }


def _candidate_skill_evidence(request: CvAnalysisModelCoreRequest) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not request.jobCandidates:
        return (), ()
    profile_skills = normalized_skill_set(request.profile.normalizedSkills)
    first_candidate_skills = normalized_skill_set(
        (*request.jobCandidates[0].model_scoring_input.requiredSkills, *request.jobCandidates[0].model_scoring_input.requirements)
    )
    matched = tuple(sorted(profile_skills & first_candidate_skills))
    missing = tuple(sorted(first_candidate_skills - profile_skills))
    return matched, missing


def build_cv_analysis_response_payload(
    request: CvAnalysisModelCoreRequest,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
    timeout_ms: int | None = None,
) -> dict[str, object]:
    """Build validated model-core CV-analysis response for HTTP route/tests."""

    state = service.state
    if state.model_identity is None:
        service.require_ready()
        state = service.state
    if state.model_identity is None:
        raise ModelNotReadyError("TensorFlow model identity is not available")

    candidate_vectors = build_feature_vectors_for_request(
        request,
        feature_config=feature_config,
        embedding_backend=embedding_backend,
        environment=environment,
    )
    recommendations = service.predict_recommendations(
        tuple(vector.normalized for vector in candidate_vectors),
        max_recommendations=request.maxRecommendations,
        calibration_policy=calibration_policy,
        timeout_ms=timeout_ms,
    )
    top_score = recommendations[0].matchScore if recommendations else 0
    matched_skills, missing_skills = _candidate_skill_evidence(request)
    profile_evidence_keys = [
        key
        for key, enabled in {
            "cvText": bool(request.profile.cvText.strip()),
            "profileText": bool(request.profile.profileText.strip()),
            "normalizedSkills": bool(request.profile.normalizedSkills),
            "targetRoles": bool(request.profile.targetRoles),
            "detectedCvSectionNames": bool(request.profile.detectedCvSectionNames),
        }.items()
        if enabled
    ]
    job_fit = JobFitAlignmentCore(
        score=top_score,
        matchedSkills=matched_skills,
        missingSkills=missing_skills,
        summarySignals=tuple(ScoreSignal(key=feature_name) for feature_name in PHASE25_FEATURE_ORDER),
        confidenceNotes=("model-core job-fit score from Phase 25 TensorFlow candidate scorer",),
    )
    ats_score = 85 if request.profile.detectedCvSectionNames else 70
    ats = AtsFriendlinessCore(
        score=ats_score,
        detectedIssues=() if request.profile.detectedCvSectionNames else ("section evidence not provided",),
        evidence={"evidenceKeys": profile_evidence_keys, "placeholderPolicy": "deterministic_model_core_evidence"},
        fallback=True,
    )
    overall_score = round((job_fit.score + ats.score) / 2)
    overall = OverallImpressionCore(
        score=overall_score,
        summary="Model-core evidence prepared for backend GenAI wrapper.",
        evidenceKeys=("jobFitAlignment", "atsFriendliness", "candidateReranking"),
        confidenceNotes=("No external GenAI call is made by Model API core inference.",),
    )
    payload: dict[str, object] = {
        "requestId": request.requestId,
        "schemaVersion": MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
        "language": request.language,
        "jobFitAlignment": {
            "score": job_fit.score,
            "matchedSkills": list(job_fit.matchedSkills),
            "missingSkills": list(job_fit.missingSkills),
            "summarySignals": [_score_signal_payload(signal) for signal in job_fit.summarySignals],
            "confidenceNotes": list(job_fit.confidenceNotes),
        },
        "atsFriendliness": {
            "score": ats.score,
            "detectedIssues": list(ats.detectedIssues),
            "evidence": dict(ats.evidence),
            "fallback": ats.fallback,
        },
        "overallImpression": {
            "score": overall.score,
            "summary": overall.summary,
            "evidenceKeys": list(overall.evidenceKeys),
            "confidenceNotes": list(overall.confidenceNotes),
        },
        "candidateReranking": {
            "requestId": request.requestId,
            "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
            "candidateSetId": request.requestId,
            "language": request.language,
            "recommendations": [_recommendation_payload(recommendation) for recommendation in recommendations],
            "model": _model_identity_payload(state.model_identity),
            "rankedAt": utc_now_iso(),
        },
        "model": _model_identity_payload(state.model_identity),
        "analyzedAt": utc_now_iso(),
    }
    validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
    return payload


def create_app(
    config: RuntimeConfig | None = None,
    service: InferenceService | None = None,
    embedding_backend: TextEmbeddingBackend | None = None,
):
    """Create HTTP app, verify artifacts, and load model during lifespan startup.

    Raises a clear error when FastAPI is not installed instead of failing during
    package import.
    """

    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError("FastAPI dependency missing; install serving requirements after Step 26.13") from exc

    runtime_config = config or RuntimeConfig.from_env()
    artifact_report = verify_runtime_artifacts(runtime_config.artifact_paths)
    feature_config = TensorFlowFeatureConfig.from_path(runtime_config.artifact_paths.tensorflow_feature_config_path)
    calibration_policy = ScoreCalibrationPolicy.from_path(runtime_config.artifact_paths.score_calibration_path)
    inference_service = service or InferenceService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.artifact_report = artifact_report
        app.state.inference_service = inference_service
        app.state.feature_config = feature_config
        app.state.calibration_policy = calibration_policy
        inference_service.load_once(runtime_config.artifact_paths, artifact_report)
        yield

    app = FastAPI(title="Bisakerja Model API", version="0.1.0-phase26.4", lifespan=lifespan)

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(_request: Request, exc: ModelNotReadyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "TensorFlow model is not ready",
                "data": None,
                "error": exc.to_error_payload(),
            },
        )

    @app.exception_handler(ContractValidationError)
    async def contract_validation_error_handler(_request: Request, exc: ContractValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": "Invalid model-core request/response", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(FeatureBuildError)
    async def feature_build_error_handler(_request: Request, exc: FeatureBuildError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": "Invalid feature inputs", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(InferenceTimeoutError)
    async def inference_timeout_handler(_request: Request, exc: InferenceTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"success": False, "message": "TensorFlow inference timeout", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(_request: Request, exc: ModelLoadError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": "TensorFlow inference failed", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(UnsupportedArtifactVersionError)
    async def unsupported_artifact_version_handler(_request: Request, exc: UnsupportedArtifactVersionError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Unsupported artifact version", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ArtifactError)
    async def artifact_error_handler(_request: Request, exc: ArtifactError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Artifact verification failed", "data": None, "error": exc.to_error_payload()},
        )

    @app.exception_handler(ModelApiError)
    async def model_api_error_handler(_request: Request, exc: ModelApiError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Model API error", "data": None, "error": exc.to_error_payload()},
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        state = inference_service.state
        identity = state.model_identity
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "ready": state.ready,
            "message": state.message,
            "modelName": None if identity is None else identity.name,
            "modelVersion": None if identity is None else identity.version,
            "artifactHash": None if identity is None else identity.artifact_sha256,
        }

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        state = inference_service.state
        return {
            "ready": state.ready,
            "readiness": _runtime_state_payload(state),
            "model": _model_identity_payload(state.model_identity),
            "artifacts": runtime_config.artifact_paths.as_dict(),
            "artifactVerification": _artifact_verification_payload(artifact_report),
            "openrouter": {
                "baseUrl": runtime_config.openrouter.base_url,
                "modelsUrl": runtime_config.openrouter.models_url,
                "model": runtime_config.openrouter.model,
                "enabled": runtime_config.openrouter.enabled,
            },
        }

    @app.post("/inference/cv-analysis")
    def cv_analysis(payload: Any = Body(...)) -> dict[str, object]:
        request = parse_cv_analysis_model_core_request(payload)
        data = build_cv_analysis_response_payload(
            request,
            service=inference_service,
            feature_config=feature_config,
            calibration_policy=calibration_policy,
            embedding_backend=embedding_backend,
            environment=runtime_config.environment,
            timeout_ms=runtime_config.timeout_ms,
        )
        return {"success": True, "message": "Model-core inference completed", "data": data, "error": None}

    return app
