"""FastAPI application factory for Model API serving.

The factory keeps FastAPI optional until serving dependencies are pinned in the
Phase 26 packaging step. Importing this module must remain lightweight for tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from secrets import compare_digest
from time import perf_counter
from typing import Any

from .artifacts import ArtifactVerificationReport, load_json, verify_runtime_artifacts
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
    EmbeddingModelMetadata,
    SentenceTransformerE5Embedder,
    TensorFlowFeatureConfig,
    TextEmbeddingBackend,
    build_feature_vectors_for_request,
    normalized_skill_set,
    validate_e5_backend,
)
from .observability import build_safe_observability_event
from .pdf_parser import ats_score_from_pdf_evidence, normalized_skills_from_text, parse_pdf_bytes
from .inference import InferenceService, RuntimeState, ScoreCalibrationPolicy, utc_now_iso
from .schemas import (
    MAX_JOB_ROLE_CHARS,
    MAX_JOB_ROLES,
    MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
    MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
    AtsFriendlinessCore,
    CandidateRerankingCoreRequest,
    CvAnalysisModelCoreRequest,
    JobFitAlignmentCore,
    OverallImpressionCore,
    SanitizedProfileInput,
    ScoreSignal,
    parse_candidate_reranking_core_request,
    parse_cv_analysis_model_core_request,
)
from .schemas import MODEL_CORE_CV_ANALYZER_INPUT_VERSION, ModelIdentity
from .validators import validate_model_core_payload


def _model_identity_payload(identity: ModelIdentity | None, *, include_artifact: bool = False) -> dict[str, object] | None:
    if identity is None:
        return None
    payload: dict[str, object] = {"name": identity.name, "version": identity.version}
    if not include_artifact:
        return payload
    if identity.artifact is not None:
        payload["artifact"] = {
            "format": identity.artifact.format,
            "path": identity.artifact.path,
            "sha256": identity.artifact.sha256,
        }
    payload["artifactPhase"] = identity.artifact_phase
    payload["embeddingModel"] = identity.embedding_model
    return payload


def _metadata_from_payload(payload: dict[str, Any]) -> EmbeddingModelMetadata | None:
    direct = payload.get("embedding_model_metadata")
    if isinstance(direct, dict):
        return EmbeddingModelMetadata.from_mapping(direct)
    embedding_contract = payload.get("embedding_contract")
    if isinstance(embedding_contract, dict):
        return EmbeddingModelMetadata.from_mapping(embedding_contract)
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("embedding_contract"), dict):
        return EmbeddingModelMetadata.from_mapping(data["embedding_contract"])
    model = payload.get("model")
    if isinstance(model, dict) and model.get("embedding_model"):
        return EmbeddingModelMetadata.from_mapping(model)
    return None


def _manifest_embedding_metadata(report: ArtifactVerificationReport) -> EmbeddingModelMetadata | None:
    if report.manifest.embedding_model_metadata:
        return EmbeddingModelMetadata.from_mapping(report.manifest.embedding_model_metadata)
    for entry in report.manifest.required_for_inference_entries():
        if entry.embedding_model_metadata:
            return EmbeddingModelMetadata.from_mapping(entry.embedding_model_metadata)
    return None


def _assert_same_embedding_metadata(name: str, actual: EmbeddingModelMetadata, expected: EmbeddingModelMetadata) -> None:
    mismatches: list[str] = []
    if actual.embedding_model != expected.embedding_model:
        mismatches.append(f"embedding_model expected={expected.embedding_model!r} actual={actual.embedding_model!r}")
    if actual.profile_prefix != expected.profile_prefix:
        mismatches.append(f"profile_prefix expected={expected.profile_prefix!r} actual={actual.profile_prefix!r}")
    if actual.job_prefix != expected.job_prefix:
        mismatches.append(f"job_prefix expected={expected.job_prefix!r} actual={actual.job_prefix!r}")
    if actual.normalized_embeddings != expected.normalized_embeddings:
        mismatches.append(
            "normalized_embeddings "
            f"expected={expected.normalized_embeddings!r} actual={actual.normalized_embeddings!r}"
        )
    if mismatches:
        raise ArtifactError(f"Embedding metadata mismatch in {name}: " + "; ".join(mismatches))


def validate_runtime_embedding_contract(
    *,
    paths: Any,
    artifact_report: ArtifactVerificationReport,
    feature_config: TensorFlowFeatureConfig,
    expected_embedding_model: str | None = None,
) -> EmbeddingModelMetadata:
    """Validate artifact-declared embedding model across runtime metadata."""

    sources: list[tuple[str, EmbeddingModelMetadata]] = []
    tf_payload = load_json(paths.tensorflow_feature_config_path)
    feature_payload = load_json(paths.feature_config_path)
    model_card_payload = load_json(paths.model_card_path)

    for name, metadata in (
        ("tensorflow_feature_config.json", _metadata_from_payload(tf_payload)),
        ("feature_config.json", _metadata_from_payload(feature_payload)),
        ("model_card.json", _metadata_from_payload(model_card_payload)),
        ("artifact_manifest.json", _manifest_embedding_metadata(artifact_report)),
    ):
        if metadata is not None:
            sources.append((name, metadata))

    if artifact_report.manifest.phase_id != "phase_25_tensorflow_training_delivery" and not any(
        source in {"tensorflow_feature_config.json", "feature_config.json"} for source, _metadata in sources
    ):
        raise ArtifactError("Embedding metadata missing from tensorflow_feature_config.json or feature_config.json")
    if not sources:
        sources.append(("tensorflow_feature_config.json", feature_config.embedding_metadata))

    declared = sources[0][1]
    for name, metadata in sources[1:]:
        _assert_same_embedding_metadata(name, metadata, declared)
    if expected_embedding_model and declared.embedding_model != expected_embedding_model:
        raise ArtifactError(
            "MODEL_API_EXPECTED_EMBEDDING_MODEL mismatch: "
            f"expected={expected_embedding_model!r} actual={declared.embedding_model!r}"
        )
    _assert_same_embedding_metadata("loaded TensorFlowFeatureConfig", feature_config.embedding_metadata, declared)
    return declared


def _runtime_state_payload(state: RuntimeState) -> dict[str, object]:
    return {
        "ready": state.ready,
        "message": state.message,
        "artifactManifestPhase": state.artifact_manifest_phase,
        "loadedAt": state.loaded_at,
        "errorCode": state.error_code,
        "model": _model_identity_payload(state.model_identity, include_artifact=True),
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


def _score_signal_text(signal: ScoreSignal) -> str:
    if signal.label and signal.value is not None:
        return f"{signal.label}: {signal.value:.3f}"
    if signal.label:
        return signal.label
    if signal.value is not None:
        return f"{signal.key}: {signal.value:.3f}"
    return signal.key


def _recommendation_payload(recommendation) -> dict[str, object]:
    return {
        "jobId": recommendation.jobId,
        "matchScore": recommendation.matchScore,
        "matchLevel": recommendation.matchLevel,
        "rankingSignals": [_score_signal_text(signal) for signal in recommendation.rankingSignals],
        "matchedSkills": list(recommendation.matchedSkills),
        "missingSkills": list(recommendation.missingSkills),
    }


def _candidate_skill_evidence(profile: SanitizedProfileInput, candidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile_skills = normalized_skill_set(profile.normalizedSkills)
    candidate_skills = normalized_skill_set((*candidate.model_scoring_input.requiredSkills, *candidate.model_scoring_input.requirements))
    matched = tuple(sorted(profile_skills & candidate_skills))
    missing = tuple(sorted(candidate_skills - profile_skills))
    return matched, missing


def _recommendation_payloads_with_evidence(recommendations, profile: SanitizedProfileInput, candidates) -> list[dict[str, object]]:
    candidates_by_id = {candidate.jobId: candidate for candidate in candidates}
    payloads: list[dict[str, object]] = []
    for recommendation in recommendations:
        payload = _recommendation_payload(recommendation)
        candidate = candidates_by_id.get(recommendation.jobId)
        if candidate is not None:
            matched, missing = _candidate_skill_evidence(profile, candidate)
            payload["matchedSkills"] = list(matched)
            payload["missingSkills"] = list(missing)
        payloads.append(payload)
    return payloads


def _latency_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _attach_observability(payload: dict[str, object], **values: object) -> None:
    event = build_safe_observability_event(**values)
    if event:
        payload["observability"] = event


def _warmup_cv_analysis_request() -> CvAnalysisModelCoreRequest:
    return parse_cv_analysis_model_core_request(
        {
            "requestId": "warmup_cv_analysis_staging_fixture",
            "inputVersion": MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
            "language": "en",
            "inputMode": "UPLOAD",
            "compareSource": "JOB_SEARCH",
            "profile": {
                "cvText": "Summary Backend engineer. Skills Python SQL REST APIs. Experience 2020 to 2024.",
                "profileText": "Backend engineer building REST APIs with Python and SQL.",
                "targetRoles": ["Backend Engineer"],
                "normalizedSkills": ["python", "sql", "rest api"],
                "detectedCvSectionNames": ["summary", "skills", "experience"],
            },
            "jobCandidates": [
                {
                    "jobId": "warmup-job-backend-engineer",
                    "scoringInput": {
                        "titleText": "Backend Engineer",
                        "requirementSummary": "Build REST APIs using Python and SQL.",
                        "requiredSkills": ["python", "sql", "rest api"],
                        "roleFamily": "backend",
                    },
                }
            ],
            "rankingPolicy": {
                "maxRecommendations": 1,
                "requireCandidateJobIds": True,
                "deduplicateByJobId": True,
                "backendOwnsHydration": True,
            },
            "maxRecommendations": 1,
        }
    )


def _run_cv_analysis_warmup(
    *,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend,
    environment: str,
    timeout_ms: int | None,
) -> dict[str, object]:
    started_at = perf_counter()
    payload = build_cv_analysis_response_payload(
        _warmup_cv_analysis_request(),
        service=service,
        feature_config=feature_config,
        calibration_policy=calibration_policy,
        embedding_backend=embedding_backend,
        environment=environment,
        timeout_ms=timeout_ms,
        include_observability=True,
    )
    return {
        "completed": True,
        "latencyMs": _latency_ms(started_at),
        "modelVersion": (payload.get("model") or {}).get("version") if isinstance(payload.get("model"), dict) else None,
        "observability": payload.get("observability", {}),
    }


def build_cv_analysis_response_payload(
    request: CvAnalysisModelCoreRequest,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
    timeout_ms: int | None = None,
    include_observability: bool = False,
) -> dict[str, object]:
    """Build validated model-core CV-analysis response for HTTP route/tests."""

    state = service.state
    if state.model_identity is None:
        service.require_ready()
        state = service.state
    if state.model_identity is None:
        raise ModelNotReadyError("TensorFlow model identity is not available")

    total_started_at = perf_counter()
    embedding_started_at = perf_counter()
    candidate_vectors = build_feature_vectors_for_request(
        request,
        feature_config=feature_config,
        embedding_backend=embedding_backend,
        environment=environment,
    )
    embedding_latency_ms = _latency_ms(embedding_started_at)
    tensorflow_started_at = perf_counter()
    recommendations = service.predict_recommendations(
        tuple(vector.normalized for vector in candidate_vectors),
        max_recommendations=request.maxRecommendations,
        calibration_policy=calibration_policy,
        timeout_ms=timeout_ms,
    )
    tensorflow_latency_ms = _latency_ms(tensorflow_started_at)
    top_score = recommendations[0].matchScore if recommendations else 0
    top_candidate_by_id = {candidate.jobId: candidate for candidate in request.jobCandidates}
    top_candidate = top_candidate_by_id.get(recommendations[0].jobId) if recommendations else None
    matched_skills, missing_skills = _candidate_skill_evidence(request.profile, top_candidate) if top_candidate is not None else ((), ())
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
    detected_sections = list(request.profile.detectedCvSectionNames)
    parse_quality = "high" if detected_sections else ("medium" if request.profile.cvText.strip() else "failed")
    payload: dict[str, object] = {
        "schemaVersion": MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
        "parsedCv": {
            "status": "parsed" if request.profile.cvText.strip() else "empty_text",
            "pageCount": 0,
            "textLength": len(request.profile.cvText),
            "detectedSections": detected_sections,
            "extractionEvidence": profile_evidence_keys,
        },
        "jobFitAlignment": {
            "score": job_fit.score,
            "matchedSignals": [_score_signal_text(signal) for signal in job_fit.summarySignals],
            "missingSignals": list(job_fit.confidenceNotes),
            "matchedSkills": list(job_fit.matchedSkills),
            "missingSkills": list(job_fit.missingSkills),
            "evidence": ["top candidate model-core score", *profile_evidence_keys],
        },
        "atsFriendliness": {
            "score": ats.score,
            "detectedIssues": list(ats.detectedIssues),
            "parseQuality": parse_quality,
            "evidence": [str(value) for value in ats.evidence.get("evidenceKeys", profile_evidence_keys)] if isinstance(ats.evidence, dict) else profile_evidence_keys,
        },
        "overallImpression": {
            "score": overall.score,
            "evidence": [*overall.evidenceKeys, *overall.confidenceNotes],
        },
        "candidateReranking": {
            "recommendations": _recommendation_payloads_with_evidence(recommendations, request.profile, request.jobCandidates),
        },
        "model": _model_identity_payload(state.model_identity),
        "createdAt": utc_now_iso(),
    }
    if include_observability:
        _attach_observability(
            payload,
            requestId=request.requestId,
            modelVersion=state.model_identity.version,
            artifactHash=state.model_identity.artifact_sha256,
            candidateCount=len(request.jobCandidates),
            parseQuality=parse_quality,
            parseLatencyMs=0,
            embeddingLatencyMs=embedding_latency_ms,
            tensorflowLatencyMs=tensorflow_latency_ms,
            wrapperLatencyMs=0,
            totalLatencyMs=_latency_ms(total_started_at),
        )
    validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
    return payload


def build_candidate_reranking_response_payload(
    request: CandidateRerankingCoreRequest,
    service: InferenceService,
    feature_config: TensorFlowFeatureConfig,
    calibration_policy: ScoreCalibrationPolicy,
    embedding_backend: TextEmbeddingBackend | None = None,
    environment: str = "local",
    timeout_ms: int | None = None,
) -> dict[str, object]:
    state = service.state
    if state.model_identity is None:
        service.require_ready()
        state = service.state
    if state.model_identity is None:
        raise ModelNotReadyError("TensorFlow model identity is not available")

    cv_request = CvAnalysisModelCoreRequest(
        requestId=request.requestId,
        inputVersion=MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
        language=request.language,
        inputMode="UPLOAD",
        compareSource="JOB_SEARCH",
        profile=request.profileFeatures,
        jobCandidates=request.jobCandidates,
        maxRecommendations=request.maxRecommendations,
        rankingPolicy=request.rankingPolicy,
    )
    total_started_at = perf_counter()
    embedding_started_at = perf_counter()
    candidate_vectors = build_feature_vectors_for_request(
        cv_request,
        feature_config=feature_config,
        embedding_backend=embedding_backend,
        environment=environment,
    )
    embedding_latency_ms = _latency_ms(embedding_started_at)
    tensorflow_started_at = perf_counter()
    recommendations = service.predict_recommendations(
        tuple(vector.normalized for vector in candidate_vectors),
        max_recommendations=request.maxRecommendations,
        calibration_policy=calibration_policy,
        timeout_ms=timeout_ms,
    )
    tensorflow_latency_ms = _latency_ms(tensorflow_started_at)
    payload: dict[str, object] = {
        "requestId": request.requestId,
        "schemaVersion": MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
        "candidateSetId": request.candidateSetId,
        "language": request.language,
        "recommendations": _recommendation_payloads_with_evidence(recommendations, request.profileFeatures, request.jobCandidates),
        "model": _model_identity_payload(state.model_identity),
        "rankedAt": utc_now_iso(),
    }
    _attach_observability(
        payload,
        requestId=request.requestId,
        modelVersion=state.model_identity.version,
        artifactHash=state.model_identity.artifact_sha256,
        candidateCount=len(request.jobCandidates),
        parseQuality="preparsed_profile_features",
        parseLatencyMs=0,
        embeddingLatencyMs=embedding_latency_ms,
        tensorflowLatencyMs=tensorflow_latency_ms,
        wrapperLatencyMs=0,
        totalLatencyMs=_latency_ms(total_started_at),
    )
    validate_model_core_payload(payload, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
    return payload


def _auth_error() -> dict[str, object]:
    return {
        "success": False,
        "message": "Unauthorized Model API request",
        "data": None,
        "error": {"code": "MODEL_API_UNAUTHORIZED", "details": ["valid bearer service token required"]},
    }


def _authorize_internal_request(headers: Any, config: RuntimeConfig) -> dict[str, object] | None:
    if config.environment.lower() in {"local", "test"} and config.allow_unauthenticated_local and not config.service_token:
        return None
    expected = config.service_token
    authorization = headers.get("authorization") if hasattr(headers, "get") else None
    if not expected or not isinstance(authorization, str):
        return _auth_error()
    expected_header = f"Bearer {expected}"
    if not compare_digest(authorization, expected_header):
        return _auth_error()
    return None


def _json_form_value(value: str | None, field_name: str) -> Any:
    if value is None or not str(value).strip():
        if field_name == "rankingPolicy":
            return None
        raise ContractValidationError([f"$.{field_name} is required"])
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractValidationError([f"$.{field_name} must be valid JSON: {exc.msg}"]) from exc


def _form_string_values(form: Any, field_name: str) -> list[str]:
    if hasattr(form, "getlist"):
        values = list(form.getlist(field_name))
    else:
        value = form.get(field_name) if hasattr(form, "get") else None
        values = value if isinstance(value, list) else ([] if value is None else [value])
    return [value for value in values if isinstance(value, str)]


def _job_roles_from_multipart(form: Any) -> list[str]:
    raw_values = _form_string_values(form, "jobRoles")
    if len(raw_values) == 1 and raw_values[0].lstrip().startswith("["):
        parsed = _json_form_value(raw_values[0], "jobRoles")
        raw_values = parsed if isinstance(parsed, list) else []
    roles: list[str] = []
    errors: list[str] = []
    for index, value in enumerate(raw_values):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"$.jobRoles[{index}] must be non-empty string")
            continue
        role = value.strip()
        if len(role) > MAX_JOB_ROLE_CHARS:
            errors.append(f"$.jobRoles[{index}] max length {MAX_JOB_ROLE_CHARS}; actual={len(role)}")
        roles.append(role)
    if not roles:
        errors.append("$.jobRoles must contain at least 1 role")
    if len(roles) > MAX_JOB_ROLES:
        errors.append(f"$.jobRoles max items {MAX_JOB_ROLES}; actual={len(roles)}")
    if errors:
        raise ContractValidationError(errors)
    return roles


def _candidate_requirement_hints(values: Any) -> list[str]:
    hints: list[str] = []
    if not isinstance(values, list):
        return hints
    for value in values:
        if isinstance(value, str):
            hints.append(value)
        elif isinstance(value, dict) and isinstance(value.get("value"), str):
            hints.append(value["value"])
    return hints


def _validate_pdf_upload_bytes(pdf_bytes: bytes, runtime_config: RuntimeConfig) -> None:
    if len(pdf_bytes) > runtime_config.max_pdf_bytes:
        raise ContractValidationError(["cvFile exceeds MODEL_API_MAX_PDF_BYTES"])
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ContractValidationError(["cvFile must start with PDF magic bytes"])


def _build_cv_payload_from_multipart(form: Any, pdf_bytes: bytes, runtime_config: RuntimeConfig) -> dict[str, object]:
    _validate_pdf_upload_bytes(pdf_bytes, runtime_config)
    job_candidates = _json_form_value(form.get("jobCandidates"), "jobCandidates")
    ranking_policy = _json_form_value(form.get("rankingPolicy"), "rankingPolicy")
    candidate_skill_hints: list[str] = []
    if isinstance(job_candidates, list):
        for candidate in job_candidates:
            if not isinstance(candidate, dict):
                continue
            scoring = candidate.get("scoringInput") if isinstance(candidate.get("scoringInput"), dict) else candidate
            if not isinstance(scoring, dict):
                continue
            required_skills = scoring.get("requiredSkills")
            if isinstance(required_skills, list):
                candidate_skill_hints.extend(str(value) for value in required_skills if isinstance(value, str))
            candidate_skill_hints.extend(_candidate_requirement_hints(scoring.get("requirements")))

    parse_started_at = perf_counter()
    parsed_pdf = parse_pdf_bytes(pdf_bytes, max_bytes=runtime_config.max_pdf_bytes, max_pages=runtime_config.max_pdf_pages)
    if not parsed_pdf.text.strip():
        raise ContractValidationError(["cvFile has no extractable PDF text"])
    parse_latency_ms = _latency_ms(parse_started_at)
    ats_score, detected_issues, ats_fallback = ats_score_from_pdf_evidence(parsed_pdf)
    profile = {
        "cvText": parsed_pdf.text,
        "profileText": parsed_pdf.text[:2000],
        "targetRoles": _job_roles_from_multipart(form),
        "normalizedSkills": list(normalized_skills_from_text(parsed_pdf.text, candidate_skill_hints)),
        "detectedCvSectionNames": list(parsed_pdf.section_names),
    }
    payload: dict[str, object] = {
        "requestId": str(form.get("requestId") or ""),
        "inputVersion": MODEL_CORE_CV_ANALYZER_INPUT_VERSION,
        "language": str(form.get("language") or ""),
        "inputMode": str(form.get("inputMode") or "UPLOAD"),
        "compareSource": str(form.get("compareSource") or ""),
        "profile": profile,
        "jobCandidates": job_candidates,
        "rankingPolicy": ranking_policy,
        "maxRecommendations": (ranking_policy or {}).get("maxRecommendations", 5) if isinstance(ranking_policy, dict) else 5,
    }
    payload["_parsedPdfEvidence"] = {
        "pageCount": parsed_pdf.page_count,
        "parseQuality": parsed_pdf.parse_quality,
        "atsScore": ats_score,
        "detectedIssues": list(detected_issues),
        "fallback": ats_fallback,
        "parseLatencyMs": parse_latency_ms,
    }
    return payload


def create_app(
    config: RuntimeConfig | None = None,
    service: InferenceService | None = None,
    embedding_backend: TextEmbeddingBackend | None = None,
):
    """Create HTTP app, verify artifacts, and load model in the background during lifespan startup.

    Raises a clear error when FastAPI is not installed instead of failing during
    package import.
    """

    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError("FastAPI dependency missing; install serving requirements after Step 26.13") from exc

    globals()["Request"] = Request
    runtime_config = config or RuntimeConfig.from_env()
    runtime_config.validate_security()
    artifact_report = verify_runtime_artifacts(runtime_config.artifact_paths)
    feature_config = TensorFlowFeatureConfig.from_path(runtime_config.artifact_paths.tensorflow_feature_config_path)
    embedding_contract = validate_runtime_embedding_contract(
        paths=runtime_config.artifact_paths,
        artifact_report=artifact_report,
        feature_config=feature_config,
        expected_embedding_model=runtime_config.expected_embedding_model,
    )
    calibration_policy = ScoreCalibrationPolicy.from_path(runtime_config.artifact_paths.score_calibration_path)
    inference_service = service or InferenceService()
    runtime_embedding_backend = embedding_backend or SentenceTransformerE5Embedder(embedding_contract.embedding_model)
    validate_e5_backend(runtime_embedding_backend, runtime_config.environment, expected_model_name=embedding_contract.embedding_model)
    warmup_state: dict[str, object] = {"completed": False, "latencyMs": None, "error": None}

    def _load_runtime_once() -> RuntimeState:
        state = inference_service.load_once(runtime_config.artifact_paths, artifact_report)
        if runtime_config.warmup_on_startup and state.ready:
            try:
                warmup_state.update(
                    _run_cv_analysis_warmup(
                        service=inference_service,
                        feature_config=feature_config,
                        calibration_policy=calibration_policy,
                        embedding_backend=runtime_embedding_backend,
                        environment=runtime_config.environment,
                        timeout_ms=runtime_config.timeout_ms,
                    )
                )
            except Exception as exc:  # pragma: no cover - depends on live TensorFlow/E5 runtime
                warmup_state.update({"completed": False, "error": repr(exc)})
        return inference_service.state

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.artifact_report = artifact_report
        app.state.inference_service = inference_service
        app.state.feature_config = feature_config
        app.state.calibration_policy = calibration_policy
        app.state.embedding_contract = embedding_contract
        load_task = asyncio.create_task(asyncio.to_thread(_load_runtime_once))
        app.state.model_load_task = load_task
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

    @app.get("/live")
    def live() -> dict[str, object]:
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "live": True,
            "message": "alive",
        }

    @app.get("/")
    def root() -> dict[str, object]:
        state = inference_service.state
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "message": "Bisakerja Model API is running",
            "ready": state.ready,
            "status": state.message,
            "endpoints": {
                "live": "/live",
                "health": "/health",
                "ready": "/ready",
                "modelInfo": "/model-info",
                "cvAnalysis": "/inference/cv-analysis",
            },
        }

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

    @app.get("/ready")
    def ready() -> dict[str, object]:
        state = inference_service.state
        identity = state.model_identity
        checks = {
            "artifactsVerified": bool(artifact_report.artifact_hashes),
            "tensorflowModelLoaded": state.ready and identity is not None,
            "embeddingModelDeclared": bool(embedding_contract.embedding_model),
            "embeddingModelMatchesBackend": getattr(runtime_embedding_backend, "model_name", "") == embedding_contract.embedding_model,
            "e5BackendConfigured": getattr(runtime_embedding_backend, "backend_name", "") != "local-hash",
            "pdfParserAvailable": callable(parse_pdf_bytes),
            "serviceTokenConfigured": (not runtime_config.requires_service_token) or bool(runtime_config.service_token),
            "warmupCompleted": (not runtime_config.warmup_required) or bool(warmup_state.get("completed")),
        }
        return {
            "service": runtime_config.service_name,
            "environment": runtime_config.environment,
            "ready": all(checks.values()),
            "checks": checks,
            "warmup": {
                "required": runtime_config.warmup_required,
                "onStartup": runtime_config.warmup_on_startup,
                "completed": bool(warmup_state.get("completed")),
                "latencyMs": warmup_state.get("latencyMs"),
                "error": warmup_state.get("error"),
            },
            "modelVersion": None if identity is None else identity.version,
            "embeddingModel": embedding_contract.embedding_model,
            "artifactPhase": artifact_report.manifest.phase_id,
            "artifactHash": None if identity is None else identity.artifact_sha256,
        }

    @app.get("/model-info")
    def model_info(http_request: Request) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        state = inference_service.state
        return {
            "ready": state.ready,
            "readiness": _runtime_state_payload(state),
            "model": _model_identity_payload(state.model_identity, include_artifact=True),
            "artifacts": runtime_config.artifact_paths.as_dict(),
            "artifactVerification": _artifact_verification_payload(artifact_report),
            "embeddingPolicy": embedding_contract.as_dict(),
            "openrouter": {
                "baseUrl": runtime_config.openrouter.base_url,
                "modelsUrl": runtime_config.openrouter.models_url,
                "model": runtime_config.openrouter.model,
                "enabled": runtime_config.openrouter.enabled,
            },
        }

    @app.post("/inference/cv-analysis")
    def cv_analysis(http_request: Request, payload: Any = Body(...)) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        request = parse_cv_analysis_model_core_request(payload)
        include_observability = http_request.headers.get("x-model-api-include-observability") == "true"
        data = build_cv_analysis_response_payload(
            request,
            service=inference_service,
            feature_config=feature_config,
            calibration_policy=calibration_policy,
            embedding_backend=runtime_embedding_backend,
            environment=runtime_config.environment,
            timeout_ms=runtime_config.timeout_ms,
            include_observability=include_observability,
        )
        observability = data.get("observability") if isinstance(data, dict) else None
        warmup_state.update(
            {
                "completed": True,
                "latencyMs": observability.get("totalLatencyMs") if isinstance(observability, dict) else None,
                "error": None,
            }
        )
        return {"success": True, "message": "Model-core inference completed", "data": data, "error": None}

    @app.post("/inference/candidate-reranking")
    def candidate_reranking(http_request: Request, payload: Any = Body(...)) -> dict[str, object]:
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        request = parse_candidate_reranking_core_request(payload)
        data = build_candidate_reranking_response_payload(
            request,
            service=inference_service,
            feature_config=feature_config,
            calibration_policy=calibration_policy,
            embedding_backend=runtime_embedding_backend,
            environment=runtime_config.environment,
            timeout_ms=runtime_config.timeout_ms,
        )
        return {"success": True, "message": "Model-core candidate reranking completed", "data": data, "error": None}

    @app.post("/internal/model/cv-analysis")
    async def internal_model_cv_analysis(http_request: Request):
        auth_error = _authorize_internal_request(http_request.headers, runtime_config)
        if auth_error is not None:
            return JSONResponse(status_code=401, content=auth_error)
        content_type = http_request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise ContractValidationError(["Content-Type must be multipart/form-data"])
        form = await http_request.form()
        file_values = [value for value in form.values() if hasattr(value, "filename") and hasattr(value, "read")]
        if len(file_values) != 1 or "cvFile" not in form:
            raise ContractValidationError(["multipart request must include exactly one cvFile"])
        upload = form["cvFile"]
        if getattr(upload, "content_type", None) not in {"application/pdf", "application/octet-stream"}:
            raise ContractValidationError(["cvFile content type must be application/pdf"])
        pdf_bytes = await upload.read()
        _validate_pdf_upload_bytes(pdf_bytes, runtime_config)
        payload = _build_cv_payload_from_multipart(form, pdf_bytes, runtime_config)
        parsed_pdf_evidence = payload.pop("_parsedPdfEvidence")
        request = parse_cv_analysis_model_core_request(payload)
        include_observability = http_request.headers.get("x-model-api-include-observability") == "true"
        data = build_cv_analysis_response_payload(
            request,
            service=inference_service,
            feature_config=feature_config,
            calibration_policy=calibration_policy,
            embedding_backend=runtime_embedding_backend,
            environment=runtime_config.environment,
            timeout_ms=runtime_config.timeout_ms,
            include_observability=include_observability,
        )
        observability = data.get("observability") if isinstance(data, dict) else None
        warmup_state.update(
            {
                "completed": True,
                "latencyMs": observability.get("totalLatencyMs") if isinstance(observability, dict) else None,
                "error": None,
            }
        )
        if isinstance(data.get("atsFriendliness"), dict) and isinstance(parsed_pdf_evidence, dict):
            parse_quality = parsed_pdf_evidence["parseQuality"]
            data["parsedCv"] = {
                "status": "parsed" if request.profile.cvText.strip() else "empty_text",
                "textLength": len(request.profile.cvText),
                "pageCount": parsed_pdf_evidence["pageCount"],
                "detectedSections": list(request.profile.detectedCvSectionNames),
                "extractionEvidence": ["deterministic_pdf_parser", f"parseLatencyMs={parsed_pdf_evidence['parseLatencyMs']}"],
            }
            data["atsFriendliness"]["score"] = parsed_pdf_evidence["atsScore"]
            data["atsFriendliness"]["detectedIssues"] = parsed_pdf_evidence["detectedIssues"]
            data["atsFriendliness"]["parseQuality"] = parse_quality if parse_quality in {"high", "medium", "low", "failed"} else "low"
            data["atsFriendliness"]["evidence"] = ["deterministic_pdf_parser"]
        validate_model_core_payload(data, {candidate.jobId for candidate in request.jobCandidates}, request.maxRecommendations)
        return data

    return app
