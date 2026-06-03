"""Response contract validation for model-core outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ContractValidationError
from .schemas import (
    ALLOWED_LANGUAGES,
    ALLOWED_MATCH_LEVELS,
    MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION,
    MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION,
)

FORBIDDEN_MODEL_CORE_FIELDS: frozenset[str] = frozenset(
    {
        "auth",
        "availability",
        "company",
        "companyName",
        "cvFile",
        "cvFileId",
        "database",
        "db",
        "experienceLevel",
        "generatedCv",
        "hasApplied",
        "hydratedJob",
        "hydratedJobData",
        "id",
        "isBookmarked",
        "jobDetail",
        "jobDetails",
        "jobRecommendation",
        "jobRecommendations",
        "location",
        "nextStep",
        "nextSteps",
        "persistence",
        "reason",
        "sectionReviews",
        "title",
        "topActionables",
        "userId",
        "visibility",
        "workType",
    }
)

DEFAULT_MAX_RECOMMENDATIONS = 5


def ensure_score_0_100(value: Any, path: str) -> list[str]:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 100:
        return [f"{path} must be integer 0-100; actual={value!r}"]
    return []


def find_forbidden_fields(payload: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_MODEL_CORE_FIELDS:
                errors.append(f"{child_path} is wrapper/backend-owned")
            errors.extend(find_forbidden_fields(value, child_path))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            errors.extend(find_forbidden_fields(item, f"{path}[{index}]"))
    return errors


def validate_candidate_recommendations(
    recommendations: Any,
    candidate_ids: set[str],
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS,
    path: str = "recommendations",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(recommendations, Sequence) or isinstance(recommendations, (str, bytes, bytearray)):
        return [f"{path} must be array"]
    seen: set[str] = set()
    if len(recommendations) > max_recommendations:
        errors.append(f"{path} max items {max_recommendations}; actual={len(recommendations)}")
    for index, recommendation in enumerate(recommendations):
        item_path = f"{path}[{index}]"
        if not isinstance(recommendation, Mapping):
            errors.append(f"{item_path} must be object")
            continue
        job_id_value = recommendation.get("jobId", recommendation.get("job_id", ""))
        job_id = job_id_value if isinstance(job_id_value, str) else ""
        if not job_id:
            errors.append(f"{item_path}.jobId is required")
            continue
        if job_id not in candidate_ids:
            errors.append(f"{item_path}.jobId not in candidate set: {job_id!r}")
        if job_id in seen:
            errors.append(f"{item_path}.jobId duplicate: {job_id!r}")
        seen.add(job_id)
        score = recommendation.get("matchScore", recommendation.get("match_score"))
        errors.extend(ensure_score_0_100(score, f"{item_path}.matchScore"))
        match_level = recommendation.get("matchLevel", recommendation.get("match_level"))
        if match_level is not None and match_level not in ALLOWED_MATCH_LEVELS:
            errors.append(f"{item_path}.matchLevel must be one of {sorted(ALLOWED_MATCH_LEVELS)}")
    return errors


def _validate_nested_candidate_reranking(
    payload: Mapping[str, Any],
    parent_language: Any,
    candidate_ids: set[str],
    max_recommendations: int,
    path: str,
) -> list[str]:
    errors: list[str] = []
    schema_version = payload.get("schemaVersion")
    if schema_version != MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION:
        errors.append(f"{path}.schemaVersion must be {MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION!r}")
    language = payload.get("language")
    if language not in ALLOWED_LANGUAGES:
        errors.append(f"{path}.language must be one of {sorted(ALLOWED_LANGUAGES)}")
    elif parent_language in ALLOWED_LANGUAGES and language != parent_language:
        errors.append(f"{path}.language must match $.language")
    errors.extend(
        validate_candidate_recommendations(
            payload.get("recommendations", []),
            candidate_ids,
            max_recommendations,
            path=f"{path}.recommendations",
        )
    )
    return errors


def validate_model_core_payload(
    payload: Mapping[str, Any],
    candidate_ids: set[str],
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS,
) -> None:
    """Validate common model-core handoff rules; raise deterministic error list."""

    if not isinstance(payload, Mapping):
        raise ContractValidationError(["$ must be object"])

    errors = find_forbidden_fields(payload)
    schema_version = payload.get("schemaVersion")
    if schema_version not in {MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION, MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION}:
        errors.append(
            "schemaVersion must be model-core schema; "
            f"expected one of {[MODEL_CORE_CV_ANALYSIS_SCHEMA_VERSION, MODEL_CORE_CANDIDATE_RERANKING_SCHEMA_VERSION]}"
        )
    language = payload.get("language")
    if language not in ALLOWED_LANGUAGES:
        errors.append(f"language must be one of {sorted(ALLOWED_LANGUAGES)}")
    if isinstance(payload.get("jobFitAlignment"), Mapping):
        errors.extend(ensure_score_0_100(payload["jobFitAlignment"].get("score"), "$.jobFitAlignment.score"))
    if isinstance(payload.get("atsFriendliness"), Mapping):
        errors.extend(ensure_score_0_100(payload["atsFriendliness"].get("score"), "$.atsFriendliness.score"))
    if isinstance(payload.get("overallImpression"), Mapping):
        errors.extend(ensure_score_0_100(payload["overallImpression"].get("score"), "$.overallImpression.score"))
    candidate_reranking = payload.get("candidateReranking")
    if isinstance(candidate_reranking, Mapping):
        errors.extend(
            _validate_nested_candidate_reranking(
                candidate_reranking,
                language,
                candidate_ids,
                max_recommendations,
                path="$.candidateReranking",
            )
        )
    elif candidate_reranking is not None:
        errors.append("$.candidateReranking must be object")

    recommendations = payload.get("recommendations")
    if recommendations is not None:
        errors.extend(validate_candidate_recommendations(recommendations, candidate_ids, max_recommendations))
    if errors:
        raise ContractValidationError(errors)
