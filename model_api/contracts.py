"""Canonical backend contract readers for Phase 26 layout.

This module reads reference artifacts only. It does not connect to the backend DB
or import backend TypeScript code. Runtime schema validation is added in later
Phase 26 steps; these helpers keep request/response constants tied to OpenAPI
and Prisma references from day one.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .config import DEFAULT_OPENAPI_PATH, DEFAULT_PRISMA_SCHEMA_PATH
from .errors import ContractReferenceError


@dataclass(frozen=True)
class BackendOpenApiContract:
    """Subset of backend OpenAPI needed by Model API boundary."""

    source_path: Path
    api_version: str
    cv_analyzer_path: str
    analyze_cv_language_enum: tuple[str, ...]
    analyze_cv_input_mode_enum: tuple[str, ...]
    analyze_cv_compare_source_enum: tuple[str, ...]
    cv_analysis_schema_version: str
    cv_analysis_job_recommendations_max_items: int
    top_actionables_max_items: int
    top_actionables_min_items: int
    backend_response_required_fields: tuple[str, ...]


@dataclass(frozen=True)
class PrismaCvAnalysisContract:
    """DB enums/models relevant to backend-owned CV analysis persistence."""

    source_path: Path
    analysis_language_enum: tuple[str, ...]
    cv_input_mode_enum: tuple[str, ...]
    cv_compare_source_enum: tuple[str, ...]
    job_recommendation_match_level_enum: tuple[str, ...]
    cv_analysis_result_json_fields: tuple[str, ...]
    job_recommendation_item_json_fields: tuple[str, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractReferenceError(f"Contract reference not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractReferenceError(f"Contract reference JSON invalid: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractReferenceError(f"Contract reference root must be object: {path}")
    return raw


def _schema(openapi: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        schema = openapi["components"]["schemas"][name]
    except KeyError as exc:
        raise ContractReferenceError(f"OpenAPI schema missing: {name}") from exc
    if not isinstance(schema, dict):
        raise ContractReferenceError(f"OpenAPI schema must be object: {name}")
    return schema


def _required_tuple(schema: dict[str, Any], path: str) -> tuple[str, ...]:
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ContractReferenceError(f"OpenAPI required must be list: {path}")
    return tuple(str(item) for item in required)


def _enum_tuple(schema: dict[str, Any], property_name: str, path: str) -> tuple[str, ...]:
    try:
        enum_values = schema["properties"][property_name]["enum"]
    except KeyError as exc:
        raise ContractReferenceError(f"OpenAPI enum missing: {path}.{property_name}") from exc
    if not isinstance(enum_values, list):
        raise ContractReferenceError(f"OpenAPI enum must be list: {path}.{property_name}")
    return tuple(str(item) for item in enum_values)


def load_backend_openapi_contract(path: Path = DEFAULT_OPENAPI_PATH) -> BackendOpenApiContract:
    """Load CV Analyzer payload/response expectations from generated OpenAPI."""

    openapi = _read_json_object(path)
    analyze_cv = _schema(openapi, "AnalyzeCvMultipartRequest")
    cv_analysis = _schema(openapi, "CvAnalysis")
    analysis_result = cv_analysis["properties"]["analysisResult"]
    analysis_properties = analysis_result["properties"]
    job_recommendations = analysis_properties["jobRecommendations"]
    top_actionables = analysis_properties["topActionables"]
    schema_version = analysis_properties["schemaVersion"].get("const")
    if not schema_version:
        raise ContractReferenceError("OpenAPI CvAnalysis.analysisResult.schemaVersion const missing")
    cv_analyzer_path = "/api/v1/ai/cv-analyzer"
    if cv_analyzer_path not in openapi.get("paths", {}):
        raise ContractReferenceError(f"OpenAPI path missing: {cv_analyzer_path}")
    return BackendOpenApiContract(
        source_path=path,
        api_version=str(openapi.get("openapi", "")),
        cv_analyzer_path=cv_analyzer_path,
        analyze_cv_language_enum=_enum_tuple(analyze_cv, "language", "AnalyzeCvMultipartRequest"),
        analyze_cv_input_mode_enum=_enum_tuple(analyze_cv, "inputMode", "AnalyzeCvMultipartRequest"),
        analyze_cv_compare_source_enum=_enum_tuple(analyze_cv, "compareSource", "AnalyzeCvMultipartRequest"),
        cv_analysis_schema_version=str(schema_version),
        cv_analysis_job_recommendations_max_items=int(job_recommendations.get("maxItems", 5)),
        top_actionables_max_items=int(top_actionables.get("maxItems", 3)),
        top_actionables_min_items=int(top_actionables.get("minItems", 1)),
        backend_response_required_fields=_required_tuple(analysis_result, "CvAnalysis.analysisResult"),
    )


def _extract_enum(schema_text: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"enum\s+{re.escape(name)}\s*\{{(?P<body>.*?)\}}", schema_text, flags=re.S)
    if not match:
        raise ContractReferenceError(f"Prisma enum missing: {name}")
    values: list[str] = []
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        values.append(line.split()[0])
    return tuple(values)


def load_prisma_cv_analysis_contract(path: Path = DEFAULT_PRISMA_SCHEMA_PATH) -> PrismaCvAnalysisContract:
    """Read backend DB enum/storage fields from Prisma schema text."""

    try:
        schema_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractReferenceError(f"Prisma schema not found: {path}") from exc

    return PrismaCvAnalysisContract(
        source_path=path,
        analysis_language_enum=_extract_enum(schema_text, "AnalysisLanguage"),
        cv_input_mode_enum=_extract_enum(schema_text, "CvInputMode"),
        cv_compare_source_enum=_extract_enum(schema_text, "CvCompareSource"),
        job_recommendation_match_level_enum=_extract_enum(schema_text, "JobRecommendationMatchLevel"),
        cv_analysis_result_json_fields=(
            "jobFitAlignment",
            "atsFriendliness",
            "topActionables",
            "sectionReviews",
            "jobRecommendations",
            "inputSummary",
        ),
        job_recommendation_item_json_fields=("reasons", "matchedSkills", "missingSkills", "nextSteps"),
    )
